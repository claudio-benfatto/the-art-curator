"""Crawl the pilot sites' listing pages into `venue_pages`, and purge them (PLAN.md § 8, P2).

In the order `cli crawl` runs it:

1. `purge_pages()` — `raw_text` older than `PAGE_TTL_DAYS` is set NULL. First, every run: the TTL
   has to hold from the first write, not from the day a scheduler exists (CLAUDE.md § 1).
2. `load_targets()` — per site (`seeds.load_sites`), the listing pages to fetch and the hashes
   already stored for them.
3. `crawl_site()` — network only, no database, no model. Each page becomes a `CrawledPage`.
4. `write_pages()` — upsert by URL, in the caller's transaction, and **never commits**
   (`--dry-run` is the real write rolled back, as in `sync-graf`).
5. `report()` — pure.

Rules the code encodes:

- **A pin replaces the machine's seeds.** A venue pinned in `seeds.yaml` is crawled at its pins
  and nowhere else; otherwise at its accepted seeds, minus any URL a person rejected.
- **Only an `ok` page holds text.** `thin`, `blocked`, `robots` and `error` are written as a
  status with `raw_text` and `content_hash` NULL. No headless browser, no challenge bypass.
- **`unavailable` is not an answer**, so it is reported and not written: a site that timed out
  tonight keeps what last night's crawl stored.
- **`content_hash` is the sha256 of the text**, never of the HTML, which carries nonces.
- **A broken seed is flagged, not worked around**: the listing 404s, or now redirects to the
  homepage. The third sign, a listing that yields no items after yielding some, needs extraction
  and lands with it (PR 22).

`Target.details` is the slot for the detail pages extraction will name (PR 22). Nothing fills it
yet; the cap and the listings-first order already apply to it.

`CrawledPage.text` is third-party prose. It goes into `venue_pages.raw_text` and nowhere else: it
is kept out of `repr`, and the report prints a word count, never the words.
"""

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.db.models import VenuePage
from art_curator.ingest.discover import url_key
from art_curator.ingest.http import PoliteClient, RobotsPolicy
from art_curator.ingest.matching import normalize_name
from art_curator.ingest.pages import Fetched, PageStatus, fetch_page
from art_curator.ingest.seeds import SeedOverrides, Site, load_sites

PAGES = VenuePage.__table__

PAGE_TTL_DAYS = 7
MAX_PAGES_PER_SITE = 8
GONE_STATUSES = frozenset({404, 410})


class Change(StrEnum):
    """An `ok` page's text against the hash stored for its URL."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class Broken(StrEnum):
    """Why a listing page can no longer be the listing. The value is what the report prints."""

    GONE = "the page is gone"
    HOMEPAGE = "redirects to the homepage"


# --- targets -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One site's pages for this run. `hashes` is the stored `content_hash` per URL already in
    `venue_pages` (None where the last fetch was not `ok`)."""

    site: Site
    listings: tuple[str, ...] = ()
    details: tuple[str, ...] = ()
    hashes: Mapping[str, str | None] = field(default_factory=dict)

    @property
    def urls(self) -> tuple[str, ...]:
        """Listings first, then detail pages, `MAX_PAGES_PER_SITE` in all."""
        return (*self.listings, *self.details)[:MAX_PAGES_PER_SITE]


async def load_targets(session: AsyncSession, overrides: SeedOverrides) -> list[Target]:
    """One `Target` per crawlable pilot site, including those with no seed yet: the report names
    them, since a site nobody has found a listing for is a gap worth seeing."""
    sites = await load_sites(session, overrides)
    if not sites:
        return []
    stored: dict[int, dict[str, str | None]] = {}
    for venue_id, url, content_hash in await session.execute(
        select(PAGES.c.venue_id, PAGES.c.url, PAGES.c.content_hash).where(
            PAGES.c.venue_id.in_([site.venue_id for site in sites])
        )
    ):
        stored.setdefault(venue_id, {})[url] = content_hash

    targets = []
    for site in sites:
        hashes = stored.get(site.venue_id, {})
        # `/ca/agenda` and `/ca/agenda/` are one page: reuse the spelling already stored.
        spelling = {url_key(url): url for url in hashes}
        listings: dict[str, str] = {}
        for url in site.pinned or site.seeds_in_use:
            listings.setdefault(url_key(url), spelling.get(url_key(url), url))
        targets.append(Target(site, tuple(listings.values()), hashes=hashes))
    return targets


def select_targets(targets: Sequence[Target], venue: str | None) -> list[Target]:
    """`--venue`: the targets whose site matches, compared as `normalize_name`. None means all."""
    if venue is None:
        return list(targets)
    key = normalize_name(venue)
    return [target for target in targets if normalize_name(target.site.name) == key]


# --- one site ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CrawledPage:
    url: str  # as requested: the key the row is upserted on
    status: PageStatus
    http_status: int | None = None
    words: int = 0
    content_hash: str | None = None  # `ok` only
    text: str = field(default="", repr=False)  # `ok` only; third-party prose
    change: Change | None = None  # `ok` only
    broken: Broken | None = None  # listing pages only

    @property
    def detail(self) -> str:
        """What the report says after the URL."""
        if self.status is PageStatus.OK:
            return f"{self.words} words   {self.change}"
        if self.status is PageStatus.THIN:
            return f"{self.words} words"
        return f"HTTP {self.http_status}" if self.http_status else ""


@dataclass(frozen=True)
class SiteCrawl:
    site: Site
    pages: tuple[CrawledPage, ...] = ()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def crawl_site(client: PoliteClient, robots: RobotsPolicy, target: Target) -> SiteCrawl:
    """Fetch every page of `target`. Never raises for something the site did."""
    pages = []
    for url in target.urls:
        fetched = await fetch_page(client, robots, url)
        pages.append(_crawled(fetched, target, is_listing=url in target.listings))
    return SiteCrawl(target.site, tuple(pages))


def _crawled(fetched: Fetched, target: Target, *, is_listing: bool) -> CrawledPage:
    broken = _broken(fetched, target.site) if is_listing else None
    words = len(fetched.text.split())
    if fetched.status is not PageStatus.OK:
        return CrawledPage(fetched.url, fetched.status, fetched.http_status, words, broken=broken)

    content_hash = text_hash(fetched.text)
    stored = target.hashes.get(fetched.url)
    change = (
        Change.NEW
        if stored is None
        else Change.UNCHANGED
        if stored == content_hash
        else Change.CHANGED
    )
    return CrawledPage(
        fetched.url, fetched.status, fetched.http_status, words, content_hash, fetched.text,
        change, broken,
    )  # fmt: skip


def _broken(fetched: Fetched, site: Site) -> Broken | None:
    if fetched.status is PageStatus.ERROR and fetched.http_status in GONE_STATUSES:
        return Broken.GONE
    home = url_key(site.url)
    # A site whose pinned listing *is* its homepage has not been redirected anywhere.
    if fetched.page is not None and url_key(fetched.page.url) == home != url_key(fetched.url):
        return Broken.HOMEPAGE
    return None


# --- venue_pages ---------------------------------------------------------------------------------


async def write_pages(session: AsyncSession, crawls: Sequence[SiteCrawl]) -> int:
    """Upsert every page that gave an answer. Returns the number of rows written. Runs in the
    caller's transaction and never commits."""
    fetched_at = await session.scalar(select(func.now()))
    rows = [
        {
            "venue_id": crawl.site.venue_id,
            "url": page.url,
            "status": page.status.value,
            "http_status": page.http_status,
            "content_hash": page.content_hash,
            "raw_text": page.text or None,
            "fetched_at": fetched_at,
        }
        for crawl in crawls
        for page in crawl.pages
        if page.status is not PageStatus.UNAVAILABLE
    ]
    if not rows:
        return 0
    stmt = insert(PAGES).values(rows)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[PAGES.c.url],
            set_={c: stmt.excluded[c] for c in rows[0] if c != "url"},
        )
    )
    return len(rows)


async def purge_pages(session: AsyncSession) -> int:
    """Set `raw_text` NULL on every page fetched more than `PAGE_TTL_DAYS` ago. Returns how many.
    The row and its `content_hash` stay. The age is computed in Postgres (CLAUDE.md § 8)."""
    result = await session.execute(
        update(PAGES)
        .where(
            PAGES.c.raw_text.is_not(None),
            PAGES.c.fetched_at < func.now() - timedelta(days=PAGE_TTL_DAYS),
        )
        .values(raw_text=None)
    )
    return result.rowcount


# --- the report ----------------------------------------------------------------------------------

BROKEN_ADVICE = "re-run discover, or fix its pin in seeds.yaml"
NO_SEED_ADVICE = "run discover, or pin its listing in seeds.yaml"


def broken_seeds(crawls: Sequence[SiteCrawl]) -> list[tuple[Site, CrawledPage]]:
    return [(crawl.site, page) for crawl in crawls for page in crawl.pages if page.broken]


def report(crawls: Sequence[SiteCrawl]) -> list[str]:
    """What `crawl` prints: counts, then every site with its pages under their status. A `broken`
    line follows a listing page that can no longer be the listing, and sites with nothing to
    fetch come last."""
    lines: list[str] = []

    def block(label: str, *parts: str) -> None:
        lines.append(f"{label:<11} " + "   ".join(part for part in parts if part))

    pages = [page for crawl in crawls for page in crawl.pages]
    statuses = Counter(page.status for page in pages)
    changes = Counter(page.change for page in pages)
    by_name = sorted(crawls, key=lambda crawl: normalize_name(crawl.site.name))
    seedless = [crawl for crawl in by_name if not crawl.pages]

    block(
        "sites",
        str(len(crawls)),
        f"{len(crawls) - len(seedless)} crawled",
        f"{len(seedless)} without a seed" if seedless else "",
    )
    block("pages", str(len(pages)), *(f"{statuses[s]} {s}" for s in PageStatus if statuses[s]))
    if statuses[PageStatus.OK]:
        block("text", *(f"{changes[c]} {c}" for c in Change))
    for crawl in by_name:
        if not crawl.pages:
            continue
        block("site", crawl.site.name)
        for page in crawl.pages:
            block(page.status, page.url, page.detail)
            if page.broken:
                block("broken", page.url, f"{page.broken}: {BROKEN_ADVICE}")
    for crawl in seedless:
        block("no seed", crawl.site.name, NO_SEED_ADVICE)
    return lines
