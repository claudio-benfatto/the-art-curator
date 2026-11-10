"""Everything `cli eval-extraction` does around the scorer (PLAN.md § 8, P2), in the order it
runs:

1. `crawl.purge_pages()`: first, as in every command that reads page text. Text past the TTL is
   never sent to a model (CLAUDE.md § 1).
2. `load_stored()` and `load_verdicts()`: the `ok` pages in `venue_pages`, and what `discover`
   left in `venue_seeds`.
3. `find_texts()`: each confirmed page's text as it was labelled. The stored text if its hash is
   the label's; otherwise a fresh fetch, if *that* hash is the label's; otherwise the page is
   stale. Network, no database.
4. `extract_all()`: model only. Each page goes through `llm/extract.extract_page`, the call
   `cli extract` makes, with the same prompt by page kind.
5. `judge_all()`, with `--judge`: model only. Each page extraction wrote summaries for goes to
   `llm/judge.judge_summaries` with the text they were written from.

**The eval reads the corpus and never changes it.** A page it fetches is held in memory and
dropped; what extraction finds is scored and dropped. The only rows it writes are the `llm_calls`
the client records, and the purge.

`--draft` runs the same extraction over the crawled pages the gold set does not label yet and
writes what it found as unconfirmed labels (`draft_yaml`). A person then checks every item
against the live page: that check is what makes a label gold, so a draft is never scored. A long
listing is drafted as far as its first `DRAFTED_ITEMS` of each kind, marked `partial`: most
listings are archives, and checking 40 past shows tells the eval little that 10 do not.

Page text and link text are third-party prose: they stay out of `repr`, go to the model and are
never printed or written.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import yaml
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import Date, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.db.models import Venue, VenuePage, VenueSeed
from art_curator.eval.extraction import (
    GoldPage,
    GoldSet,
    JudgedPage,
    SeedVerdict,
    Skipped,
    SkipReason,
    summaries,
)
from art_curator.ingest.crawl import text_hash
from art_curator.ingest.discover import collect_links, url_key
from art_curator.ingest.extraction import SITE_TIMEZONE, StoredLink
from art_curator.ingest.http import PoliteClient, RobotsPolicy
from art_curator.ingest.matching import normalize_name
from art_curator.ingest.pages import PageStatus, fetch_page
from art_curator.llm.client import LlmClient
from art_curator.llm.extract import Extraction, ExtractionError, PageLink, extract_page
from art_curator.llm.judge import JudgeError, judge_summaries

PAGES = VenuePage.__table__
VENUES = Venue.__table__
SEEDS = VenueSeed.__table__


# --- the database --------------------------------------------------------------------------------


@dataclass(frozen=True)
class PageText:
    """A page as extraction is handed it."""

    url: str
    venue_name: str
    listing: bool
    content_hash: str
    fetched_on: date  # what a date without a year is read against
    text: str = field(repr=False)  # third-party prose
    links: tuple[PageLink, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class StoredPage:
    """An `ok` row of `venue_pages`. `page` is None once its text is purged."""

    page: PageText | None
    url: str
    venue_name: str
    listing: bool
    content_hash: str


async def today(session: AsyncSession) -> date:
    """Today as the venues name it. Worked out in Postgres, like `fetched_on` (CLAUDE.md § 8)."""
    return await session.scalar(select(cast(func.timezone(SITE_TIMEZONE, func.now()), Date)))


async def load_stored(session: AsyncSession) -> dict[str, StoredPage]:
    """Every `ok` page, by `url_key`, listings first."""
    stored = {}
    for row in await session.execute(
        select(
            PAGES.c.url,
            VENUES.c.name.label("venue_name"),
            PAGES.c.kind,
            PAGES.c.content_hash,
            PAGES.c.raw_text,
            PAGES.c.links,
            cast(func.timezone(SITE_TIMEZONE, PAGES.c.fetched_at), Date).label("fetched_on"),
        )
        .join(VENUES, VENUES.c.id == PAGES.c.venue_id)
        .where(PAGES.c.status == "ok", PAGES.c.content_hash.is_not(None))
        .order_by(case((PAGES.c.kind == "listing", 0), else_=1), PAGES.c.id)
    ):
        listing = row.kind == "listing"
        text = None
        if row.raw_text is not None:
            links = tuple(StoredLink(**link) for link in row.links or ())
            text = PageText(
                row.url, row.venue_name, listing, row.content_hash, row.fetched_on,
                row.raw_text, links,
            )  # fmt: skip
        stored[url_key(row.url)] = StoredPage(
            text, row.url, row.venue_name, listing, row.content_hash
        )
    return stored


async def load_verdicts(session: AsyncSession, venue: str | None = None) -> list[SeedVerdict]:
    """What `discover` left in `venue_seeds`. `venue` keeps only that venue's, compared as
    `normalize_name`, as `GoldSet.for_venue` keeps its listings: otherwise every other venue's
    verdicts would be reported as unlabelled."""
    verdicts = [
        SeedVerdict(**row._mapping)
        for row in await session.execute(
            select(
                VENUES.c.name.label("venue_name"),
                SEEDS.c.url,
                SEEDS.c.page_type,
                SEEDS.c.confidence,
                SEEDS.c.status,
            )
            .join(VENUES, VENUES.c.id == SEEDS.c.venue_id)
            .order_by(SEEDS.c.id)
        )
    ]
    if venue is None:
        return verdicts
    return [v for v in verdicts if normalize_name(v.venue_name) == normalize_name(venue)]


# --- the labelled pages' text --------------------------------------------------------------------


async def find_texts(
    client: PoliteClient,
    robots: RobotsPolicy,
    pages: Sequence[GoldPage],
    stored: Mapping[str, StoredPage],
    fetched_on: date,
) -> tuple[list[tuple[GoldPage, PageText]], list[Skipped]]:
    """Each confirmed page with its text as it was labelled, and the pages that cannot be scored.
    `fetched_on` is today, for a page fetched now."""
    ready: list[tuple[GoldPage, PageText]] = []
    skipped: list[Skipped] = []
    for page in pages:
        if not page.confirmed:
            skipped.append(Skipped(page, SkipReason.UNCONFIRMED))
            continue
        row = stored.get(url_key(page.url))
        if row is not None and row.page is not None and row.content_hash == page.content_hash:
            ready.append((page, _as_labelled(row.page, page)))
            continue

        fetched = await fetch_page(client, robots, page.url)
        if fetched.status is not PageStatus.OK:
            skipped.append(Skipped(page, SkipReason.UNREADABLE, fetched.detail))
            continue
        if text_hash(fetched.text) != page.content_hash:
            skipped.append(Skipped(page, SkipReason.STALE))
            continue
        # As `crawl` stores them: a listing's links, and none for a show's own page.
        links = tuple(collect_links(fetched.page.content, fetched.page.url)) if page.listing else ()
        text = PageText(
            page.url, page.venue, page.listing, page.content_hash, fetched_on, fetched.text, links
        )
        ready.append((page, text))
    return ready, skipped


def _as_labelled(text: PageText, page: GoldPage) -> PageText:
    """The stored text, read as the label says: under its venue and kind."""
    return PageText(
        text.url, page.venue, page.listing, text.content_hash, text.fetched_on, text.text,
        text.links,
    )  # fmt: skip


# --- the model -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    page: PageText
    extraction: Extraction | None = None  # None when it failed
    error: str = ""


@dataclass(frozen=True)
class Extracted:
    results: list[Result]
    stopped: str = ""  # what the endpoint said when it refused a call; the pages after it wait


async def extract_all(llm: LlmClient, model: str, pages: Sequence[PageText]) -> Extracted:
    """Extract every page, as `ingest/extraction.extract_pages` does: an answer that stays
    unusable fails that page, and a call the endpoint refuses stops the run there."""
    results = []
    for page in pages:
        try:
            extraction = await extract_page(
                llm,
                model,
                venue_name=page.venue_name,
                url=page.url,
                text=page.text,
                fetched_on=page.fetched_on,
                listing=page.listing,
                links=page.links,
            )
        except ExtractionError as exc:
            results.append(Result(page, error=str(exc)))
        except (BotoCoreError, ClientError) as exc:
            return Extracted(results, stopped=f"{type(exc).__name__}: {exc}")
        else:
            results.append(Result(page, extraction))
    return Extracted(results)


@dataclass(frozen=True)
class Judged:
    pages: list[JudgedPage]
    stopped: str = ""  # as in `Extracted`


async def judge_all(llm: LlmClient, model: str, results: Sequence[Result]) -> Judged:
    """Judge the summaries on every page that has some, against the text they were written
    from. A page the judge gives no usable answer for is kept, unjudged; a call the endpoint
    refuses stops the run there."""
    pages = []
    for result in results:
        if result.extraction is None or not (written := summaries(result.extraction)):
            continue
        page = result.page
        try:
            judgements = await judge_summaries(
                llm,
                model,
                venue_name=page.venue_name,
                url=page.url,
                text=page.text,
                summaries=written,
            )
        except JudgeError as exc:
            pages.append(JudgedPage(page.url, tuple(written), error=str(exc)))
        except (BotoCoreError, ClientError) as exc:
            return Judged(pages, stopped=f"{type(exc).__name__}: {exc}")
        else:
            pages.append(JudgedPage(page.url, tuple(written), tuple(judgements)))
    return Judged(pages)


# --- drafting labels -----------------------------------------------------------------------------


def draft_pages(
    stored: Mapping[str, StoredPage], gold: GoldSet, venue: str | None
) -> list[PageText]:
    """The crawled pages that still hold text and that the gold set does not label, listings
    first. `venue` keeps only that venue's, compared as `normalize_name`."""
    labelled = {url_key(page.url) for page in gold.pages}
    return [
        row.page
        for key, row in stored.items()
        if row.page is not None
        and key not in labelled
        and (venue is None or normalize_name(row.venue_name) == normalize_name(venue))
    ]


# How far a listing is drafted, per kind. Past that, the draft is a partial label.
DRAFTED_ITEMS = 10

DRAFT_HEADER = """\
# Drafted by `eval-extraction --draft` on {today} with {model}: what extraction found, NOT gold.
#
# For each page, open the URL in a browser and check every item against it: fix titles, artists,
# dates and at_venue, add what was missed, delete what is not there. A listing is labelled as
# extraction reads it, in page order. One marked `partial: true` keeps only its first {drafted}
# exhibitions and first {drafted} events: check those, and that nothing before them is missing.
# Then set `confirmed: true` and move the entry into src/art_curator/eval/gold/extraction.yaml.
#
# content_hash is the text this draft was extracted from. If the live page has changed since,
# the label will be reported stale: crawl again and re-draft.

"""


def draft_yaml(results: Sequence[Result], model: str, today: date) -> str:
    """What extraction found on each page, as unconfirmed labels. A page that failed is left out:
    there is nothing to draft from."""
    pages = []
    for result in results:
        if result.extraction is None:
            continue
        shows, events = result.extraction.exhibitions, result.extraction.events
        partial = result.page.listing and max(len(shows), len(events)) > DRAFTED_ITEMS
        if partial:
            shows, events = shows[:DRAFTED_ITEMS], events[:DRAFTED_ITEMS]
        page = {
            "url": result.page.url,
            "venue": result.page.venue_name,
            "kind": "listing" if result.page.listing else "detail",
            "content_hash": result.page.content_hash,
            "confirmed": False,
        }
        if partial:
            page["partial"] = True
        page["items"] = [_draft_item("exhibition", show) for show in shows] + [
            _draft_item("event", event) for event in events
        ]
        pages.append(page)
    body = yaml.safe_dump({"pages": pages}, allow_unicode=True, sort_keys=False, width=100)
    return DRAFT_HEADER.format(today=today, model=model, drafted=DRAFTED_ITEMS) + body


def _draft_item(kind: str, item: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "title": item.title,
        "artists": list(item.artists),
        "start": item.start_date,
        "end": item.end_date,
        "at_venue": item.at_venue,
    }
