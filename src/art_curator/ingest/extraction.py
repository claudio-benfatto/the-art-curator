"""Run extraction over the crawled pages and keep what it found (PLAN.md § 8, P2).

`llm/extract.py` is the model call, with no I/O. This module is everything around it, in the
order `cli extract` runs it:

1. `crawl.purge_pages()`: first, every run, as in `crawl`. Text past the TTL is never sent to a
   model (CLAUDE.md § 1).
2. `load_pages()`: every `ok` page, sorted into the ones to extract, the ones unchanged since
   their last extraction, and the ones that changed but whose text has since been purged.
3. `extract_pages()`: model only, no database. Each page becomes a `PageResult`.
4. `write_results()`: upserts `exhibitions` and `art_events` and marks each page, in the caller's
   transaction, and **never commits** (`--dry-run` is the real write rolled back).
5. `report()`: pure.

Rules the code encodes:

- **An unchanged page costs nothing.** A page goes to the model only when its `content_hash`,
  `EXTRACT_VERSION` or the model differs from what its last extraction ran with. A page that
  failed is marked as well: at temperature 0 the same text fails the same way, so it waits for
  one of the three to move.
- **Listings first, then detail pages**, so a show's own page is written after the listing that
  named it.
- **One show, one row**, on `(venue_id, title_key)`; an event adds its start date. When a show is
  seen again, each field takes the newest value that is not empty, so a date corrected on the
  listing lands, and a detail page that gives no dates keeps the listing's. The exception is
  `summary_en`: the longer one stays, so a listing's one-liner does not replace what the show's
  own page said. Nothing is deleted.
- **A listing that stops yielding is a broken seed**, the third sign after "gone" and "redirects
  home" (`crawl.Broken`): it found items last time and finds none now.
- **A model endpoint that refuses the call stops the run**, as in `discover`: that is not the
  page's doing, and marking every page failed would hide it.

Page text and link text are third-party prose: they are kept out of `repr`, go to the model, and
are never printed. What is written is facts and our own summaries (`extract.longest_shared_run`).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import Date, case, cast, func, select, update
from sqlalchemy.dialects.postgresql import ARRAY, insert
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.db.models import ArtEvent, Exhibition, Venue, VenuePage
from art_curator.ingest.crawl import BROKEN_ADVICE
from art_curator.ingest.matching import normalize_name
from art_curator.llm.client import LlmClient
from art_curator.llm.extract import EXTRACT_VERSION, Extraction, ExtractionError, extract_page

PAGES = VenuePage.__table__
VENUES = Venue.__table__
EXHIBITIONS = Exhibition.__table__
EVENTS = ArtEvent.__table__

# The day a page was fetched, as its venue would name it: what a date without a year is read
# against. Worked out in Postgres (CLAUDE.md § 8).
SITE_TIMEZONE = "Europe/Madrid"

# Columns that identify a row, and the one column where the longer value wins (module docstring).
_KEYS = {"venue_id", "title_key"}
_LONGER_WINS = "summary_en"


def title_key(title: str) -> str:
    """What makes two titles one show: `Prova U`, `PROVA U` and `Prova U.` agree."""
    return normalize_name(title) or title.casefold().strip()


# --- pages ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredLink:
    """One entry of `venue_pages.links`; an `extract.PageLink`."""

    number: int
    url: str
    text: str = field(repr=False)  # third-party


@dataclass(frozen=True)
class PageRef:
    venue_name: str
    url: str


@dataclass(frozen=True)
class PageToExtract:
    page_id: int
    venue_id: int
    venue_name: str
    url: str
    listing: bool
    content_hash: str
    fetched_on: date
    previous_items: int | None  # what the last successful extraction found, if there was one
    text: str = field(repr=False)  # third-party prose
    links: tuple[StoredLink, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class PagePlan:
    to_extract: tuple[PageToExtract, ...] = ()
    unchanged: tuple[PageRef, ...] = ()
    expired: tuple[PageRef, ...] = ()  # changed, but the text is gone: crawl again

    @property
    def venue_names(self) -> list[str]:
        pages = (*self.to_extract, *self.unchanged, *self.expired)
        return sorted({page.venue_name for page in pages}, key=normalize_name)


async def load_pages(session: AsyncSession, model: str) -> PagePlan:
    """Every `ok` page, sorted by what this run should do with it. Listings come first."""
    to_extract, unchanged, expired = [], [], []
    for row in await session.execute(
        select(
            PAGES.c.id,
            PAGES.c.venue_id,
            VENUES.c.name.label("venue_name"),
            PAGES.c.url,
            PAGES.c.kind,
            PAGES.c.content_hash,
            PAGES.c.raw_text,
            PAGES.c.links,
            cast(func.timezone(SITE_TIMEZONE, PAGES.c.fetched_at), Date).label("fetched_on"),
            PAGES.c.extract_status,
            PAGES.c.extracted_hash,
            PAGES.c.extract_version,
            PAGES.c.extract_model,
            PAGES.c.extracted_items,
        )
        .join(VENUES, VENUES.c.id == PAGES.c.venue_id)
        .where(PAGES.c.status == "ok", PAGES.c.content_hash.is_not(None))
        .order_by(case((PAGES.c.kind == "listing", 0), else_=1), PAGES.c.id)
    ):
        ref = PageRef(row.venue_name, row.url)
        last = (row.extracted_hash, row.extract_version, row.extract_model)
        if last == (row.content_hash, EXTRACT_VERSION, model):
            unchanged.append(ref)
        elif row.raw_text is None:
            expired.append(ref)
        else:
            to_extract.append(
                PageToExtract(
                    page_id=row.id,
                    venue_id=row.venue_id,
                    venue_name=row.venue_name,
                    url=row.url,
                    listing=row.kind == "listing",
                    content_hash=row.content_hash,
                    fetched_on=row.fetched_on,
                    previous_items=row.extracted_items if row.extract_status == "ok" else None,
                    text=row.raw_text,
                    links=tuple(
                        StoredLink(link["number"], link["url"], link["text"])
                        for link in row.links or ()
                    ),
                )
            )
    return PagePlan(tuple(to_extract), tuple(unchanged), tuple(expired))


def select_pages(plan: PagePlan, venue: str | None) -> PagePlan:
    """`--venue`: the pages of the venue that matches, compared as `normalize_name`."""
    if venue is None:
        return plan
    key = normalize_name(venue)

    def matches(page: PageToExtract | PageRef) -> bool:
        return normalize_name(page.venue_name) == key

    return PagePlan(
        tuple(filter(matches, plan.to_extract)),
        tuple(filter(matches, plan.unchanged)),
        tuple(filter(matches, plan.expired)),
    )


# --- the model -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PageResult:
    page: PageToExtract
    extraction: Extraction | None = None  # None when it failed
    error: str = ""

    @property
    def items(self) -> int:
        if self.extraction is None:
            return 0
        return len(self.extraction.exhibitions) + len(self.extraction.events)

    @property
    def broken(self) -> bool:
        """A listing that found items last time and finds none now."""
        return (
            self.page.listing
            and self.extraction is not None
            and self.items == 0
            and bool(self.page.previous_items)
        )


async def extract_pages(
    llm: LlmClient, model: str, pages: Sequence[PageToExtract]
) -> list[PageResult]:
    """Extract every page. An answer that stays unusable fails that page and the run goes on;
    anything else `extract_page` raises (the endpoint refusing the call) stops it."""
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
                links=page.links,
            )
        except ExtractionError as exc:
            results.append(PageResult(page, error=str(exc)))
        else:
            results.append(PageResult(page, extraction))
    return results


# --- exhibitions, art_events, venue_pages --------------------------------------------------------


@dataclass(frozen=True)
class Written:
    pages: int = 0
    exhibitions: int = 0
    events: int = 0


async def write_results(
    session: AsyncSession, results: Sequence[PageResult], model: str
) -> Written:
    """Upsert what every page yielded and mark every page, in order. Runs in the caller's
    transaction and never commits."""
    exhibitions_written = events_written = 0
    for result in results:
        page, extraction = result.page, result.extraction
        if extraction is not None:
            shows = _dedupe(
                [_row(item, page, model) for item in extraction.exhibitions], ("title_key",)
            )
            await _upsert(session, EXHIBITIONS, shows, index_elements=["venue_id", "title_key"])
            ids = await _exhibition_ids(session, page.venue_id, extraction)
            events = _dedupe(
                [
                    _row(event, page, model)
                    | {
                        "kind": event.kind,
                        "start_time": event.start_time,
                        "exhibition_id": ids.get(title_key(event.exhibition_title))
                        if event.exhibition_title
                        else None,
                    }
                    for event in extraction.events
                ],
                ("title_key", "start_date"),
            )
            await _upsert(
                session, EVENTS, events, constraint="uq_art_events_venue_id_title_key_start_date"
            )
            exhibitions_written += len(shows)
            events_written += len(events)
        await session.execute(
            update(PAGES)
            .where(PAGES.c.id == page.page_id)
            .values(
                extract_status="ok" if extraction is not None else "failed",
                extracted_hash=page.content_hash,
                extract_version=EXTRACT_VERSION,
                extract_model=model,
                extracted_items=result.items if extraction is not None else None,
                extracted_at=func.now(),
            )
        )
    return Written(len(results), exhibitions_written, events_written)


def _row(item: Any, page: PageToExtract, model: str) -> dict[str, Any]:
    return {
        "venue_id": page.venue_id,
        "title": item.title,
        "title_key": title_key(item.title),
        "artists": list(item.artists),
        "start_date": item.start_date,
        "end_date": item.end_date,
        "at_venue": item.at_venue,
        "detail_url": item.detail_url,
        "summary_en": item.summary_en,
        "themes": list(item.themes),
        "media": list(item.media),
        "confidence": item.confidence,
        "source_url": page.url,
        "content_hash": page.content_hash,
        "model": model,
        "extract_version": EXTRACT_VERSION,
    }


def merge(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """The merge rule in Python, for one page naming a show twice. `_merge_set` is the same rule
    in SQL, for a show already stored."""
    merged = dict(old)
    for column, value in new.items():
        if column == _LONGER_WINS:
            if value and len(value) > len(merged.get(column) or ""):
                merged[column] = value
        elif value is not None and value != []:
            merged[column] = value
    return merged


def _dedupe(rows: list[dict[str, Any]], key: tuple[str, ...]) -> list[dict[str, Any]]:
    """One row per key, merged in page order. Postgres refuses an upsert that touches a row
    twice in one statement."""
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        k = tuple(row[c] for c in key)
        by_key[k] = merge(by_key[k], row) if k in by_key else row
    return list(by_key.values())


def _merge_set(table: Any, excluded: Any, columns: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in columns:
        new, old = excluded[column], table.c[column]
        if column == _LONGER_WINS:
            # length(NULL) is NULL, so a NULL newcomer never wins.
            out[column] = case(
                (func.length(new) > func.coalesce(func.length(old), 0), new), else_=old
            )
        elif isinstance(old.type, ARRAY):
            out[column] = case((func.cardinality(new) > 0, new), else_=old)
        else:
            out[column] = func.coalesce(new, old)
    return out | {"last_seen_at": func.now()}


async def _upsert(session: AsyncSession, table: Any, rows: list[dict[str, Any]], **target) -> None:
    if not rows:
        return
    stmt = insert(table).values(rows)
    columns = [c for c in rows[0] if c not in _KEYS]
    await session.execute(
        stmt.on_conflict_do_update(**target, set_=_merge_set(table, stmt.excluded, columns))
    )


async def _exhibition_ids(
    session: AsyncSession, venue_id: int, extraction: Extraction
) -> dict[str, int]:
    """The stored exhibitions the page's events name, by `title_key`. Any page's exhibition
    counts, including one written earlier in this transaction."""
    keys = {title_key(e.exhibition_title) for e in extraction.events if e.exhibition_title}
    if not keys:
        return {}
    rows = await session.execute(
        select(EXHIBITIONS.c.title_key, EXHIBITIONS.c.id).where(
            EXHIBITIONS.c.venue_id == venue_id, EXHIBITIONS.c.title_key.in_(keys)
        )
    )
    return dict(rows.tuples().all())


# --- the report ----------------------------------------------------------------------------------

EXPIRED_ADVICE = "its text was purged before it was extracted: crawl again"


def report(plan: PagePlan, results: Sequence[PageResult]) -> list[str]:
    """What `extract` prints: page counts, item counts, then every page sent to the model under
    its venue. A `broken` line follows a listing that stopped yielding; pages whose text expired
    come last."""
    lines: list[str] = []

    def block(label: str, *parts: str) -> None:
        lines.append(f"{label:<11} " + "   ".join(part for part in parts if part))

    failed = [r for r in results if r.extraction is None]
    done = [r.extraction for r in results if r.extraction is not None]
    left = len(plan.to_extract) - len(results)
    total = len(plan.to_extract) + len(plan.unchanged) + len(plan.expired)

    block(
        "pages",
        str(total),
        f"{len(plan.to_extract)} to extract",
        f"{len(plan.unchanged)} unchanged",
        f"{len(plan.expired)} expired" if plan.expired else "",
    )
    block(
        "extracted",
        str(len(results)),
        f"{len(done)} ok",
        f"{len(failed)} failed" if failed else "",
        f"{left} left by --limit" if left else "",
    )
    if done:
        dropped = sum(e.summaries_dropped for e in done)
        block(
            "items",
            f"{sum(len(e.exhibitions) for e in done)} exhibitions",
            f"{sum(len(e.events) for e in done)} events",
            f"{dropped} summaries dropped by the verbatim guard" if dropped else "",
        )

    by_venue: dict[str, list[PageResult]] = {}
    for result in results:
        by_venue.setdefault(result.page.venue_name, []).append(result)
    for name in sorted(by_venue, key=normalize_name):
        block("site", name)
        for result in by_venue[name]:
            url = result.page.url
            if result.extraction is None:
                block("failed", url, result.error)
                continue
            block(
                "ok",
                url,
                f"{len(result.extraction.exhibitions)} exhibitions",
                f"{len(result.extraction.events)} events",
            )
            if result.broken:
                block(
                    "broken",
                    url,
                    f"no items, after {result.page.previous_items} last time: {BROKEN_ADVICE}",
                )
    for page in plan.expired:
        block("expired", page.url, EXPIRED_ADVICE)
    return lines
