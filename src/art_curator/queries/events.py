"""The shared query module (PLAN.md § 3): venues, their pages, what extraction found, GRAF's
events and what the model calls cost.

Every function takes an `AsyncSession`, only reads, and returns frozen dataclasses. Whoever calls
them owns the session and its transaction; `mcp_server.py` makes it a read-only one.

Rules the code encodes:

- **Page text never leaves the database.** `venue_pages.raw_text` and `venue_pages.links` are
  third-party prose (CLAUDE.md § 1). A page is described by its metadata: status, hash, word and
  link counts worked out in SQL, and whether the text is still held. Neither column is ever
  selected, and `tests/test_no_verbatim.py` plants a sentence in both and checks that no MCP tool
  returns it. `summary_en` is ours, so it is returned.
- **Dates are decided in Postgres** (CLAUDE.md § 8), on the day as the venue would name it
  (`SITE_TIMEZONE`). "Not over" means what `crawl` means by it: an exhibition with no end date
  may still be on. An event with no end date is over after its start date.
- **Rows are capped by the caller's `limit`.** These rows land in a conversation, so a query
  that could return hundreds takes one.
"""

from dataclasses import dataclass, fields
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import Date, Select, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from art_curator.db.models import (
    ArtEvent,
    EventOccurrence,
    EventSnapshot,
    Exhibition,
    LlmCall,
    Venue,
    VenuePage,
    VenueSeed,
)
from art_curator.ingest.extraction import SITE_TIMEZONE

VENUES = Venue.__table__
SEEDS = VenueSeed.__table__
PAGES = VenuePage.__table__
EXHIBITIONS = Exhibition.__table__
EVENTS = ArtEvent.__table__
SNAPSHOTS = EventSnapshot.__table__
OCCURRENCES = EventOccurrence.__table__
CALLS = LlmCall.__table__


def _local_day(column: Any) -> ColumnElement[date]:
    """The day a timestamp falls on at the venues."""
    return cast(func.timezone(SITE_TIMEZONE, column), Date)


async def _rows[T](session: AsyncSession, row_type: type[T], query: Select) -> list[T]:
    return [row_type(**row._mapping) for row in await session.execute(query)]


# --- venues --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VenueRow:
    id: int
    name: str
    city: str | None
    website_url: str | None
    instagram_url: str | None
    crawl_enabled: bool
    is_pilot: bool
    latitude: float | None  # NULL for the terms GRAF puts at 0, 0
    longitude: float | None
    pages: int  # rows in venue_pages, whatever their status
    exhibitions: int
    events: int


async def venues(
    session: AsyncSession, *, pilot_only: bool = True, name: str | None = None, limit: int = 100
) -> list[VenueRow]:
    """Venues by name. `name` matches any part of it, ignoring case."""

    def count(table: Any, label: str) -> Any:
        return (
            select(func.count())
            .where(table.c.venue_id == VENUES.c.id)
            .scalar_subquery()
            .label(label)
        )

    # ST_X / ST_Y take a geometry, not a geography. Same type and SRID, so the cast is exact.
    point = cast(VENUES.c.geom, Geometry("POINT", srid=4326))
    query = (
        select(
            VENUES.c.id,
            VENUES.c.name,
            VENUES.c.city,
            VENUES.c.website_url,
            VENUES.c.instagram_url,
            VENUES.c.crawl_enabled,
            VENUES.c.is_pilot,
            func.ST_Y(point).label("latitude"),
            func.ST_X(point).label("longitude"),
            count(PAGES, "pages"),
            count(EXHIBITIONS, "exhibitions"),
            count(EVENTS, "events"),
        )
        .order_by(VENUES.c.name, VENUES.c.id)
        .limit(limit)
    )
    if pilot_only:
        query = query.where(VENUES.c.is_pilot)
    if name:
        query = query.where(VENUES.c.name.icontains(name, autoescape=True))
    return await _rows(session, VenueRow, query)


# --- discovery and crawl -------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedRow:
    """What `discover` concluded about one candidate listing page. A person's pins and
    rejections are in `ingest/seeds.yaml`, not here."""

    venue_id: int
    venue_name: str
    url: str
    page_type: str
    confidence: str
    dated_items: int
    language: str
    status: str
    model: str
    discovered_at: datetime


async def seeds(
    session: AsyncSession, *, venue_id: int | None = None, limit: int = 100
) -> list[SeedRow]:
    query = (
        select(
            SEEDS.c.venue_id,
            VENUES.c.name.label("venue_name"),
            SEEDS.c.url,
            SEEDS.c.page_type,
            SEEDS.c.confidence,
            SEEDS.c.dated_items,
            SEEDS.c.language,
            SEEDS.c.status,
            SEEDS.c.model,
            SEEDS.c.discovered_at,
        )
        .join(VENUES, VENUES.c.id == SEEDS.c.venue_id)
        .order_by(VENUES.c.name, SEEDS.c.url)
        .limit(limit)
    )
    if venue_id is not None:
        query = query.where(SEEDS.c.venue_id == venue_id)
    return await _rows(session, SeedRow, query)


@dataclass(frozen=True)
class PageRow:
    """A crawled page, by its metadata. Its text is never returned (module docstring)."""

    id: int
    venue_id: int
    venue_name: str
    url: str
    kind: str  # listing or detail
    status: str  # what the last fetch found
    http_status: int | None
    content_hash: str | None
    fetched_at: datetime
    text_held: bool  # False once the 7-day purge has run, or for a page that was not `ok`
    words: int | None  # NULL when the text is not held
    link_count: int | None  # a listing's numbered links; NULL for a detail page or once purged
    extract_status: str | None  # NULL until first extracted
    extract_version: int | None
    extract_model: str | None
    extracted_items: int | None
    extracted_at: datetime | None
    # The last extraction read the text stored now. `extract` also re-runs a page when
    # EXTRACT_VERSION or the model moved: compare those two columns for that.
    text_extracted: bool


async def pages(
    session: AsyncSession, *, venue_id: int | None = None, limit: int = 100
) -> list[PageRow]:
    """Every page in `venue_pages`, listings first within each venue."""
    query = (
        select(
            PAGES.c.id,
            PAGES.c.venue_id,
            VENUES.c.name.label("venue_name"),
            PAGES.c.url,
            PAGES.c.kind,
            PAGES.c.status,
            PAGES.c.http_status,
            PAGES.c.content_hash,
            PAGES.c.fetched_at,
            PAGES.c.raw_text.is_not(None).label("text_held"),
            func.cardinality(func.regexp_split_to_array(func.btrim(PAGES.c.raw_text), r"\s+"))
            .label("words"),
            func.jsonb_array_length(PAGES.c.links).label("link_count"),
            PAGES.c.extract_status,
            PAGES.c.extract_version,
            PAGES.c.extract_model,
            PAGES.c.extracted_items,
            PAGES.c.extracted_at,
            func.coalesce(PAGES.c.extracted_hash == PAGES.c.content_hash, False)
            .label("text_extracted"),
        )
        .join(VENUES, VENUES.c.id == PAGES.c.venue_id)
        .order_by(VENUES.c.name, PAGES.c.kind != "listing", PAGES.c.url)
        .limit(limit)
    )  # fmt: skip
    if venue_id is not None:
        query = query.where(PAGES.c.venue_id == venue_id)
    return await _rows(session, PageRow, query)


# --- what extraction found -----------------------------------------------------------------------


@dataclass(frozen=True)
class _ItemRow:
    """What exhibitions and events share (`models._Extracted`), less `title_key`."""

    id: int
    venue_id: int
    venue_name: str
    title: str
    artists: list[str]
    start_date: date | None
    end_date: date | None
    at_venue: bool
    detail_url: str | None
    summary_en: str | None  # ours, never the page's (CLAUDE.md § 1)
    themes: list[str]
    media: list[str]
    confidence: str
    source_url: str  # the page that wrote it last
    content_hash: str  # of that page's text, when it did
    model: str
    extract_version: int
    first_seen_at: datetime
    last_seen_at: datetime


@dataclass(frozen=True)
class ExhibitionRow(_ItemRow):
    pass


@dataclass(frozen=True)
class ArtEventRow(_ItemRow):
    kind: str
    start_time: time | None
    exhibition_id: int | None  # the show it belongs to, when the page said


def _item_columns(table: Any) -> list[Any]:
    return [
        VENUES.c.name.label("venue_name") if f.name == "venue_name" else table.c[f.name]
        for f in fields(_ItemRow)
    ]


def _item_filters(
    query: Select,
    table: Any,
    *,
    venue_id: int | None,
    source_url: str | None,
    title: str | None,
    not_over: ColumnElement[bool] | None,
) -> Select:
    if venue_id is not None:
        query = query.where(table.c.venue_id == venue_id)
    if source_url is not None:
        query = query.where(table.c.source_url == source_url)
    if title:
        query = query.where(table.c.title.icontains(title, autoescape=True))
    if not_over is not None:
        query = query.where(not_over)
    return query


async def exhibitions(
    session: AsyncSession,
    *,
    venue_id: int | None = None,
    source_url: str | None = None,
    title: str | None = None,
    not_over_on: date | None = None,
    limit: int = 100,
) -> list[ExhibitionRow]:
    """Stored exhibitions, earliest start first within each venue. `source_url` is the page that
    wrote a show last; `title` matches any part of it, ignoring case; `not_over_on` keeps the
    shows that have not ended by that day, an open-ended one included."""
    query = (
        select(*_item_columns(EXHIBITIONS))
        .join(VENUES, VENUES.c.id == EXHIBITIONS.c.venue_id)
        .order_by(VENUES.c.name, EXHIBITIONS.c.start_date.nulls_last(), EXHIBITIONS.c.id)
        .limit(limit)
    )
    not_over = (
        func.coalesce(EXHIBITIONS.c.end_date, not_over_on) >= not_over_on
        if not_over_on is not None
        else None
    )
    query = _item_filters(
        query,
        EXHIBITIONS,
        venue_id=venue_id,
        source_url=source_url,
        title=title,
        not_over=not_over,
    )
    return await _rows(session, ExhibitionRow, query)


async def art_events(
    session: AsyncSession,
    *,
    venue_id: int | None = None,
    source_url: str | None = None,
    title: str | None = None,
    kind: str | None = None,
    not_over_on: date | None = None,
    limit: int = 100,
) -> list[ArtEventRow]:
    """Stored openings, talks, tours…, earliest first within each venue. Filters as in
    `exhibitions`, plus `kind`. An event with no end date is over after its start date, and
    one with neither date is kept."""
    query = (
        select(*_item_columns(EVENTS), EVENTS.c.kind, EVENTS.c.start_time, EVENTS.c.exhibition_id)
        .join(VENUES, VENUES.c.id == EVENTS.c.venue_id)
        .order_by(
            VENUES.c.name,
            EVENTS.c.start_date.nulls_last(),
            EVENTS.c.start_time.nulls_last(),
            EVENTS.c.id,
        )
        .limit(limit)
    )
    if kind is not None:
        query = query.where(EVENTS.c.kind == kind)
    not_over = (
        func.coalesce(EVENTS.c.end_date, EVENTS.c.start_date, not_over_on) >= not_over_on
        if not_over_on is not None
        else None
    )
    query = _item_filters(
        query,
        EVENTS,
        venue_id=venue_id,
        source_url=source_url,
        title=title,
        not_over=not_over,
    )
    return await _rows(session, ArtEventRow, query)


# --- GRAF ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GrafEventRow:
    """One occurrence of a GRAF event: facts only, never its description (CLAUDE.md § 1)."""

    event_id: int
    source_event_id: int  # GRAF's post id
    venue_id: int | None
    venue_name: str | None
    title: str
    title_en: str | None
    starts_at: datetime
    ends_at: datetime | None
    is_free: bool | None
    price_min: Decimal | None
    price_max: Decimal | None
    is_online: bool | None
    source_url: str
    web_url_ca: str | None
    web_url_es: str | None
    web_url_en: str | None
    # The last sync that saw it. GRAF serves a live window, so an older one has left it.
    last_seen_at: datetime


async def graf_events(
    session: AsyncSession,
    *,
    venue_id: int | None = None,
    not_over_on: date | None = None,
    limit: int = 100,
) -> list[GrafEventRow]:
    """GRAF's events, one row per occurrence, earliest start first. `not_over_on` keeps the
    occurrences that end (or, with no end, start) on or after that day."""
    query = (
        select(
            SNAPSHOTS.c.id.label("event_id"),
            SNAPSHOTS.c.source_event_id,
            SNAPSHOTS.c.venue_id,
            VENUES.c.name.label("venue_name"),
            SNAPSHOTS.c.title,
            SNAPSHOTS.c.title_en,
            OCCURRENCES.c.starts_at,
            OCCURRENCES.c.ends_at,
            SNAPSHOTS.c.is_free,
            SNAPSHOTS.c.price_min,
            SNAPSHOTS.c.price_max,
            SNAPSHOTS.c.is_online,
            SNAPSHOTS.c.source_url,
            SNAPSHOTS.c.web_url_ca,
            SNAPSHOTS.c.web_url_es,
            SNAPSHOTS.c.web_url_en,
            SNAPSHOTS.c.last_seen_at,
        )
        .join(OCCURRENCES, OCCURRENCES.c.event_id == SNAPSHOTS.c.id)
        .outerjoin(VENUES, VENUES.c.id == SNAPSHOTS.c.venue_id)
        .order_by(OCCURRENCES.c.starts_at, SNAPSHOTS.c.id, OCCURRENCES.c.id)
        .limit(limit)
    )
    if venue_id is not None:
        query = query.where(SNAPSHOTS.c.venue_id == venue_id)
    if not_over_on is not None:
        last_day = _local_day(func.coalesce(OCCURRENCES.c.ends_at, OCCURRENCES.c.starts_at))
        query = query.where(last_day >= not_over_on)
    return await _rows(session, GrafEventRow, query)


# --- cost ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CostRow:
    """The `llm_calls` of one purpose and model, added up."""

    purpose: str
    model: str
    calls: int
    errors: int  # calls that raised; recorded and costed like the rest
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: Decimal
    first_call: datetime
    last_call: datetime


async def llm_costs(
    session: AsyncSession, *, since: date | None = None, purpose: str | None = None
) -> list[CostRow]:
    """Model calls by purpose and model, costliest first. `since` is a day at the venues."""

    def total(column: Any) -> Any:
        return func.sum(column).label(column.name)

    query = (
        select(
            CALLS.c.purpose,
            CALLS.c.model,
            func.count().label("calls"),
            func.count(CALLS.c.error_type).label("errors"),
            total(CALLS.c.input_tokens),
            total(CALLS.c.output_tokens),
            total(CALLS.c.cache_creation_input_tokens),
            total(CALLS.c.cache_read_input_tokens),
            total(CALLS.c.cost_usd),
            func.min(CALLS.c.created_at).label("first_call"),
            func.max(CALLS.c.created_at).label("last_call"),
        )
        .group_by(CALLS.c.purpose, CALLS.c.model)
        .order_by(func.sum(CALLS.c.cost_usd).desc(), CALLS.c.purpose, CALLS.c.model)
    )
    if since is not None:
        query = query.where(_local_day(CALLS.c.created_at) >= since)
    if purpose is not None:
        query = query.where(CALLS.c.purpose == purpose)
    return await _rows(session, CostRow, query)
