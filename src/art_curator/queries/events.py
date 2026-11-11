"""The shared query module (PLAN.md § 3): venues, their pages, what extraction found, GRAF's
events and what the model calls cost, plus `search_events` and `get_events`, the curator's view
of all three sources at once.

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

import re
from collections.abc import Collection
from dataclasses import dataclass, fields
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from geoalchemy2 import Geography, Geometry
from sqlalchemy import (
    Boolean,
    Date,
    Float,
    Numeric,
    Select,
    Text,
    Time,
    and_,
    case,
    cast,
    exists,
    func,
    literal,
    literal_column,
    null,
    or_,
    select,
    union_all,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import CompoundSelect

from art_curator.db.models import (
    EVENT_KINDS,
    PLACE_KINDS,
    ArtEvent,
    EventOccurrence,
    EventSnapshot,
    Exhibition,
    LlmCall,
    Place,
    Venue,
    VenuePage,
    VenueSeed,
)
from art_curator.ingest.extraction import SITE_TIMEZONE
from art_curator.ingest.graf import CATEGORIES

VENUES = Venue.__table__
SEEDS = VenueSeed.__table__
PAGES = VenuePage.__table__
EXHIBITIONS = Exhibition.__table__
EVENTS = ArtEvent.__table__
SNAPSHOTS = EventSnapshot.__table__
OCCURRENCES = EventOccurrence.__table__
PLACES = Place.__table__
CALLS = LlmCall.__table__


def _local_day(column: Any) -> ColumnElement[date]:
    """The day a timestamp falls on at the venues."""
    return cast(func.timezone(SITE_TIMEZONE, column), Date)


def _local_time(column: Any) -> ColumnElement[time]:
    """The time a timestamp shows on a clock at the venues."""
    return cast(func.timezone(SITE_TIMEZONE, column), Time)


async def _rows[T](
    session: AsyncSession, row_type: type[T], query: Select | CompoundSelect
) -> list[T]:
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


# --- search and hydrate: the curator's tools -----------------------------------------------------
#
# Exhibitions, art events and GRAF occurrences as one list in one row shape (PLAN.md § 8, P3).
# An id names its table by a prefix, so `get_events` knows where to look:
#
#   x:12   exhibitions.id         a:7   art_events.id         g:301   event_occurrences.id
#
# Each source becomes a *run* of local days, the same rules as the `not_over_on` filters above:
#
# - an exhibition runs from `start_date` to `end_date`. A missing start may have passed, and a
#   missing end may still be to come: both are open.
# - an art event without an end lasts its start day. One with no date at all cannot be placed
#   in any window, so it is never searched.
# - a GRAF occurrence runs from the day it starts to the day it ends at the venues. Only the
#   ones the latest sync saw are searched (`_graf_live`).
#
# A GRAF event that extraction also found is hidden (`_has_extracted_twin`), so one show is not
# listed twice. Distance is to the venue's point; an item shown elsewhere (`at_venue` false) or
# online has no place, so it has no distance and never falls inside a radius.

KINDS = ("exhibition", *EVENT_KINDS)
MAX_HYDRATE = 10  # ids per `get_events` call (PLAN.md § 5)

EVENT_ID_PATTERN = r"^([xag]):([0-9]+)$"  # plain groups: tool schemas take it as is
_EVENT_ID = re.compile(EVENT_ID_PATTERN)
_NO_ARRAY = literal_column("ARRAY[]::text[]", ARRAY(Text))


def parse_event_id(event_id: str) -> tuple[str, int]:
    """`"x:12"` -> `("x", 12)`. Raises ValueError for anything that is not a prefixed id."""
    match = _EVENT_ID.match(event_id)
    if match is None:
        raise ValueError(f"{event_id!r} is not an id from search_events (x:12, a:7 or g:301)")
    prefix, number = match.groups()
    return prefix, int(number)


def _prefixed(prefix: str, column: Any) -> ColumnElement[str]:
    """`x:` || id. NULL stays NULL, so an event with no exhibition has no `x:` id."""
    return literal(f"{prefix}:").concat(cast(column, Text))


def _match_key(column: Any) -> ColumnElement[str]:
    """A title made comparable in SQL: decomposed (NFKD) with its accents dropped, lower case, `&`
    as ` and `, every other run of non-alphanumerics one space.

    It follows `extraction.title_key`, but it is not the stored `title_key` column: GRAF titles
    have none. Both titles of a comparison go through this one expression, so the two sides can
    never disagree about how a title is normalised.
    """
    decomposed = func.normalize(column, literal_column("NFKD"))
    unaccented = func.lower(func.regexp_replace(decomposed, "[\u0300-\u036f]", "", "g"))
    spaced = func.regexp_replace(func.replace(unaccented, "&", " and "), "[^[:alnum:]]+", " ", "g")
    return func.btrim(spaced)


def _overlap(first_a: Any, last_a: Any, first_b: Any, last_b: Any) -> ColumnElement[bool]:
    """Two runs of days share at least one. A NULL first day is open towards the past and a NULL
    last day towards the future."""
    return and_(
        or_(first_a.is_(None), last_b.is_(None), first_a <= last_b),
        or_(last_a.is_(None), first_b.is_(None), last_a >= first_b),
    )


# What every source contributes to a search, in union order. `place_venue_id` is the venue whose
# point locates the item, NULL when it is not at its venue.
_CANDIDATE = (
    "id", "kind", "category", "title", "venue_id", "venue", "first_day", "last_day", "at_venue",
    "is_free", "artists", "media", "themes", "place_venue_id",
)  # fmt: skip


def _candidate(**columns: Any) -> Select:
    return select(*(columns[name].label(name) for name in _CANDIDATE))


def _exhibition_candidates() -> Select:
    return _candidate(
        id=_prefixed("x", EXHIBITIONS.c.id),
        kind=literal("exhibition"),
        category=cast(null(), Text),  # GRAF's label; extraction answers with a kind only
        title=EXHIBITIONS.c.title,
        venue_id=EXHIBITIONS.c.venue_id,
        venue=VENUES.c.name,
        first_day=EXHIBITIONS.c.start_date,
        last_day=EXHIBITIONS.c.end_date,
        at_venue=EXHIBITIONS.c.at_venue,
        is_free=cast(null(), Boolean),  # venue pages give no price
        artists=EXHIBITIONS.c.artists,
        media=EXHIBITIONS.c.media,
        themes=EXHIBITIONS.c.themes,
        place_venue_id=case((EXHIBITIONS.c.at_venue, EXHIBITIONS.c.venue_id)),
    ).join_from(EXHIBITIONS, VENUES, VENUES.c.id == EXHIBITIONS.c.venue_id)


def _event_candidates() -> Select:
    return (
        _candidate(
            id=_prefixed("a", EVENTS.c.id),
            kind=EVENTS.c.kind,
            category=cast(null(), Text),
            title=EVENTS.c.title,
            venue_id=EVENTS.c.venue_id,
            venue=VENUES.c.name,
            first_day=EVENTS.c.start_date,
            last_day=func.coalesce(EVENTS.c.end_date, EVENTS.c.start_date),
            at_venue=EVENTS.c.at_venue,
            is_free=cast(null(), Boolean),
            artists=EVENTS.c.artists,
            media=EVENTS.c.media,
            themes=EVENTS.c.themes,
            place_venue_id=case((EVENTS.c.at_venue, EVENTS.c.venue_id)),
        )
        .join_from(EVENTS, VENUES, VENUES.c.id == EVENTS.c.venue_id)
        .where(or_(EVENTS.c.start_date.is_not(None), EVENTS.c.end_date.is_not(None)))
    )


def _graf_kind() -> ColumnElement[str]:
    kinds = {category_id: category.kind for category_id, category in CATEGORIES.items()}
    return case(kinds, value=SNAPSHOTS.c.source_category_id, else_=literal("other"))


def _graf_category() -> ColumnElement[str | None]:
    """GRAF's own label (`seminari`, `festival`…); NULL for an id `CATEGORIES` does not know."""
    slugs = {category_id: category.slug for category_id, category in CATEGORIES.items()}
    return case(slugs, value=SNAPSHOTS.c.source_category_id)


def _graf_first_day() -> ColumnElement[date]:
    return _local_day(OCCURRENCES.c.starts_at)


def _graf_last_day() -> ColumnElement[date]:
    return _local_day(func.coalesce(OCCURRENCES.c.ends_at, OCCURRENCES.c.starts_at))


def _graf_at_venue() -> ColumnElement[bool]:
    return func.coalesce(SNAPSHOTS.c.is_online, False).is_(False)


def _graf_live() -> list[ColumnElement[bool]]:
    """The occurrences GRAF still serves. The sync that last saw the event also saw this
    occurrence: one GRAF regenerated is left behind with stale dates (`models.EventOccurrence`).
    And that sync is the latest: an event that left the live window before it ended was taken
    down. Every row one sync touches has the same `last_seen_at`, so both are equalities."""
    latest_sync = select(func.max(SNAPSHOTS.c.last_seen_at)).scalar_subquery()
    return [
        OCCURRENCES.c.last_seen_at == SNAPSHOTS.c.last_seen_at,
        SNAPSHOTS.c.last_seen_at == latest_sync,
    ]


def _has_extracted_twin() -> ColumnElement[bool]:
    """The GRAF overlap rule (PLAN.md § 8, P3): extraction found the same show at a venue of the
    same organisation (the same venue, or one sharing its `source_profile_id`), under the same
    title (GRAF's own or its English one) and with dates that overlap. Exact, in SQL; a fuzzy
    match waits for traces that show duplicates."""
    twins = union_all(_exhibition_candidates(), _event_candidates()).subquery("twins")
    twin_venue = VENUES.alias("twin_venue")
    return (
        exists()
        .select_from(twins.join(twin_venue, twin_venue.c.id == twins.c.venue_id))
        .where(
            or_(
                twin_venue.c.id == SNAPSHOTS.c.venue_id,
                twin_venue.c.source_profile_id == VENUES.c.source_profile_id,
            ),
            _match_key(twins.c.title).in_(
                [_match_key(SNAPSHOTS.c.title), _match_key(SNAPSHOTS.c.title_en)]
            ),
            _overlap(twins.c.first_day, twins.c.last_day, _graf_first_day(), _graf_last_day()),
        )
    )


def _graf_candidates() -> Select:
    return (
        _candidate(
            id=_prefixed("g", OCCURRENCES.c.id),
            kind=_graf_kind(),
            category=_graf_category(),
            title=SNAPSHOTS.c.title,
            venue_id=SNAPSHOTS.c.venue_id,
            venue=VENUES.c.name,
            first_day=_graf_first_day(),
            last_day=_graf_last_day(),
            at_venue=_graf_at_venue(),
            is_free=SNAPSHOTS.c.is_free,
            artists=_NO_ARRAY,
            media=_NO_ARRAY,
            themes=_NO_ARRAY,
            place_venue_id=case((_graf_at_venue(), SNAPSHOTS.c.venue_id)),
        )
        .select_from(
            OCCURRENCES.join(SNAPSHOTS, SNAPSHOTS.c.id == OCCURRENCES.c.event_id).outerjoin(
                VENUES, VENUES.c.id == SNAPSHOTS.c.venue_id
            )
        )
        .where(*_graf_live(), ~_has_extracted_twin())
    )


# --- places: what `near` measures from ----------------------------------------------------------
#
# A name becomes a point here, never in the model (CLAUDE.md § 8). It names a place (`places`: a
# barri, district or municipality) or a venue that has a point. A place answers to its name, its
# aliases, and its name with what it lies within ("Canyelles, Garraf"); a venue to its name, and
# its name with its city ("PiRA, Barcelona"). Case, accents and punctuation never count: both sides
# go through `_match_key`.
#
# The matches are the names equal to what was asked or, when none is, those that hold every word
# of it: a name or alias, never the qualified form, and only from MIN_PART_CHARS characters, so a
# stray "22" or "de" is no place. Matches all within MERGE_RADIUS_KM of the first are one spot,
# and the first is taken: the most specific kind (RESOLVE_KINDS order), then the shortest name.
# That is how "Les Corts" is the barri rather than also its district (centroids 1.4 km apart), and
# "Barcelona" the city rather than also GRAF's catch-all venue term of that name. Matches further
# apart are candidates, for whoever asked to pick one of: "Canyelles" is a barri of Nou Barris and
# a town in the Garraf.

RESOLVE_KINDS = (*PLACE_KINDS, "venue")
MERGE_RADIUS_KM = 1.5
MIN_PART_CHARS = 4  # "22@" holds no word of four characters, so it is not "Aparador 22"
MAX_CANDIDATES = 10


@dataclass(frozen=True)
class PlaceRow:
    name: str
    kind: str  # one of RESOLVE_KINDS
    # A barri's district, a district's city, a municipality's comarca, a venue's city.
    within: str | None
    latitude: float
    longitude: float

    def qualified(self) -> str:
        """How to ask for this one and nothing else: "Canyelles, Garraf"."""
        return f"{self.name}, {self.within}" if self.within else self.name


@dataclass(frozen=True)
class PlaceMatch:
    place: PlaceRow | None  # what the name means; None when it means nothing, or several things
    candidates: list[PlaceRow]  # the several things, most specific first; empty when unknown


class UnresolvedPlace(ValueError):
    """`near` names no place, or several far apart."""

    def __init__(self, asked: str, candidates: list[PlaceRow]) -> None:
        self.candidates = candidates
        if candidates:
            listed = "; ".join(f"{c.qualified()!r} ({c.kind})" for c in candidates)
            message = f"{asked!r} could be any of: {listed}. Ask again with one of those names."
        else:
            message = f"No barri, district, municipality or venue is called {asked!r}."
        super().__init__(message)


def _names() -> CompoundSelect:
    """One row per name a place or venue answers to. `whole_only` marks a name qualified by where
    it is, which only an exact match may use: its words would otherwise match every venue in a
    town. A venue without a point is nowhere to measure from, so it has no names here."""
    p, v = PLACES.c, VENUES.c

    def row(kind: Any, id_: Any, name: Any, within: Any, geom: Any, label: Any, whole_only: bool):
        return select(
            kind.label("kind"),
            id_.label("id"),
            name.label("name"),
            within.label("within"),
            geom.label("geom"),
            label.label("label"),
            literal(whole_only).label("whole_only"),
        )

    own = func.unnest(func.array_prepend(p.name, p.aliases), type_=Text)
    venue = literal("venue", Text)
    return union_all(
        row(p.kind, p.id, p.name, p.within, p.geom, own, False),
        row(p.kind, p.id, p.name, p.within, p.geom, p.name.concat(", ").concat(p.within), True),
        row(venue, v.id, v.name, v.city, v.geom, v.name, False).where(v.geom.is_not(None)),
        row(venue, v.id, v.name, v.city, v.geom, v.name.concat(", ").concat(v.city), True).where(
            v.geom.is_not(None)
        ),
    )


async def resolve_place(session: AsyncSession, name: str) -> PlaceMatch:
    """The place or venue `name` means, or the candidates when it could mean several far apart
    (rules at the top of this section)."""
    asked = _match_key(literal(name, Text))
    names = _names().subquery("names")
    key = _match_key(names.c.label)
    holds_every_word = func.string_to_array(key, " ").op("@>")(func.string_to_array(asked, " "))
    tier = case(
        (key == asked, 0),
        (
            and_(~names.c.whole_only, func.length(asked) >= MIN_PART_CHARS, holds_every_word),
            1,
        ),
    )
    # One row per place or venue, at the best tier any of its names reached.
    matched = (
        select(
            names.c.kind, names.c.id, names.c.name, names.c.within, names.c.geom, tier.label("tier")
        )
        .where(asked != "", tier.is_not(None))
        .distinct(names.c.kind, names.c.id)
        .order_by(names.c.kind, names.c.id, tier)
        .subquery("matched")
    )
    m = matched.c
    specific = case({kind: rank for rank, kind in enumerate(RESOLVE_KINDS)}, value=m.kind)
    order = [m.tier, specific, func.length(m.name), m.name]
    point = cast(m.geom, Geometry("POINT", srid=4326))
    query = select(
        m.tier,
        m.name,
        m.kind,
        m.within,
        func.ST_Y(point).label("latitude"),
        func.ST_X(point).label("longitude"),
        func.ST_DWithin(
            m.geom, func.first_value(m.geom).over(order_by=order), MERGE_RADIUS_KM * 1000
        ).label("near_first"),
    ).order_by(*order)

    rows = (await session.execute(query)).all()
    best = [row for row in rows if row.tier == rows[0].tier]
    places = [PlaceRow(r.name, r.kind, r.within, r.latitude, r.longitude) for r in best]
    if places and all(row.near_first for row in best):
        return PlaceMatch(place=places[0], candidates=[])
    return PlaceMatch(place=None, candidates=places[:MAX_CANDIDATES])


@dataclass(frozen=True)
class EventRow:
    """One thing to see, as `search_events` lists it: what the model ranks by, ~50 tokens.
    Never a summary: hydrating the shortlist with `get_events` is what that is for."""

    id: str  # x:12, a:7 or g:301
    kind: str  # one of KINDS: what `kinds` filters on
    category: str | None  # GRAF's own label (seminari, festival…), finer than `kind`; NULL off GRAF
    title: str
    venue: str | None  # NULL for a GRAF event with no venue
    start: date | None  # NULL: no start given; an exhibition without one may already be on
    end: date | None  # NULL: an exhibition with no end date, which may still be on
    ends_in_days: int | None  # from date_from to `end`: 0 means it ends on date_from
    distance_km: float | None  # from the point asked about; NULL with no point or no place
    at_venue: bool  # False: shown elsewhere (a fair, another space) or online
    is_free: bool | None  # known only for GRAF events; NULL is unknown, not paid
    artists: list[str]
    media: list[str]
    themes: list[str]


@dataclass(frozen=True)
class SearchResult:
    total: int  # rows that matched; more than len(events) when `limit` cut the list
    events: list[EventRow]
    near: PlaceRow | None = None  # what `near` was taken to mean, so a wrong guess shows


async def search_events(
    session: AsyncSession,
    *,
    date_from: date,
    date_to: date,
    near: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float | None = None,
    kinds: Collection[str] | None = None,
    free_only: bool = False,
    limit: int = 40,
) -> SearchResult:
    """Everything on at some point from `date_from` to `date_to` (days at the venues), from all
    three sources. Dates and distances are worked out here, never by the model (CLAUDE.md § 8).

    `near` names where to measure from: a barri, district, municipality or venue (`resolve_place`),
    or `lat`/`lon` give the point itself. A point gives each row its distance and puts the nearest
    first; `radius_km` then keeps only the rows inside it. A `near` that names nothing, or several
    places far apart, raises UnresolvedPlace with the candidates.

    `kinds` keeps those of the listed KINDS. `free_only` drops what is known to cost money: only
    GRAF knows prices, so an extracted item stays, with `is_free` NULL.

    Rows come nearest first when there is a point, then the soonest to end, open-ended last: if
    `limit` cuts the list, what goes is what is furthest away and on longest. `total` says how
    many matched, so a cut shows.
    """
    if date_from > date_to:
        raise ValueError(f"date_from {date_from} is after date_to {date_to}")
    if (lat is None) != (lon is None):
        raise ValueError("lat and lon come together")
    if near is not None and lat is not None:
        raise ValueError("near or lat and lon, not both")
    if radius_km is not None and near is None and lat is None:
        raise ValueError("radius_km needs near, or lat and lon")
    if kinds is not None and (unknown := set(kinds) - set(KINDS)):
        raise ValueError(f"unknown kinds {sorted(unknown)}; known: {', '.join(KINDS)}")
    origin = None
    if near is not None:
        match = await resolve_place(session, near)
        if match.place is None:
            raise UnresolvedPlace(near, match.candidates)
        origin = match.place
        lat, lon = origin.latitude, origin.longitude

    candidates = union_all(
        _exhibition_candidates(), _event_candidates(), _graf_candidates()
    ).subquery("candidates")
    c = candidates.c
    place = VENUES.alias("place")

    distance_km: ColumnElement[float | None] = cast(null(), Float)
    within: list[ColumnElement[bool]] = []
    order: list[Any] = [c.last_day.nulls_last(), c.first_day.nulls_first(), c.id]
    if lat is not None:
        point = cast(
            func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326), Geography("POINT", srid=4326)
        )
        # Geography: metres on the spheroid, rounded to 100 m.
        distance_km = cast(
            func.round(cast(func.ST_Distance(place.c.geom, point) / 1000, Numeric), 1), Float
        )
        order.insert(0, distance_km.nulls_last())
        if radius_km is not None:
            within.append(func.ST_DWithin(place.c.geom, point, radius_km * 1000))

    query = (
        select(
            c.id,
            c.kind,
            c.category,
            c.title,
            c.venue,
            c.first_day.label("start"),
            c.last_day.label("end"),
            (c.last_day - date_from).label("ends_in_days"),
            distance_km.label("distance_km"),
            c.at_venue,
            c.is_free,
            c.artists,
            c.media,
            c.themes,
            func.count().over().label("total"),  # before LIMIT: every row that matched
        )
        .select_from(candidates.outerjoin(place, place.c.id == c.place_venue_id))
        .where(
            or_(c.first_day.is_(None), c.first_day <= date_to),
            or_(c.last_day.is_(None), c.last_day >= date_from),
            *within,
        )
        .order_by(*order)
        .limit(limit)
    )
    if kinds is not None:
        query = query.where(c.kind.in_(list(kinds)))
    if free_only:
        query = query.where(c.is_free.is_not(False))

    rows = (await session.execute(query)).mappings().all()
    events = [EventRow(**{k: v for k, v in row.items() if k != "total"}) for row in rows]
    return SearchResult(total=rows[0]["total"] if rows else 0, events=events, near=origin)


@dataclass(frozen=True)
class EventDetail:
    """One item hydrated: every fact we hold, our summary, and where it is."""

    id: str
    kind: str
    category: str | None  # GRAF's own label, as in EventRow
    title: str
    title_en: str | None  # GRAF's English title, when it has one
    artists: list[str]
    media: list[str]
    themes: list[str]
    summary_en: str | None  # ours, never the page's (CLAUDE.md § 1); GRAF events have none
    start: date | None
    end: date | None  # an art event without one lasts its start day, as in search_events
    start_time: time | None  # at the venue's clock; art events and GRAF occurrences only
    at_venue: bool
    is_free: bool | None  # GRAF only
    price_min: Decimal | None  # GRAF only
    price_max: Decimal | None
    url: str | None  # its own page: the venue's page for it, or the link GRAF gives
    source_url: str  # where we read it: a venue page, or GRAF's page for the event
    exhibition_id: str | None  # an art event's show, as an x: id, when the page said
    venue_id: int | None
    venue: str | None
    address: str | None
    city: str | None
    website: str | None


_DETAIL = tuple(f.name for f in fields(EventDetail))
_VENUE_COLUMNS = {
    "venue_id": VENUES.c.id,
    "venue": VENUES.c.name,
    "address": VENUES.c.address,
    "city": VENUES.c.city,
    "website": VENUES.c.website_url,
}


def _detail(**columns: Any) -> Select:
    columns |= _VENUE_COLUMNS
    return select(*(columns[name].label(name) for name in _DETAIL))


def _exhibition_details(ids: list[int]) -> Select:
    return (
        _detail(
            id=_prefixed("x", EXHIBITIONS.c.id),
            kind=literal("exhibition"),
            category=cast(null(), Text),
            title=EXHIBITIONS.c.title,
            title_en=cast(null(), Text),
            artists=EXHIBITIONS.c.artists,
            media=EXHIBITIONS.c.media,
            themes=EXHIBITIONS.c.themes,
            summary_en=EXHIBITIONS.c.summary_en,
            start=EXHIBITIONS.c.start_date,
            end=EXHIBITIONS.c.end_date,
            start_time=cast(null(), Time),
            at_venue=EXHIBITIONS.c.at_venue,
            is_free=cast(null(), Boolean),
            price_min=cast(null(), Numeric),
            price_max=cast(null(), Numeric),
            url=EXHIBITIONS.c.detail_url,
            source_url=EXHIBITIONS.c.source_url,
            exhibition_id=cast(null(), Text),
        )
        .join_from(EXHIBITIONS, VENUES, VENUES.c.id == EXHIBITIONS.c.venue_id)
        .where(EXHIBITIONS.c.id.in_(ids))
    )


def _event_details(ids: list[int]) -> Select:
    return (
        _detail(
            id=_prefixed("a", EVENTS.c.id),
            kind=EVENTS.c.kind,
            category=cast(null(), Text),
            title=EVENTS.c.title,
            title_en=cast(null(), Text),
            artists=EVENTS.c.artists,
            media=EVENTS.c.media,
            themes=EVENTS.c.themes,
            summary_en=EVENTS.c.summary_en,
            start=EVENTS.c.start_date,
            end=func.coalesce(EVENTS.c.end_date, EVENTS.c.start_date),
            start_time=EVENTS.c.start_time,
            at_venue=EVENTS.c.at_venue,
            is_free=cast(null(), Boolean),
            price_min=cast(null(), Numeric),
            price_max=cast(null(), Numeric),
            url=EVENTS.c.detail_url,
            source_url=EVENTS.c.source_url,
            exhibition_id=_prefixed("x", EVENTS.c.exhibition_id),
        )
        .join_from(EVENTS, VENUES, VENUES.c.id == EVENTS.c.venue_id)
        .where(EVENTS.c.id.in_(ids))
    )


def _graf_details(ids: list[int]) -> Select:
    return (
        _detail(
            id=_prefixed("g", OCCURRENCES.c.id),
            kind=_graf_kind(),
            category=_graf_category(),
            title=SNAPSHOTS.c.title,
            title_en=SNAPSHOTS.c.title_en,
            artists=_NO_ARRAY,
            media=_NO_ARRAY,
            themes=_NO_ARRAY,
            summary_en=cast(null(), Text),
            start=_graf_first_day(),
            end=_graf_last_day(),
            start_time=_local_time(OCCURRENCES.c.starts_at),
            at_venue=_graf_at_venue(),
            is_free=SNAPSHOTS.c.is_free,
            price_min=SNAPSHOTS.c.price_min,
            price_max=SNAPSHOTS.c.price_max,
            url=func.coalesce(
                SNAPSHOTS.c.web_url_en, SNAPSHOTS.c.web_url_es, SNAPSHOTS.c.web_url_ca
            ),
            source_url=SNAPSHOTS.c.source_url,
            exhibition_id=cast(null(), Text),
        )
        .select_from(
            OCCURRENCES.join(SNAPSHOTS, SNAPSHOTS.c.id == OCCURRENCES.c.event_id).outerjoin(
                VENUES, VENUES.c.id == SNAPSHOTS.c.venue_id
            )
        )
        .where(OCCURRENCES.c.id.in_(ids))
    )


_DETAILS = {"x": _exhibition_details, "a": _event_details, "g": _graf_details}


async def get_events(session: AsyncSession, ids: Collection[str]) -> list[EventDetail]:
    """The items `ids` name (from `search_events`), in the order asked, at most MAX_HYDRATE.
    An id that names nothing is left out; one that is not an id at all raises ValueError.
    Hydrating does not filter: a GRAF event `search_events` hides is still returned by id."""
    parsed = list(dict.fromkeys(map(parse_event_id, ids)))
    if len(parsed) > MAX_HYDRATE:
        raise ValueError(f"{len(parsed)} ids; at most {MAX_HYDRATE} at a time")
    if not parsed:
        return []
    numbers: dict[str, list[int]] = {}
    for prefix, number in parsed:
        numbers.setdefault(prefix, []).append(number)
    wanted = [f"{prefix}:{number}" for prefix, number in parsed]  # x:012 is x:12

    query = union_all(*(_DETAILS[prefix](found) for prefix, found in numbers.items()))
    found = {row.id: row for row in await _rows(session, EventDetail, query)}
    return [found[i] for i in wanted if i in found]


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
