"""ORM models. Each table lands in the phase that first writes it (PLAN.md § 4).

CLAUDE.md § 1: source-fact tables hold facts only — no description/summary/body columns.
Third-party text exists only in `venue_pages.raw_text`, a 7-day processing cache.
`tests/test_no_verbatim.py` enforces both.

Every fact table carries a `source` column (currently always `"graf"`) so a second event
source can land without renaming columns or widening uniqueness after the fact — external
ids are only unique per source, not globally.
"""

from datetime import date, datetime, time
from decimal import Decimal

from geoalchemy2 import Geography, WKBElement
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    MetaData,
    Numeric,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

LLM_PURPOSES = ("chat", "discover", "extract", "embed", "judge", "smoke")

# What `ingest/discover.py` can say about a page. `tests/test_seeds.py` checks these against its
# Literals, so a new page type cannot reach the database without a migration.
SEED_PAGE_TYPES = ("current_listing", "agenda", "past_archive", "single_show", "other")
SEED_CONFIDENCES = ("high", "medium", "low")
SEED_LANGUAGES = ("ca", "es", "en", "other")
SEED_STATUSES = ("accepted", "rejected", "ambiguous")

# What a fetch can leave on a `venue_pages` row: `ingest/pages.PageStatus` without `unavailable`,
# which is no answer at all and is never written (`tests/test_crawl.py` checks the pair).
PAGE_STATUSES = ("ok", "thin", "blocked", "robots", "error")
PAGE_KINDS = ("listing", "detail")  # a seed or pin, or a show's own page named by extraction
EXTRACT_STATUSES = ("ok", "failed")

# What `llm/extract.py` can answer with. `tests/test_extraction.py` checks these against its
# Literals, so a new value cannot reach the database without a migration.
EXTRACT_CONFIDENCES = ("high", "medium", "low")
MEDIA = (
    "painting", "drawing", "sculpture", "installation", "photography", "video", "film", "sound",
    "performance", "print", "textile", "ceramics", "digital", "architecture", "design",
    "mixed_media", "other",
)  # fmt: skip
EVENT_KINDS = (
    "opening", "talk", "workshop", "performance", "screening", "guided_tour", "concert", "fair",
    "other",
)  # fmt: skip


def _one_of(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def _all_of(column: str, values: tuple[str, ...]) -> str:
    """Every element of the array `column` is one of `values`."""
    return f"{column} <@ ARRAY[" + ", ".join(f"'{v}'" for v in values) + "]::text[]"


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    type_annotation_map = {datetime: DateTime(timezone=True), str: Text}


def _created_at() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


def _source() -> Mapped[str]:
    return mapped_column(server_default=text("'graf'"))


# --- Facts (source) ------------------------------------------------------------------------------


class Venue(Base):
    """A venue term from an event source, joined in P1 to the venue's profile for its URL.

    `source` identifies which event source `source_venue_id` / `source_profile_id` are scoped to
    (today, always GRAF's `event-venues` term id / matched `users` profile id).

    A term is a *space* and a profile an *organisation*, so many rows may share one
    `source_profile_id` — MACBA is one profile and four terms (migration 0002). The join, the URLs,
    `crawl_enabled` and `is_pilot` are derived, and `ingest/sync.py` recomputes them every run.
    """

    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    source: Mapped[str] = _source()
    source_venue_id: Mapped[int]  # event-venues term id, scoped to `source`
    source_profile_id: Mapped[int | None]  # venue profile, if matched; scoped to `source`
    name: Mapped[str]
    slug: Mapped[str]
    address: Mapped[str | None]
    city: Mapped[str | None]
    province: Mapped[str | None]  # GRAF `state`
    postcode: Mapped[str | None]
    # Geography, not geometry: ST_DWithin / ST_Distance then work in metres (CLAUDE.md § 8).
    geom: Mapped[WKBElement | None] = mapped_column(
        Geography("POINT", srid=4326, spatial_index=False)
    )
    website_url: Mapped[str | None]
    instagram_url: Mapped[str | None]
    crawl_enabled: Mapped[bool] = mapped_column(server_default=text("false"))
    is_pilot: Mapped[bool] = mapped_column(server_default=text("false"))
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_venues_geom", "geom", postgresql_using="gist"),
        UniqueConstraint("source", "source_venue_id"),
    )


class EventSnapshot(Base):
    """One row per source event (a GRAF WordPress post, today) ever seen. `/events` is a live
    window, not an archive, so this table is the only history we have (CLAUDE.md § The GRAF API).

    Dates live on `event_occurrences`, not here: an event has one or more occurrences.
    `source_event_id` is stable across edits but only unique within `source`.
    """

    __tablename__ = "event_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    source: Mapped[str] = _source()
    source_event_id: Mapped[int]  # post id; stable across edits, scoped to `source`
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"))
    # Titles are persisted as dedup identifiers — decided, not an oversight (CLAUDE.md § 1).
    title: Mapped[str]  # WordPress `title.rendered`
    title_en: Mapped[str | None]
    source_category_id: Mapped[int | None]  # acf.event_category (taxonomy term id)
    is_free: Mapped[bool | None]
    price_min: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    price_max: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    is_online: Mapped[bool | None]
    source_url: Mapped[str]
    web_url_ca: Mapped[str | None]
    web_url_es: Mapped[str | None]
    web_url_en: Mapped[str | None]
    source_modified_at: Mapped[datetime | None]
    first_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("ix_event_snapshots_venue_id", "venue_id"),
        UniqueConstraint("source", "source_event_id"),
    )


class EventOccurrence(Base):
    """When a source event happens (GRAF's `/events/{id}/occurrences`, today). A multi-week
    exhibition is one occurrence spanning its run; a recurring event would have several.

    If the source ever regenerates an occurrence (e.g. on a date edit), the old row stays behind
    with stale dates. `last_seen_at` is how the sync tells a superseded occurrence from a current
    one. `source_occurrence_id` is only unique within `source`.
    """

    __tablename__ = "event_occurrences"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    source: Mapped[str] = _source()
    event_id: Mapped[int] = mapped_column(ForeignKey("event_snapshots.id", ondelete="CASCADE"))
    source_occurrence_id: Mapped[int] = mapped_column(BigInteger)
    starts_at: Mapped[datetime]
    ends_at: Mapped[datetime | None]
    first_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("ix_event_occurrences_event_id", "event_id"),
        Index("ix_event_occurrences_starts_at", "starts_at"),
        UniqueConstraint("source", "source_occurrence_id"),
    )


# --- Derived (processing cache) -----------------------------------------------------------------


class VenuePage(Base):
    """A crawled venue page. `raw_text` is all visible text (`ingest/pages.py`) and is
    purged (set NULL) after 7 days; the row survives so `content_hash` can still skip unchanged
    pages. Only an `ok` page holds text or a hash: a page we could not use is a status and nothing
    else.

    `links` is a listing page's numbered same-site links, `[{number, url, text}]`, which
    extraction answers with to name a show's own page. Their text is third-party too, so they
    are purged with `raw_text` and cannot exist without it.

    The `extract*` columns say what the last extraction saw: the hash, `EXTRACT_VERSION` and model
    it ran with. A page is extracted again only when one of the three moves, including a page
    that failed, so an unchanged page is never paid for twice. They outlive the purge, like
    `content_hash`."""

    __tablename__ = "venue_pages"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    venue_id: Mapped[int] = mapped_column(ForeignKey("venues.id"))
    url: Mapped[str] = mapped_column(unique=True)
    kind: Mapped[str] = mapped_column(server_default=text("'listing'"))  # one of PAGE_KINDS
    status: Mapped[str]  # what the last fetch found; one of PAGE_STATUSES
    http_status: Mapped[int | None]
    content_hash: Mapped[str | None]  # sha256 hex of raw_text
    raw_text: Mapped[str | None]  # 7-day TTL — the only third-party prose in the database
    # none_as_null: a page without links is SQL NULL, not a JSON `null` the CHECK would count.
    links: Mapped[list[dict[str, str | int]] | None] = mapped_column(JSONB(none_as_null=True))
    fetched_at: Mapped[datetime]
    extract_status: Mapped[str | None]  # one of EXTRACT_STATUSES; NULL until first extracted
    extracted_hash: Mapped[str | None]  # content_hash the last extraction read
    extract_version: Mapped[int | None]
    extract_model: Mapped[str | None]
    extracted_items: Mapped[int | None]  # exhibitions + events; NULL when it failed
    extracted_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        Index("ix_venue_pages_venue_id", "venue_id"),
        # Drives the purge job: only rows still holding text need scanning.
        Index(
            "ix_venue_pages_purge",
            "fetched_at",
            postgresql_where=text("raw_text IS NOT NULL"),
        ),
        CheckConstraint(_one_of("status", PAGE_STATUSES), name="status"),
        CheckConstraint("links IS NULL OR raw_text IS NOT NULL", name="links_with_text"),
        CheckConstraint(_one_of("kind", PAGE_KINDS), name="kind"),
        CheckConstraint(_one_of("extract_status", EXTRACT_STATUSES), name="extract_status"),
        CheckConstraint(
            "extract_status IS NULL OR (extracted_hash IS NOT NULL AND extract_version IS NOT NULL"
            " AND extract_model IS NOT NULL AND extracted_at IS NOT NULL)",
            name="extract_marker",
        ),
    )


class VenueSeed(Base):
    """What `discover` concluded about one candidate listing page: the machine's state. Human
    decisions live in `ingest/seeds.yaml` and are never written here.

    Every column is an enum, a count, a URL or an id. There is deliberately no reasoning column:
    the pass that fills this table reads third-party page text, and anything it wrote in prose
    could quote it (CLAUDE.md § 1).
    """

    __tablename__ = "venue_seeds"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    venue_id: Mapped[int] = mapped_column(ForeignKey("venues.id"))
    url: Mapped[str]
    page_type: Mapped[str]
    confidence: Mapped[str]
    dated_items: Mapped[int]
    language: Mapped[str]
    status: Mapped[str]
    model: Mapped[str]
    discovered_at: Mapped[datetime] = mapped_column(server_default=func.now())
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        UniqueConstraint("venue_id", "url"),
        CheckConstraint(_one_of("page_type", SEED_PAGE_TYPES), name="page_type"),
        CheckConstraint(_one_of("confidence", SEED_CONFIDENCES), name="confidence"),
        CheckConstraint(_one_of("language", SEED_LANGUAGES), name="language"),
        CheckConstraint(_one_of("status", SEED_STATUSES), name="status"),
        CheckConstraint("dated_items >= 0", name="dated_items"),
    )


class _Extracted:
    """What `exhibitions` and `art_events` share: one item as extraction read it off a venue page.

    Facts (title, artists, dates, `detail_url`) plus `summary_en`, which is **ours**: written by
    the model, and checked by `extract.longest_shared_run` before it gets here (CLAUDE.md § 1).
    There is no column for anything the page itself said. `ingest/extraction.py` merges a
    show seen on several pages into one row; `source_url`, `content_hash`, `model` and
    `extract_version` say which page and which run wrote it last.
    """

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    venue_id: Mapped[int] = mapped_column(ForeignKey("venues.id"))
    # Titles are persisted as dedup identifiers — decided, not an oversight (CLAUDE.md § 1).
    title: Mapped[str]
    title_key: Mapped[str]  # `normalize_name(title)`: what makes two titles one show
    artists: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    start_date: Mapped[date | None]
    end_date: Mapped[date | None]
    at_venue: Mapped[bool]
    detail_url: Mapped[str | None]
    summary_en: Mapped[str | None]  # ours, never the page's; NULL when the guard dropped it
    themes: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    media: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    confidence: Mapped[str]
    source_url: Mapped[str]
    content_hash: Mapped[str]
    model: Mapped[str]
    extract_version: Mapped[int]
    first_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Exhibition(_Extracted, Base):
    __tablename__ = "exhibitions"

    __table_args__ = (
        UniqueConstraint("venue_id", "title_key"),
        CheckConstraint(_one_of("confidence", EXTRACT_CONFIDENCES), name="confidence"),
        CheckConstraint(_all_of("media", MEDIA), name="media"),
    )


class ArtEvent(_Extracted, Base):
    """An opening, talk, workshop… A guided tour repeats under one title, so the start date is
    part of the key, and a missing one counts as a value (`NULLS NOT DISTINCT`)."""

    __tablename__ = "art_events"

    kind: Mapped[str]
    start_time: Mapped[time | None]
    exhibition_id: Mapped[int | None] = mapped_column(
        ForeignKey("exhibitions.id", ondelete="SET NULL")
    )

    __table_args__ = (
        UniqueConstraint("venue_id", "title_key", "start_date", postgresql_nulls_not_distinct=True),
        Index("ix_art_events_exhibition_id", "exhibition_id"),
        CheckConstraint(_one_of("confidence", EXTRACT_CONFIDENCES), name="confidence"),
        CheckConstraint(_all_of("media", MEDIA), name="media"),
        CheckConstraint(_one_of("kind", EVENT_KINDS), name="kind"),
    )


# --- Telemetry ----------------------------------------------------------------------------------


class LlmCall(Base):
    """One row per model call, written by `llm/client.py` and nothing else (CLAUDE.md § 2).
    Holds metadata only — never prompt or completion bodies."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    created_at: Mapped[datetime] = _created_at()
    trace_id: Mapped[str | None]  # OTel, hex
    span_id: Mapped[str | None]
    provider: Mapped[str]  # which Bedrock endpoint served it; values are set by llm/client.py
    model: Mapped[str]
    purpose: Mapped[str]
    request_id: Mapped[str | None]  # provider request id, for AWS-side lookup
    input_tokens: Mapped[int] = mapped_column(server_default=text("0"))
    output_tokens: Mapped[int] = mapped_column(server_default=text("0"))
    cache_creation_input_tokens: Mapped[int] = mapped_column(server_default=text("0"))
    cache_read_input_tokens: Mapped[int] = mapped_column(server_default=text("0"))
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int]
    stop_reason: Mapped[str | None]
    error_type: Mapped[str | None]  # exception class on failure; failed calls are recorded too

    __table_args__ = (
        CheckConstraint(
            "purpose IN (" + ", ".join(f"'{p}'" for p in LLM_PURPOSES) + ")", name="purpose"
        ),
        Index("ix_llm_calls_created_at", "created_at"),
        Index("ix_llm_calls_trace_id", "trace_id"),
    )
