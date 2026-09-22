"""ORM models. Each table lands in the phase that first writes it (PLAN.md § 4).

CLAUDE.md § 1: source-fact tables hold facts only — no description/summary/body columns.
Third-party text exists only in `venue_pages.raw_text`, a 7-day processing cache.
`tests/test_no_verbatim.py` enforces both.

Every fact table carries a `source` column (currently always `"graf"`) so a second event
source can land without renaming columns or widening uniqueness after the fact — external
ids are only unique per source, not globally.
"""

from datetime import datetime
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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

LLM_PURPOSES = ("chat", "extract", "embed", "judge", "smoke")


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
        UniqueConstraint("source", "source_profile_id"),
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
    """A crawled venue page. `raw_text` is trafilatura's main body and is purged (set NULL) after
    7 days; the row survives so `content_hash` can still skip unchanged pages."""

    __tablename__ = "venue_pages"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    venue_id: Mapped[int] = mapped_column(ForeignKey("venues.id"))
    url: Mapped[str] = mapped_column(unique=True)
    http_status: Mapped[int | None]
    content_hash: Mapped[str | None]  # sha256 hex of raw_text
    raw_text: Mapped[str | None]  # 7-day TTL — the only third-party prose in the database
    fetched_at: Mapped[datetime]
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        Index("ix_venue_pages_venue_id", "venue_id"),
        # Drives the purge job: only rows still holding text need scanning.
        Index(
            "ix_venue_pages_purge",
            "fetched_at",
            postgresql_where=text("raw_text IS NOT NULL"),
        ),
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
