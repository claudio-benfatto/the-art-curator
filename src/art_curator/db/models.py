"""ORM models. Each table lands in the phase that first writes it (PLAN.md § 4).

CLAUDE.md § 1: GRAF-sourced tables hold facts only — no description/summary/body columns.
Third-party text exists only in `venue_pages.raw_text`, a 7-day processing cache.
`tests/test_no_verbatim.py` enforces both.
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


# --- Facts (GRAF) -------------------------------------------------------------------------------


class Venue(Base):
    """A GRAF `event-venues` term, joined in P1 to the venue's profile (`users`) for its URL."""

    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    graf_venue_id: Mapped[int] = mapped_column(unique=True)  # event-venues term id
    graf_user_id: Mapped[int | None] = mapped_column(unique=True)  # venue profile, if matched
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

    __table_args__ = (Index("ix_venues_geom", "geom", postgresql_using="gist"),)


class GrafEventSnapshot(Base):
    """One row per GRAF event (WordPress post) ever seen. `/events` is a live window, not an
    archive, so this table is the only history we have (CLAUDE.md § The GRAF API).

    Dates live on `graf_event_occurrences`, not here: an event has one or more occurrences.
    """

    __tablename__ = "graf_event_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    graf_event_id: Mapped[int] = mapped_column(unique=True)  # post id; stable across edits
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"))
    # Titles are persisted as dedup identifiers — decided, not an oversight (CLAUDE.md § 1).
    title: Mapped[str]  # WordPress `title.rendered`
    title_en: Mapped[str | None]
    graf_category_id: Mapped[int | None]  # acf.event_category (taxonomy term id)
    is_free: Mapped[bool | None]
    price_min: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    price_max: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    is_online: Mapped[bool | None]
    graf_url: Mapped[str]
    web_url_ca: Mapped[str | None]
    web_url_es: Mapped[str | None]
    web_url_en: Mapped[str | None]
    graf_modified_at: Mapped[datetime | None]
    first_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("ix_graf_event_snapshots_venue_id", "venue_id"),)


class GrafEventOccurrence(Base):
    """When a GRAF event happens (`/events/{id}/occurrences`). A multi-week exhibition is one
    occurrence spanning its run; a recurring event would have several.

    If GRAF ever regenerates an occurrence (e.g. on a date edit), the old row stays behind with
    stale dates. `last_seen_at` is how the sync tells a superseded occurrence from a current one.
    """

    __tablename__ = "graf_event_occurrences"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("graf_event_snapshots.id", ondelete="CASCADE"))
    graf_occurrence_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    starts_at: Mapped[datetime]
    ends_at: Mapped[datetime | None]
    first_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("ix_graf_event_occurrences_event_id", "event_id"),
        Index("ix_graf_event_occurrences_starts_at", "starts_at"),
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
