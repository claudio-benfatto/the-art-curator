"""P0 schema: extensions, GRAF facts, venue_pages cache, llm_calls.

Revision ID: 0001
Revises:
Create Date: 2026-09-22 10:40:35.131915

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The db image only makes these available; creating them here keeps local, CI and RDS on one
    # path. pgvector has no columns until P3 (`exhibitions.embedding`).
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "llm_calls",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("trace_id", sa.Text(), nullable=True),
        sa.Column("span_id", sa.Text(), nullable=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("output_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "cache_creation_input_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "cache_read_input_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("stop_reason", sa.Text(), nullable=True),
        sa.Column("error_type", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "purpose IN ('chat', 'extract', 'embed', 'judge', 'smoke')",
            name=op.f("ck_llm_calls_purpose"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_calls")),
    )
    op.create_index("ix_llm_calls_created_at", "llm_calls", ["created_at"], unique=False)
    op.create_index("ix_llm_calls_trace_id", "llm_calls", ["trace_id"], unique=False)
    op.create_table(
        "venues",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("graf_venue_id", sa.Integer(), nullable=False),
        sa.Column("graf_user_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("address", sa.Text(), nullable=True),
        sa.Column("city", sa.Text(), nullable=True),
        sa.Column("province", sa.Text(), nullable=True),
        sa.Column("postcode", sa.Text(), nullable=True),
        sa.Column(
            "geom",
            Geography("POINT", srid=4326, spatial_index=False),
            nullable=True,
        ),
        sa.Column("website_url", sa.Text(), nullable=True),
        sa.Column("instagram_url", sa.Text(), nullable=True),
        sa.Column("crawl_enabled", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_pilot", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_venues")),
        sa.UniqueConstraint("graf_user_id", name=op.f("uq_venues_graf_user_id")),
        sa.UniqueConstraint("graf_venue_id", name=op.f("uq_venues_graf_venue_id")),
    )
    op.create_index(
        "ix_venues_geom",
        "venues",
        ["geom"],
        unique=False,
        postgresql_using="gist",
    )
    op.create_table(
        "graf_event_snapshots",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("graf_event_id", sa.Integer(), nullable=False),
        sa.Column("venue_id", sa.BigInteger(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("title_en", sa.Text(), nullable=True),
        sa.Column("graf_category_id", sa.Integer(), nullable=True),
        sa.Column("is_free", sa.Boolean(), nullable=True),
        sa.Column("price_min", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("price_max", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("is_online", sa.Boolean(), nullable=True),
        sa.Column("graf_url", sa.Text(), nullable=False),
        sa.Column("web_url_ca", sa.Text(), nullable=True),
        sa.Column("web_url_es", sa.Text(), nullable=True),
        sa.Column("web_url_en", sa.Text(), nullable=True),
        sa.Column("graf_modified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["venue_id"], ["venues.id"], name=op.f("fk_graf_event_snapshots_venue_id_venues")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_graf_event_snapshots")),
        sa.UniqueConstraint("graf_event_id", name=op.f("uq_graf_event_snapshots_graf_event_id")),
    )
    op.create_index(
        "ix_graf_event_snapshots_venue_id", "graf_event_snapshots", ["venue_id"], unique=False
    )
    op.create_table(
        "venue_pages",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("venue_id", sa.BigInteger(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["venue_id"], ["venues.id"], name=op.f("fk_venue_pages_venue_id_venues")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_venue_pages")),
        sa.UniqueConstraint("url", name=op.f("uq_venue_pages_url")),
    )
    op.create_index(
        "ix_venue_pages_purge",
        "venue_pages",
        ["fetched_at"],
        unique=False,
        postgresql_where=sa.text("raw_text IS NOT NULL"),
    )
    op.create_index("ix_venue_pages_venue_id", "venue_pages", ["venue_id"], unique=False)
    op.create_table(
        "graf_event_occurrences",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("event_id", sa.BigInteger(), nullable=False),
        sa.Column("graf_occurrence_id", sa.BigInteger(), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["graf_event_snapshots.id"],
            name=op.f("fk_graf_event_occurrences_event_id_graf_event_snapshots"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_graf_event_occurrences")),
        sa.UniqueConstraint(
            "graf_occurrence_id", name=op.f("uq_graf_event_occurrences_graf_occurrence_id")
        ),
    )
    op.create_index(
        "ix_graf_event_occurrences_event_id", "graf_event_occurrences", ["event_id"], unique=False
    )
    op.create_index(
        "ix_graf_event_occurrences_starts_at", "graf_event_occurrences", ["starts_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_graf_event_occurrences_starts_at", table_name="graf_event_occurrences")
    op.drop_index("ix_graf_event_occurrences_event_id", table_name="graf_event_occurrences")
    op.drop_table("graf_event_occurrences")
    op.drop_index("ix_venue_pages_venue_id", table_name="venue_pages")
    op.drop_index(
        "ix_venue_pages_purge",
        table_name="venue_pages",
        postgresql_where=sa.text("raw_text IS NOT NULL"),
    )
    op.drop_table("venue_pages")
    op.drop_index("ix_graf_event_snapshots_venue_id", table_name="graf_event_snapshots")
    op.drop_table("graf_event_snapshots")
    op.drop_index("ix_venues_geom", table_name="venues")
    op.drop_table("venues")
    op.drop_index("ix_llm_calls_trace_id", table_name="llm_calls")
    op.drop_index("ix_llm_calls_created_at", table_name="llm_calls")
    op.drop_table("llm_calls")
    op.execute("DROP EXTENSION IF EXISTS vector")
    op.execute("DROP EXTENSION IF EXISTS postgis")
