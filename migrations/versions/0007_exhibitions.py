"""Add `exhibitions` and `art_events`, and what extraction leaves on `venue_pages`.

`cli extract` (P2) reads each `ok` page's text, and keeps what the model found in it: facts
(titles, artists, dates, the show's own page) and an English summary that is ours, checked by the
verbatim guard before it is written. Neither table has a column for anything the page itself
said (CLAUDE.md § 1). A show seen on several pages is one row, keyed on the venue and its
normalised title; a recurring event is one row per start date.

On `venue_pages`:

- `kind`: `listing` (a seed or pin) or `detail` (a show's own page, named by extraction). Every
  row before this revision is a listing, which the default backfills.
- `extract_status`, `extracted_hash`, `extract_version`, `extract_model`, `extracted_items`,
  `extracted_at`: what the last extraction ran with. A page whose hash, version and model all
  match is not sent to a model again. Like `content_hash`, they outlive the purge, and they hold
  no text.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-15 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONFIDENCE = "confidence IN ('high', 'medium', 'low')"
MEDIA = (
    "media <@ ARRAY['painting', 'drawing', 'sculpture', 'installation', 'photography', 'video', "
    "'film', 'sound', 'performance', 'print', 'textile', 'ceramics', 'digital', 'architecture', "
    "'design', 'mixed_media', 'other']::text[]"
)
EVENT_KIND = (
    "kind IN ('opening', 'talk', 'workshop', 'performance', 'screening', 'guided_tour', "
    "'concert', 'fair', 'other')"
)


def _item_columns() -> list[sa.Column]:
    """The columns `exhibitions` and `art_events` share (`db.models._Extracted`)."""
    now = sa.text("now()")
    empty = sa.text("'{}'")
    return [
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("venue_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("title_key", sa.Text(), nullable=False),
        sa.Column("artists", postgresql.ARRAY(sa.Text()), server_default=empty, nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("at_venue", sa.Boolean(), nullable=False),
        sa.Column("detail_url", sa.Text(), nullable=True),
        sa.Column("summary_en", sa.Text(), nullable=True),
        sa.Column("themes", postgresql.ARRAY(sa.Text()), server_default=empty, nullable=False),
        sa.Column("media", postgresql.ARRAY(sa.Text()), server_default=empty, nullable=False),
        sa.Column("confidence", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("extract_version", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=now, nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "exhibitions",
        *_item_columns(),
        sa.CheckConstraint(CONFIDENCE, name=op.f("ck_exhibitions_confidence")),
        sa.CheckConstraint(MEDIA, name=op.f("ck_exhibitions_media")),
        sa.ForeignKeyConstraint(
            ["venue_id"], ["venues.id"], name=op.f("fk_exhibitions_venue_id_venues")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_exhibitions")),
        sa.UniqueConstraint(
            "venue_id", "title_key", name=op.f("uq_exhibitions_venue_id_title_key")
        ),
    )
    op.create_table(
        "art_events",
        *_item_columns(),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("start_time", sa.Time(), nullable=True),
        sa.Column("exhibition_id", sa.BigInteger(), nullable=True),
        sa.CheckConstraint(CONFIDENCE, name=op.f("ck_art_events_confidence")),
        sa.CheckConstraint(MEDIA, name=op.f("ck_art_events_media")),
        sa.CheckConstraint(EVENT_KIND, name=op.f("ck_art_events_kind")),
        sa.ForeignKeyConstraint(
            ["venue_id"], ["venues.id"], name=op.f("fk_art_events_venue_id_venues")
        ),
        sa.ForeignKeyConstraint(
            ["exhibition_id"],
            ["exhibitions.id"],
            name=op.f("fk_art_events_exhibition_id_exhibitions"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_art_events")),
        sa.UniqueConstraint(
            "venue_id",
            "title_key",
            "start_date",
            name=op.f("uq_art_events_venue_id_title_key_start_date"),
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_index(op.f("ix_art_events_exhibition_id"), "art_events", ["exhibition_id"])

    op.add_column(
        "venue_pages",
        sa.Column("kind", sa.Text(), server_default=sa.text("'listing'"), nullable=False),
    )
    for name, type_ in [
        ("extract_status", sa.Text()),
        ("extracted_hash", sa.Text()),
        ("extract_version", sa.Integer()),
        ("extract_model", sa.Text()),
        ("extracted_items", sa.Integer()),
        ("extracted_at", sa.DateTime(timezone=True)),
    ]:
        op.add_column("venue_pages", sa.Column(name, type_, nullable=True))
    op.create_check_constraint(
        op.f("ck_venue_pages_kind"), "venue_pages", "kind IN ('listing', 'detail')"
    )
    op.create_check_constraint(
        op.f("ck_venue_pages_extract_status"), "venue_pages", "extract_status IN ('ok', 'failed')"
    )
    op.create_check_constraint(
        op.f("ck_venue_pages_extract_marker"),
        "venue_pages",
        "extract_status IS NULL OR (extracted_hash IS NOT NULL AND extract_version IS NOT NULL"
        " AND extract_model IS NOT NULL AND extracted_at IS NOT NULL)",
    )


def downgrade() -> None:
    for name in ("extract_marker", "extract_status", "kind"):
        op.drop_constraint(op.f(f"ck_venue_pages_{name}"), "venue_pages", type_="check")
    for name in (
        "extracted_at",
        "extracted_items",
        "extract_model",
        "extract_version",
        "extracted_hash",
        "extract_status",
        "kind",
    ):
        op.drop_column("venue_pages", name)
    op.drop_index(op.f("ix_art_events_exhibition_id"), table_name="art_events")
    op.drop_table("art_events")
    op.drop_table("exhibitions")
