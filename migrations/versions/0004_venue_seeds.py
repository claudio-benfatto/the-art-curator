"""Add `venue_seeds`: what discovery concluded about each candidate listing page.

One row per (venue, URL) the model was shown. Enums, a count and ids only: the pass that writes
this table reads third-party page text, so it has no column a sentence could go in (CLAUDE.md § 1).
Human decisions are not here; they live in `ingest/seeds.yaml`.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-07 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "venue_seeds",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("venue_id", sa.BigInteger(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("page_type", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Text(), nullable=False),
        sa.Column("dated_items", sa.Integer(), nullable=False),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column(
            "discovered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "page_type IN ('current_listing', 'agenda', 'past_archive', 'single_show', 'other')",
            name=op.f("ck_venue_seeds_page_type"),
        ),
        sa.CheckConstraint(
            "confidence IN ('high', 'medium', 'low')", name=op.f("ck_venue_seeds_confidence")
        ),
        sa.CheckConstraint(
            "language IN ('ca', 'es', 'en', 'other')", name=op.f("ck_venue_seeds_language")
        ),
        sa.CheckConstraint(
            "status IN ('accepted', 'rejected', 'ambiguous')", name=op.f("ck_venue_seeds_status")
        ),
        sa.CheckConstraint("dated_items >= 0", name=op.f("ck_venue_seeds_dated_items")),
        sa.ForeignKeyConstraint(
            ["venue_id"], ["venues.id"], name=op.f("fk_venue_seeds_venue_id_venues")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_venue_seeds")),
        sa.UniqueConstraint("venue_id", "url", name=op.f("uq_venue_seeds_venue_id_url")),
    )


def downgrade() -> None:
    op.drop_table("venue_seeds")
