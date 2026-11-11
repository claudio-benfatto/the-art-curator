"""Add `places`: Barcelona barris and districts and Catalan municipalities, one point each.

`search_events(near=...)` turns a place name into a point in SQL (`queries/events.resolve_place`),
so the model never guesses coordinates (CLAUDE.md § 8). `cli load-places` fills the table from
open data and replaces it whole each run: nothing references a place by id.

Facts only: a name, the source's code, other names it answers to, what it lies within, and a point.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-10 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "places",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "aliases", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column("within", sa.Text(), nullable=True),
        sa.Column("geom", Geography("POINT", srid=4326, spatial_index=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('barri', 'district', 'municipality')", name=op.f("ck_places_kind")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_places")),
        sa.UniqueConstraint("kind", "code", name=op.f("uq_places_kind_code")),
    )


def downgrade() -> None:
    op.drop_table("places")
