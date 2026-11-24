"""Add `embedding` and its markers to `exhibitions` and `art_events`: what semantic ranking reads.

`cli embed` (P3, PLAN.md § 8) turns each item's summary, themes and media into one vector and
keeps it here, with the model that made it (`embed_model`) and the hash of the text it was made
from (`embedded_hash`). An item whose text and model are unchanged is not embedded again.
`search_events(query=...)` ranks only by vectors of the model the query was embedded with.

Only our own text is embedded, never a page's (CLAUDE.md § 1): `summary_en` is ours, and themes
and media are enums and short tags. GRAF events have no summary, so no vector.

No index: a few hundred rows are an exact scan, faster than an HNSW lookup would be to explain.

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-20 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: str | Sequence[str] | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("exhibitions", "art_events")
EMBED_DIMENSIONS = 1024  # `db.models.EMBED_DIMENSIONS` when this revision was written
EMBED_MARKER = (
    "(embedding IS NULL) = (embed_model IS NULL) AND (embedding IS NULL) = (embedded_hash IS NULL)"
)


def upgrade() -> None:
    for table in TABLES:
        op.add_column(table, sa.Column("embedding", VECTOR(EMBED_DIMENSIONS), nullable=True))
        op.add_column(table, sa.Column("embed_model", sa.Text(), nullable=True))
        op.add_column(table, sa.Column("embedded_hash", sa.Text(), nullable=True))
        op.create_check_constraint(op.f(f"ck_{table}_embed_marker"), table, EMBED_MARKER)


def downgrade() -> None:
    for table in TABLES:
        op.drop_constraint(op.f(f"ck_{table}_embed_marker"), table, type_="check")
        for column in ("embedded_hash", "embed_model", "embedding"):
            op.drop_column(table, column)
