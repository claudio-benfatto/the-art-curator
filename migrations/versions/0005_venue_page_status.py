"""Add `venue_pages.status`: what the last fetch of a page found.

`crawl` (P2) records a page it could not use as well as one it could: `thin` (rendered by
JavaScript), `blocked`, `robots`, `error`. Without the column those are indistinguishable from an
`ok` page whose text was purged, and a broken seed would leave no trace between runs.

An enum, like every `venue_seeds` column: there is nowhere here to put a sentence (CLAUDE.md § 1).

NOT NULL with no default. Nothing wrote `venue_pages` before this revision, so there are no rows
to backfill.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-07 20:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("venue_pages", sa.Column("status", sa.Text(), nullable=False))
    op.create_check_constraint(
        op.f("ck_venue_pages_status"),
        "venue_pages",
        "status IN ('ok', 'thin', 'blocked', 'robots', 'error')",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_venue_pages_status"), "venue_pages", type_="check")
    op.drop_column("venue_pages", "status")
