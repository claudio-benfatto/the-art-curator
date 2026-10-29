"""Add `venue_pages.links`: a listing page's numbered same-site links.

Extraction names each show's own page by link *number*, so that it cannot invent a URL, and
`raw_text` is visible text with every `href` gone. The numbered list has to be built from the HTML,
which exists only while the page is being crawled, so `crawl` stores it here for `extract` to read.

Each entry is `{number, url, text}`. The link text is third-party, like `raw_text`, and lives
under the same 7-day TTL: the purge NULLs both, and the CHECK means links can never outlive the
text they were found beside (CLAUDE.md § 1).

Nullable with no backfill. Existing rows get their links on the next crawl.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-10 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "venue_pages",
        sa.Column("links", postgresql.JSONB(none_as_null=True, astext_type=sa.Text())),
    )
    op.create_check_constraint(
        op.f("ck_venue_pages_links_with_text"),
        "venue_pages",
        "links IS NULL OR raw_text IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_venue_pages_links_with_text"), "venue_pages", type_="check")
    op.drop_column("venue_pages", "links")
