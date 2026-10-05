"""Allow `discover` in `llm_calls.purpose`.

Discovery (P2) is a model call of its own kind: monthly, per site, and budgeted separately from
extraction. Counting it under `extract` would hide exactly the cost the plan asks to see.

Downgrade fails once a `discover` row exists. That is intended: `llm_calls` is the record of
spend, so rows are relabelled by hand if at all, never dropped by a migration.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-05 10:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BEFORE = "purpose IN ('chat', 'extract', 'embed', 'judge', 'smoke')"
AFTER = "purpose IN ('chat', 'discover', 'extract', 'embed', 'judge', 'smoke')"


def upgrade() -> None:
    op.drop_constraint(op.f("ck_llm_calls_purpose"), "llm_calls", type_="check")
    op.create_check_constraint(op.f("ck_llm_calls_purpose"), "llm_calls", AFTER)


def downgrade() -> None:
    op.drop_constraint(op.f("ck_llm_calls_purpose"), "llm_calls", type_="check")
    op.create_check_constraint(op.f("ck_llm_calls_purpose"), "llm_calls", BEFORE)
