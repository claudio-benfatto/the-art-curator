"""Venue profiles are many-to-one: drop UNIQUE (source, source_profile_id).

A GRAF term is a *space*, a profile an *organisation*, and one organisation can have several
spaces: MACBA is one profile and four terms. 0001 made that unrepresentable.

No index replaces the constraint's: the table holds ~570 rows and a sequential scan is cheaper
than maintaining one.

Downgrade fails once any profile serves more than one venue — which is exactly the state this
revision exists to allow. Unjoin first if you really need to go back.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-01 18:30:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(op.f("uq_venues_source_source_profile_id"), "venues", type_="unique")


def downgrade() -> None:
    op.create_unique_constraint(
        op.f("uq_venues_source_source_profile_id"), "venues", ["source", "source_profile_id"]
    )
