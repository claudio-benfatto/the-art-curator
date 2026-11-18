"""Add `profile_facts`: what the curator remembers about a user from one conversation to the next.

The `remember` tool writes one row per fact: a topic from a small fixed set and one short sentence
in the curator's own words. Append-only: a correction is a new row, and the row it replaces gets
`superseded_by`. A new conversation's context block lists the facts nothing has superseded.

No third-party text: the curator's words about what the user said (`test_no_verbatim.py`).

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-17 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | Sequence[str] | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "profile_facts",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("fact", sa.Text(), nullable=False),
        sa.Column("superseded_by", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "topic IN ('media', 'themes', 'area', 'budget', 'language', 'access', 'other')",
            name=op.f("ck_profile_facts_topic"),
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by"],
            ["profile_facts.id"],
            name=op.f("fk_profile_facts_superseded_by_profile_facts"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_profile_facts_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_profile_facts")),
    )
    op.create_index("ix_profile_facts_user_id", "profile_facts", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_profile_facts_user_id", table_name="profile_facts")
    op.drop_table("profile_facts")
