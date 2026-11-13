"""Add `users`, `conversations`, `messages` and `feedback`: conversations kept and replayed.

Until now a conversation lived in `cli chat`'s memory. Kept here, a turn is the stored history plus
the user's text, whichever process serves it, which is what lets the API (PR 34) be stateless.

- `users`: a client's opaque name for itself and its id for the person, nothing more.
- `conversations`: the context block a conversation opened with, frozen, and the local day its
  calendar starts from. At most one open conversation a user (a partial unique index); it ends
  after 6 h without a turn or at local midnight, and `ended_at` records when it was closed.
- `messages`: one row per API message, content blocks exactly as sent and received. `json`, not
  `jsonb`, because `jsonb` reorders keys and the replayed request must start with the last one
  byte for byte, or the cache misses from that point on.
- `feedback`: `good` or `bad` on a reply, one a reply.

No third-party text: user and curator words, and tool results, which carry facts and our
summaries only (`test_no_verbatim.py`).

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-11 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("client", sa.Text(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("client", "external_id", name=op.f("uq_users_client_external_id")),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("context", sa.Text(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column(
            "last_turn_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_conversations_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.create_index(
        "uq_conversations_open_user_id",
        "conversations",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("ended_at IS NULL"),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("trace_id", sa.Text(), nullable=True),
        sa.Column("iteration_count", sa.Integer(), nullable=True),
        _created_at(),
        sa.CheckConstraint("role IN ('user', 'assistant')", name=op.f("ck_messages_role")),
        sa.CheckConstraint(
            "iteration_count IS NULL OR role = 'assistant'",
            name=op.f("ck_messages_iteration_count_on_reply"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_messages_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint(
            "conversation_id", "position", name=op.f("uq_messages_conversation_id_position")
        ),
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("message_id", sa.BigInteger(), nullable=False),
        sa.Column("rating", sa.Text(), nullable=False),
        _created_at(),
        sa.CheckConstraint("rating IN ('good', 'bad')", name=op.f("ck_feedback_rating")),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_feedback_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feedback")),
        sa.UniqueConstraint("message_id", name=op.f("uq_feedback_message_id")),
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("messages")
    op.drop_index("uq_conversations_open_user_id", table_name="conversations")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_table("conversations")
    op.drop_table("users")
