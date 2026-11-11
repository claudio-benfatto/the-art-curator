"""Conversations kept in Postgres and replayed from it, so any process can serve the next turn.

A turn is: `current` loads the user's open conversation (or opens one), `Curator.turn` answers on
it, and `save` appends what the turn added. The history comes back exactly as it was stored, so
turn n+1's request starts with turn n's, byte for byte, and the cache holds (CLAUDE.md § 6).

A conversation ends after `IDLE` without a turn, or once its day is over in SITE_TIMEZONE: its
context block holds a calendar, and a calendar cannot outlive its day (PLAN.md § 8, Dates). The
next turn then opens a new one; nothing carries over until the profile (PR 30) does.

Two turns on one conversation at once are not serialised: the second to save collides with the
first on `(conversation_id, position)` and fails, leaving the first's.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from art_curator.agent.loop import Conversation, Turn
from art_curator.agent.prompt import conversation_context
from art_curator.db.models import Conversation as ConversationRow
from art_curator.db.models import Feedback, Message, User

IDLE = timedelta(hours=6)

Rating = Literal["good", "bad"]


@dataclass
class StoredConversation:
    id: int
    conversation: Conversation
    saved: int  # how many of `conversation.messages` are rows already


class ConversationStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def user(self, client: str, external_id: str) -> int:
        """The id of the user `client` knows as `external_id`, created on first sight."""
        async with self._sessions() as session, session.begin():
            await session.execute(
                insert(User)
                .values(client=client, external_id=external_id)
                .on_conflict_do_nothing(index_elements=["client", "external_id"])
            )
            found = await session.execute(
                select(User.id).where(User.client == client, User.external_id == external_id)
            )
            return found.scalar_one()

    async def current(self, user_id: int, today: date) -> StoredConversation:
        """The user's open conversation, replayed from its messages; or, if it has expired or
        there is none, a new one whose calendar starts at `today`."""
        async with self._sessions() as session, session.begin():
            found = (
                await session.execute(
                    select(ConversationRow, ConversationRow.last_turn_at < func.now() - IDLE)
                    .where(ConversationRow.user_id == user_id, ConversationRow.ended_at.is_(None))
                    .with_for_update()
                )
            ).one_or_none()
            if found is not None:
                row, idle = found
                if not idle and row.day == today:
                    return await _replay(session, row)
                row.ended_at = func.now()
                await session.flush()  # the open-conversation index allows one at a time
            row = ConversationRow(user_id=user_id, context=conversation_context(today), day=today)
            session.add(row)
            await session.flush()
            return StoredConversation(row.id, Conversation(row.context), saved=0)

    async def close(self, user_id: int) -> None:
        """End the user's open conversation, if any: the next turn starts a new one."""
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(ConversationRow)
                .where(ConversationRow.user_id == user_id, ConversationRow.ended_at.is_(None))
                .values(ended_at=func.now())
            )

    async def save(self, stored: StoredConversation, turn: Turn) -> None:
        """Append the messages `turn` added to `stored.conversation`, each with the turn's trace,
        and its iteration count on the last reply."""
        messages = stored.conversation.messages
        new = messages[stored.saved :]
        last_reply = max(i for i, message in enumerate(new) if message["role"] == "assistant")
        async with self._sessions() as session, session.begin():
            if stored.saved and messages[stored.saved - 1]["role"] == "user":
                # The last turn was cut off after its tool results, and this turn's text joined
                # that message (`Conversation.with_user_text`): it grew by one block at its end.
                await session.execute(
                    update(Message)
                    .where(
                        Message.conversation_id == stored.id,
                        Message.position == stored.saved - 1,
                    )
                    .values(content=messages[stored.saved - 1]["content"])
                )
            session.add_all(
                Message(
                    conversation_id=stored.id,
                    position=stored.saved + i,
                    role=message["role"],
                    content=message["content"],
                    trace_id=turn.trace_id,
                    iteration_count=turn.iteration_count if i == last_reply else None,
                )
                for i, message in enumerate(new)
            )
            await session.execute(
                update(ConversationRow)
                .where(ConversationRow.id == stored.id)
                .values(last_turn_at=func.now())
            )
        stored.saved = len(messages)

    async def rate(self, conversation_id: int, rating: Rating) -> bool:
        """Record `rating` on the conversation's last reply, replacing any earlier verdict on it.
        False when there is no reply yet."""
        async with self._sessions() as session, session.begin():
            reply = await session.scalar(
                select(Message.id)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.iteration_count.is_not(None),
                )
                .order_by(Message.position.desc())
                .limit(1)
            )
            if reply is None:
                return False
            stmt = insert(Feedback).values(message_id=reply, rating=rating)
            await session.execute(
                stmt.on_conflict_do_update(
                    index_elements=["message_id"],
                    set_={"rating": stmt.excluded.rating, "created_at": func.now()},
                )
            )
            return True


async def _replay(session: AsyncSession, row: ConversationRow) -> StoredConversation:
    """`row` as the loop holds a conversation: its context and every message, in order, as
    stored."""
    stored = await session.scalars(
        select(Message).where(Message.conversation_id == row.id).order_by(Message.position)
    )
    messages = [{"role": message.role, "content": message.content} for message in stored]
    return StoredConversation(row.id, Conversation(row.context, messages), saved=len(messages))
