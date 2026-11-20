"""Conversations kept in Postgres and replayed from it, so any process can serve the next turn.

A turn is: `current` loads the user's open conversation (or opens one), `Curator.turn` answers on
it, and `save` appends what the turn added. The history comes back exactly as it was stored, so
turn n+1's request starts with turn n's, byte for byte, and the cache holds (CLAUDE.md § 6).

A conversation ends after `IDLE` without a turn, or once its day is over in SITE_TIMEZONE: its
context block holds a calendar, and a calendar cannot outlive its day (PLAN.md § 8, Dates). The
next turn then opens a new one, and what carries over is the profile: the facts `remember` kept,
rendered into the new context block (`agent/profile.py`).

Two turns on one conversation at once are not serialised: the second to save collides with the
first on `(conversation_id, position)` and fails, leaving the first's.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from art_curator.agent import profile
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

    async def find_user(self, client: str, external_id: str) -> int | None:
        """The id of the user `client` knows as `external_id`, or None if never seen: for requests
        that should not create a user, such as forgetting one."""
        async with self._sessions() as session:
            return await session.scalar(
                select(User.id).where(User.client == client, User.external_id == external_id)
            )

    async def current(self, user_id: int, today: date) -> StoredConversation:
        """The user's open conversation, replayed from its messages; or, if it has expired or
        there is none, a new one whose calendar starts at `today`, with the profile as it is
        now."""
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
            context = conversation_context(today, await profile.snapshot(session, user_id))
            row = ConversationRow(user_id=user_id, context=context, day=today)
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

    async def remembered(self, user_id: int) -> list[profile.Fact]:
        """The user's live profile, as the next conversation will open with it."""
        async with self._sessions() as session:
            return await profile.snapshot(session, user_id)

    async def forget(self, user_id: int) -> None:
        """Delete everything kept about the user: profile, conversations, messages, feedback."""
        async with self._sessions() as session, session.begin():
            await profile.forget(session, user_id)

    async def save(self, stored: StoredConversation, turn: Turn) -> int:
        """Append the messages `turn` added to `stored.conversation`, each with the turn's trace,
        and its iteration count on the last reply. Returns that reply's message id: what feedback
        names it by."""
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
            rows = [
                Message(
                    conversation_id=stored.id,
                    position=stored.saved + i,
                    role=message["role"],
                    content=message["content"],
                    trace_id=turn.trace_id,
                    iteration_count=turn.iteration_count if i == last_reply else None,
                )
                for i, message in enumerate(new)
            ]
            session.add_all(rows)
            await session.flush()  # assigns the ids
            reply_id = rows[last_reply].id
            await session.execute(
                update(ConversationRow)
                .where(ConversationRow.id == stored.id)
                .values(last_turn_at=func.now())
            )
        stored.saved = len(messages)
        return reply_id

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
            await _rate(session, reply, rating)
            return True

    async def rate_reply(self, user_id: int, message_id: int, rating: Rating) -> bool:
        """Record `rating` on the reply `save` returned as `message_id`, replacing any earlier
        verdict on it. False when that is not a reply in one of the user's conversations."""
        async with self._sessions() as session, session.begin():
            reply = await session.scalar(
                select(Message.id)
                .join(ConversationRow, ConversationRow.id == Message.conversation_id)
                .where(
                    Message.id == message_id,
                    Message.iteration_count.is_not(None),
                    ConversationRow.user_id == user_id,
                )
            )
            if reply is None:
                return False
            await _rate(session, reply, rating)
            return True


async def _rate(session: AsyncSession, message_id: int, rating: Rating) -> None:
    """Upsert the verdict on one reply: a second replaces the first."""
    stmt = insert(Feedback).values(message_id=message_id, rating=rating)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=["message_id"],
            set_={"rating": stmt.excluded.rating, "created_at": func.now()},
        )
    )


async def _replay(session: AsyncSession, row: ConversationRow) -> StoredConversation:
    """`row` as the loop holds a conversation: its context and every message, in order, as
    stored."""
    stored = await session.scalars(
        select(Message).where(Message.conversation_id == row.id).order_by(Message.position)
    )
    messages = [{"role": message.role, "content": message.content} for message in stored]
    return StoredConversation(row.id, Conversation(row.context, messages), saved=len(messages))
