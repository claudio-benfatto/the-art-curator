"""What the curator remembers about a user, from one conversation to the next (PLAN.md § 8,
Profile).

The model learns taste from what the user says and writes it down with the `remember` tool: one
short fact under a topic from PROFILE_TOPICS. Facts are append-only. A correction is a new fact
that names the one it replaces, and the old one is marked `superseded_by` the new: it is kept, but
no longer part of the profile.

The profile is read once, when a conversation opens, and rendered with its fact ids into the
context block (`prompt.conversation_context`), so the model can name a fact it corrects. A fact
remembered mid-conversation is seen through its tool result and joins the context only of the
next conversation: re-rendering would edit the start of the history and break the cache from
there on (CLAUDE.md § 6).

`forget` deletes the profile and every conversation it was learned from.
"""

from dataclasses import dataclass

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.db.models import PROFILE_TOPICS, ProfileFact
from art_curator.db.models import Conversation as ConversationRow


@dataclass(frozen=True)
class Fact:
    id: int
    topic: str  # one of PROFILE_TOPICS
    fact: str


class UnknownFact(ValueError):
    """`replaces` names no live fact of this user. A ValueError, so the tool hands it back to the
    model as an error it can read."""


async def snapshot(session: AsyncSession, user_id: int) -> list[Fact]:
    """The user's live profile: every fact nothing has superseded, by topic, oldest first."""
    rows = await session.execute(
        select(ProfileFact.id, ProfileFact.topic, ProfileFact.fact)
        .where(ProfileFact.user_id == user_id, ProfileFact.superseded_by.is_(None))
        .order_by(ProfileFact.id)
    )
    facts = [Fact(*row) for row in rows]
    return sorted(facts, key=lambda fact: PROFILE_TOPICS.index(fact.topic))


async def remember(
    session: AsyncSession, user_id: int, topic: str, fact: str, replaces: int | None = None
) -> int:
    """Add `fact` to the user's profile and return its id. With `replaces`, that fact leaves the
    profile: it must be one of the user's live facts, or UnknownFact is raised and the caller's
    transaction must roll back, since the new fact is already written."""
    new_id = (
        await session.execute(
            insert(ProfileFact)
            .values(user_id=user_id, topic=topic, fact=fact)
            .returning(ProfileFact.id)
        )
    ).scalar_one()
    if replaces is not None:
        # One statement checks and marks, so two corrections of one fact cannot both win.
        replaced = await session.scalar(
            update(ProfileFact)
            .where(
                ProfileFact.id == replaces,
                ProfileFact.user_id == user_id,
                ProfileFact.superseded_by.is_(None),
            )
            .values(superseded_by=new_id)
            .returning(ProfileFact.id)
        )
        if replaced is None:
            raise UnknownFact(await _why_not_live(session, user_id, replaces))
    return new_id


async def _why_not_live(session: AsyncSession, user_id: int, fact_id: int) -> str:
    successor = await session.scalar(
        select(ProfileFact.superseded_by).where(
            ProfileFact.id == fact_id, ProfileFact.user_id == user_id
        )
    )
    if successor is None:  # another user's fact reads the same as none at all
        return f"No remembered fact has id {fact_id}; nothing was remembered."
    return (
        f"Fact {fact_id} was already replaced by fact {successor}; replace {successor} instead. "
        "Nothing was remembered."
    )


async def forget(session: AsyncSession, user_id: int) -> None:
    """Delete the user's profile and conversations, and with them their messages and feedback.
    The `users` row stays: it holds only the client's id for the person."""
    await session.execute(delete(ProfileFact).where(ProfileFact.user_id == user_id))
    await session.execute(delete(ConversationRow).where(ConversationRow.user_id == user_id))
