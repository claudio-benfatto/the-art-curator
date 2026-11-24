"""Embed what extraction found, for `search_events(query=...)` (PLAN.md § 8, P3, semantic ranking).

In the order `cli embed` runs it:

1. `load_items()`: every exhibition and art event that has a summary, with the text it would be
   embedded from and the markers its stored vector carries.
2. `plan()`: the items whose text or model moved since they were last embedded, and the rest.
3. `embed_items()`: model only, no database. One call an item: Titan takes one text a call,
   and a few hundred items do not need Cohere's batches.
4. `write_embeddings()`: in the caller's transaction, and **never commits**.

Rules the code encodes:

- **Only our own text is embedded** (CLAUDE.md § 1): `summary_en`, which extraction wrote and
  the verbatim guard checked, plus themes and media. Never the title, never a page's text, never
  anything GRAF wrote. An item without a summary has nothing of ours to embed, so it has no vector
  and a search with a query ranks it last; GRAF events are all such items.
- **An unchanged item costs nothing.** An item is embedded only when the hash of its text, or the
  model, differs from what its stored vector was made with. Extraction keeps the longer summary,
  so a show's text moves when its own page is read after its listing, and is embedded again.
- **A call the endpoint refuses stops the run**, as in `extract`: what was embedded before it is
  still written, since it is paid for, and the items after it wait for the next run.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.db.models import ArtEvent, Exhibition, Venue
from art_curator.llm.client import LlmClient

EXHIBITIONS = Exhibition.__table__
EVENTS = ArtEvent.__table__
VENUES = Venue.__table__

# The prefix `search_events` gives each table's ids, and the table it names.
TABLES = {"x": EXHIBITIONS, "a": EVENTS}


def document(summary_en: str, themes: Sequence[str], media: Sequence[str]) -> str:
    """The text an item is embedded from: our summary, then its themes and media as tags."""
    lines = [summary_en.strip()]
    if themes:
        lines.append("Themes: " + ", ".join(themes) + ".")
    if media:
        lines.append("Media: " + ", ".join(m.replace("_", " ") for m in media) + ".")
    return "\n".join(lines)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Item:
    """One exhibition or art event with a summary."""

    id: str  # x:12 or a:7, as search_events names it
    venue: str
    title: str  # to report and label it by; never embedded
    document: str  # what is embedded (`document()`)
    embed_model: str | None  # what made its stored vector; None when it has none
    embedded_hash: str | None  # the hash of the text it was made from

    @property
    def hash(self) -> str:
        return text_hash(self.document)

    def embedded_with(self, model: str) -> bool:
        """Its stored vector is of this text, by this model."""
        return (self.embed_model, self.embedded_hash) == (model, self.hash)


async def load_items(session: AsyncSession) -> list[Item]:
    """Every exhibition and art event that has a summary, venue by venue."""
    items = []
    for prefix, table in TABLES.items():
        rows = await session.execute(
            select(
                table.c.id,
                VENUES.c.name.label("venue"),
                table.c.title,
                table.c.summary_en,
                table.c.themes,
                table.c.media,
                table.c.embed_model,
                table.c.embedded_hash,
            )
            .join(VENUES, VENUES.c.id == table.c.venue_id)
            .where(table.c.summary_en.is_not(None))
            .order_by(VENUES.c.name, table.c.id)
        )
        items += [
            Item(
                id=f"{prefix}:{row.id}",
                venue=row.venue,
                title=row.title,
                document=document(row.summary_en, row.themes, row.media),
                embed_model=row.embed_model,
                embedded_hash=row.embedded_hash,
            )
            for row in rows
        ]
    return items


@dataclass(frozen=True)
class Plan:
    to_embed: tuple[Item, ...] = ()
    unchanged: tuple[Item, ...] = ()


def plan(items: Sequence[Item], model: str) -> Plan:
    return Plan(
        tuple(i for i in items if not i.embedded_with(model)),
        tuple(i for i in items if i.embedded_with(model)),
    )


@dataclass(frozen=True)
class Embedded:
    item: Item
    vector: list[float] = field(repr=False)


@dataclass(frozen=True)
class EmbedRun:
    embedded: list[Embedded]
    stopped: str = ""  # what the endpoint said when it refused a call; the items after it wait


async def embed_items(llm: LlmClient, model: str, items: Sequence[Item]) -> EmbedRun:
    """One vector per item, as a document (`LlmClient.embed`'s `input_type`)."""
    embedded = []
    for item in items:
        try:
            vector = await llm.embed(
                purpose="embed", model=model, text=item.document, input_type="document"
            )
        except (BotoCoreError, ClientError) as exc:
            return EmbedRun(embedded, stopped=f"{type(exc).__name__}: {exc}")
        embedded.append(Embedded(item, vector))
    return EmbedRun(embedded)


async def write_embeddings(session: AsyncSession, model: str, embedded: Sequence[Embedded]) -> None:
    """Store each vector with its markers. Runs in the caller's transaction and never commits."""
    for done in embedded:
        prefix, number = done.item.id.split(":")
        table: Any = TABLES[prefix]
        await session.execute(
            update(table)
            .where(table.c.id == int(number))
            .values(embedding=done.vector, embed_model=model, embedded_hash=done.item.hash)
        )


def report(model: str, planned: Plan, run: EmbedRun) -> list[str]:
    """What `embed` prints."""
    lines: list[str] = []

    def block(label: str, *parts: str) -> None:
        lines.append(f"{label:<11} " + "   ".join(part for part in parts if part))

    left = len(planned.to_embed) - len(run.embedded)
    block("model", model)
    block(
        "items",
        f"{len(planned.to_embed) + len(planned.unchanged)} with a summary",
        f"{len(planned.to_embed)} to embed",
        f"{len(planned.unchanged)} unchanged",
    )
    block(
        "embedded",
        str(len(run.embedded)),
        f"{left} left for the next run" if left else "",
    )
    if run.stopped:
        block("stopped", f"the endpoint refused a call: {run.stopped}")
    return lines
