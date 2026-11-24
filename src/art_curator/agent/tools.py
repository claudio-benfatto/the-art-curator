"""The curator's tools: their definitions as the model sees them, the pydantic models every input
is validated against, and `CuratorTools`, which runs one call against the database.

Bedrock's Messages endpoint has no `strict: true` (CLAUDE.md § 5), so the schema is advice to the
model, not a guarantee. An input that fails validation, names no tool, or that the query refuses
(dates the wrong way round, a place it cannot pin down) comes back as an `is_error` result the
model can read and correct, never as an exception out of the loop.

`search_events` and `get_events` are thin wrappers over `queries/events.py`, which does every date
and distance in SQL (CLAUDE.md § 8). Results are compact JSON of the query's own rows: facts,
titles and our summaries, never a page's text (CLAUDE.md § 1). `remember` is the one tool that
writes: a fact about the user it is bound to (`agent/profile.py`).

`lat`/`lon` on `search_events` wait for a client that sends a location (P5): until then the only
point the model could give is one it estimated, which is the distance arithmetic § 8 forbids.

`query` on `search_events` is semantic ranking (PLAN.md § 8, P3), offered only when the tools are
given a `QueryEmbedder`: the deployment set `EMBED_MODEL`, or an eval arm asked for it. Without one
the model is sent the same tools as before, byte for byte.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import date, time
from decimal import Decimal
from typing import Annotated, Any, Literal

from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from art_curator.agent import profile
from art_curator.db.models import PROFILE_TOPICS
from art_curator.llm.client import LlmClient
from art_curator.llm.tools import flat_schema, validation_feedback
from art_curator.queries import events as q

EventId = Annotated[str, Field(pattern=q.EVENT_ID_PATTERN)]


class SearchEventsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date_from: date = Field(description="First day of the visit, YYYY-MM-DD, from the calendar.")
    date_to: date = Field(description="Last day, YYYY-MM-DD; date_from again for a single day.")
    near: str | None = Field(
        default=None,
        min_length=1,
        description="Where the user will be, as they named it: a barri, district, town or venue. "
        "Distances are measured from it and the nearest come first. 'name, within' picks one of "
        "two places of a name ('Canyelles, Garraf').",
    )
    radius_km: float | None = Field(
        default=None, gt=0, le=200, description="Only items within this distance of `near`."
    )
    kinds: list[Literal[q.KINDS]] | None = Field(
        default=None, description="Only these kinds. Leave out for everything."
    )
    free_only: bool = Field(
        default=False, description="Drop what is known to cost money; unknown prices stay."
    )
    limit: int = Field(default=40, ge=1, le=100, description="Most rows to return.")


class RankedSearchEventsInput(SearchEventsInput):
    """`search_events` with semantic ranking: what the curator is sent when there is an embedder."""

    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="What this person is after, in a few English words ('video art about "
        "migration', 'abstract painting'). When more items match than `limit`, the ones closest "
        "to it in meaning are kept rather than the soonest to end; rows still come nearest first.",
    )


class GetEventsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[EventId] = Field(
        min_length=1,
        max_length=q.MAX_HYDRATE,
        description=f"Up to {q.MAX_HYDRATE} ids from search_events: x:12, a:7, g:301.",
    )


class RememberInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: Literal[PROFILE_TOPICS] = Field(description="What the fact is about.")
    fact: str = Field(
        min_length=1,
        max_length=200,
        description="One short sentence in English: 'Likes video art; finds painting dull.'",
    )
    replaces: int | None = Field(
        default=None,
        description="The id of a remembered fact this one corrects or updates; it is dropped.",
    )


class RankingUnavailable(ValueError):
    """The query could not be embedded: the model is told to search without it."""


@dataclass(frozen=True)
class QueryEmbedder:
    """Turns a search's `query` into a vector, with the model `cli embed` embedded items with."""

    llm: LlmClient
    model: str

    async def __call__(self, text: str) -> q.QueryVector:
        try:
            vector = await self.llm.embed(
                purpose="embed", model=self.model, text=text, input_type="query"
            )
        except (BotoCoreError, ClientError) as exc:
            raise RankingUnavailable(
                f"Ranking by query is unavailable ({type(exc).__name__}); search again without "
                "`query`."
            ) from exc
        return q.QueryVector(self.model, vector)


def query_embedder(llm: LlmClient, embed_model: str | None) -> QueryEmbedder | None:
    """The embedder for `EMBED_MODEL`, or None when it is unset: no semantic ranking."""
    return QueryEmbedder(llm, embed_model) if embed_model else None


@dataclass(frozen=True)
class ToolContext:
    """What a tool runs for: the user it is bound to, and the embedder, if there is one."""

    user_id: int
    embedder: QueryEmbedder | None = None


async def _search_events(
    session: AsyncSession, context: ToolContext, args: SearchEventsInput
) -> dict[str, Any]:
    arguments = args.model_dump()
    text = arguments.pop("query", None)
    if text is not None and context.embedder is not None:
        arguments["query"] = await context.embedder(text)
    result = await q.search_events(session, **arguments)
    return {**asdict(result), "truncated": result.total > len(result.events)}


async def _get_events(
    session: AsyncSession, _context: ToolContext, args: GetEventsInput
) -> dict[str, Any]:
    found = await q.get_events(session, args.ids)
    returned = {detail.id for detail in found}
    asked = [f"{prefix}:{number}" for prefix, number in map(q.parse_event_id, args.ids)]
    return {
        "events": [asdict(detail) for detail in found],
        "not_found": [i for i in dict.fromkeys(asked) if i not in returned],
    }


async def _remember(
    session: AsyncSession, context: ToolContext, args: RememberInput
) -> dict[str, Any]:
    fact_id = await profile.remember(session, context.user_id, **args.model_dump())
    return {"id": fact_id, "replaced": args.replaces}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input: type[BaseModel]
    run: Callable[[AsyncSession, ToolContext, Any], Awaitable[dict[str, Any]]]
    writes: bool = False

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": flat_schema(self.input),
        }


SEARCH_DESCRIPTION = (
    "Exhibitions, art events and GRAF listings on at some point between two dates, as one "
    "list. Each row: id (x: exhibition, a: art event, g: GRAF), kind, GRAF's own category, "
    "title, venue, start and end (no end: an exhibition that may still be on), ends_in_days "
    "(from date_from), distance_km (with `near`; null for items shown elsewhere or online), "
    "at_venue, is_free (null: unknown, not free), artists, media, themes. No descriptions: "
    "get_events has those. `total` is how many matched; `truncated` means the list was cut. "
    "`near` in the result is the place the name was taken to mean. A name that matches "
    "nothing, or several places far apart, is an error listing the candidates."
)
SEARCH = Tool("search_events", SEARCH_DESCRIPTION, SearchEventsInput, _search_events)
RANKED_SEARCH = Tool("search_events", SEARCH_DESCRIPTION, RankedSearchEventsInput, _search_events)

TOOLS = (
    SEARCH,
    Tool(
        "get_events",
        "Everything held on up to 10 ids from search_events: our English summary (exhibitions "
        "and art events read off venue sites; GRAF listings have none), dates, start time, "
        "price (GRAF only), the item's own page, and the venue's name, address and website. "
        "Ids that name nothing are listed in `not_found`.",
        GetEventsInput,
        _get_events,
    ),
    Tool(
        "remember",
        "Keep one fact about this person for future conversations, under a topic. Returns its "
        "id. With `replaces`, the fact of that id (from the context block or an earlier "
        "remember) is dropped from the profile in favour of this one.",
        RememberInput,
        _remember,
        writes=True,
    ),
)
# With semantic ranking: the same tools, `search_events` taking a `query`.
RANKED_TOOLS = (RANKED_SEARCH, *TOOLS[1:])
# What the model is sent, in a fixed order: tools render first, so they head the cached prefix.
TOOL_DEFINITIONS = [tool.definition() for tool in TOOLS]
RANKED_TOOL_DEFINITIONS = [tool.definition() for tool in RANKED_TOOLS]


@dataclass(frozen=True)
class ToolResult:
    content: str  # JSON on success, a sentence the model can act on when `is_error`
    is_error: bool = False

    def block(self, tool_use_id: str) -> dict[str, Any]:
        """The `tool_result` block that answers the call `tool_use_id`."""
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            block["is_error"] = True
        return block


def to_json(value: Any) -> str:
    """Compact JSON: these results are paid for as input tokens on every later call."""
    return json.dumps(value, default=_json_default, ensure_ascii=False, separators=(",", ":"))


def _json_default(value: Any) -> Any:
    if isinstance(value, date | time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


class CuratorTools:
    """Runs one tool call for one user, each in its own transaction: read-only, except for a tool
    that writes, and that one only ever about this user. With an `embedder`, `search_events`
    takes a `query`; `definitions` is what the model must be sent to match."""

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        user_id: int,
        embedder: QueryEmbedder | None = None,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._context = ToolContext(user_id, embedder)
        tools = TOOLS if embedder is None else RANKED_TOOLS
        self._tools = {tool.name: tool for tool in tools}
        self.definitions = TOOL_DEFINITIONS if embedder is None else RANKED_TOOL_DEFINITIONS

    async def __call__(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"No tool is called {name!r}; use {', '.join(self._tools)}.", True)
        try:
            args = tool.input.model_validate(tool_input)
        except ValidationError as exc:
            return ToolResult(f"Invalid input: {validation_feedback(exc)}", True)
        try:
            # A refusal raised inside rolls the transaction back, so it leaves nothing written.
            async with self._sessionmaker() as session, session.begin():
                if not tool.writes:
                    await session.execute(text("SET TRANSACTION READ ONLY"))
                return ToolResult(to_json(await tool.run(session, self._context, args)))
        except ValueError as exc:  # refusals: UnresolvedPlace, UnknownFact, RankingUnavailable…
            return ToolResult(str(exc), True)
