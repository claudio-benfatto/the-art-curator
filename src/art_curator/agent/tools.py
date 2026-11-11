"""The curator's tools: their definitions as the model sees them, the pydantic models every input
is validated against, and `CuratorTools`, which runs one call against the database.

Bedrock's Messages endpoint has no `strict: true` (CLAUDE.md § 5), so the schema is advice to the
model, not a guarantee. An input that fails validation, names no tool, or that the query refuses
(dates the wrong way round, a place it cannot pin down) comes back as an `is_error` result the
model can read and correct, never as an exception out of the loop.

Each tool is a thin wrapper over `queries/events.py`, which does every date and distance in SQL
(CLAUDE.md § 8). Results are compact JSON of the query's own rows: facts, titles and our
summaries, never a page's text (CLAUDE.md § 1).

`lat`/`lon` on `search_events` wait for a client that sends a location (P5): until then the only
point the model could give is one it estimated, which is the distance arithmetic § 8 forbids.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import date, time
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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


class GetEventsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[EventId] = Field(
        min_length=1,
        max_length=q.MAX_HYDRATE,
        description=f"Up to {q.MAX_HYDRATE} ids from search_events: x:12, a:7, g:301.",
    )


async def _search_events(session: AsyncSession, args: SearchEventsInput) -> dict[str, Any]:
    result = await q.search_events(session, **args.model_dump())
    return {**asdict(result), "truncated": result.total > len(result.events)}


async def _get_events(session: AsyncSession, args: GetEventsInput) -> dict[str, Any]:
    found = await q.get_events(session, args.ids)
    returned = {detail.id for detail in found}
    asked = [f"{prefix}:{number}" for prefix, number in map(q.parse_event_id, args.ids)]
    return {
        "events": [asdict(detail) for detail in found],
        "not_found": [i for i in dict.fromkeys(asked) if i not in returned],
    }


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input: type[BaseModel]
    run: Callable[[AsyncSession, Any], Awaitable[dict[str, Any]]]

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": flat_schema(self.input),
        }


TOOLS = (
    Tool(
        "search_events",
        "Exhibitions, art events and GRAF listings on at some point between two dates, as one "
        "list. Each row: id (x: exhibition, a: art event, g: GRAF), kind, GRAF's own category, "
        "title, venue, start and end (no end: an exhibition that may still be on), ends_in_days "
        "(from date_from), distance_km (with `near`; null for items shown elsewhere or online), "
        "at_venue, is_free (null: unknown, not free), artists, media, themes. No descriptions: "
        "get_events has those. `total` is how many matched; `truncated` means the list was cut. "
        "`near` in the result is the place the name was taken to mean. A name that matches "
        "nothing, or several places far apart, is an error listing the candidates.",
        SearchEventsInput,
        _search_events,
    ),
    Tool(
        "get_events",
        "Everything held on up to 10 ids from search_events: our English summary (exhibitions "
        "and art events read off venue sites; GRAF listings have none), dates, start time, "
        "price (GRAF only), the item's own page, and the venue's name, address and website. "
        "Ids that name nothing are listed in `not_found`.",
        GetEventsInput,
        _get_events,
    ),
)
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
# What the model is sent, in a fixed order: tools render first, so they head the cached prefix.
TOOL_DEFINITIONS = [tool.definition() for tool in TOOLS]


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
    """Runs one tool call, each in its own read-only transaction. The model writes nothing."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def __call__(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            return ToolResult(f"No tool is called {name!r}; use {', '.join(TOOLS_BY_NAME)}.", True)
        try:
            args = tool.input.model_validate(tool_input)
        except ValidationError as exc:
            return ToolResult(f"Invalid input: {validation_feedback(exc)}", True)
        try:
            async with self._sessionmaker() as session:
                await session.execute(text("SET TRANSACTION READ ONLY"))
                return ToolResult(to_json(await tool.run(session, args)))
        except ValueError as exc:  # the query's own refusals, UnresolvedPlace among them
            return ToolResult(str(exc), True)
