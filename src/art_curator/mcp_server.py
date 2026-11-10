"""Dev-time MCP server over `queries/events.py`, for interrogating the corpus from Claude Code
while building the P2 gold set (PLAN.md § 8). **Not on the serving path**, and `mcp` is a dev
dependency only.

    uv run python -m art_curator.mcp_server        # stdio; `.mcp.json` registers it

It reads the database `DATABASE_URL` names, and every tool runs in a read-only transaction.
No tool returns page text: what a conversation sees is metadata, facts and our own summaries,
which keeps third-party prose out of transcripts (CLAUDE.md § 1). To read a page, open it in a
browser. `tests/test_no_verbatim.py` checks every tool for it.
"""

from collections.abc import Awaitable, Callable
from datetime import date
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.db.session import get_sessionmaker
from art_curator.queries import events as q

INSTRUCTIONS = """\
Read-only views of The Art Curator's database: venues, discovery seeds, crawled pages,
the exhibitions and events extracted from them, GRAF's events, and model cost.
search_events and get_events are the curator's own tools: all three sources as one list, then
the details of a shortlist by its prefixed ids (x: exhibition, a: art event, g: GRAF).
Tools return ids, facts and metadata, never a page's text: to read a page, open its URL in a
browser. summary_en is written by our extraction model, not copied from the venue.
Dates are days in Europe/Madrid. Filter by venue_id from the venues tool."""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

Limit = Annotated[int, Field(ge=1, le=500, description="Most rows to return.")]
VenueId = Annotated[int | None, Field(description="Only this venue (an id from `venues`).")]
NotOverOn = Annotated[
    date | None, Field(description="Only items not over on this day (YYYY-MM-DD).")
]
Kind = Literal[q.KINDS]
EventId = Annotated[str, Field(pattern=q.EVENT_ID_PATTERN)]

# Constructing the server calls `logging.basicConfig` at this level, on the root logger. At INFO,
# merely importing this module would route every library's INFO lines (httpx's requests among
# them) to stderr for the whole process, CLI output included. WARNING is Python's default.
mcp = MCPServer("art-curator", instructions=INSTRUCTIONS, log_level="WARNING")


async def _read[T](query: Callable[..., Awaitable[T]], **kwargs: Any) -> T:
    """Run one query in its own read-only transaction, which closes (rolls back) on exit."""
    session: AsyncSession
    async with get_sessionmaker()() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))
        return await query(session, **kwargs)


@mcp.tool(annotations=READ_ONLY)
async def venues(
    pilot_only: Annotated[bool, Field(description="Only the pilot venues.")] = True,
    name: Annotated[str | None, Field(description="Part of the name, any case.")] = None,
    limit: Limit = 100,
) -> list[q.VenueRow]:
    """Venues with their website, crawl flags, coordinates and how many pages, exhibitions and
    events are stored for each."""
    return await _read(q.venues, pilot_only=pilot_only, name=name, limit=limit)


@mcp.tool(annotations=READ_ONLY)
async def seeds(venue_id: VenueId = None, limit: Limit = 100) -> list[q.SeedRow]:
    """What discovery concluded about each candidate listing page: page type, confidence,
    dated-item count, language and accepted/rejected/ambiguous. Human pins and rejections are
    in ingest/seeds.yaml, not here."""
    return await _read(q.seeds, venue_id=venue_id, limit=limit)


@mcp.tool(annotations=READ_ONLY)
async def pages(venue_id: VenueId = None, limit: Limit = 100) -> list[q.PageRow]:
    """Crawled pages by metadata: status, content hash, word and link counts, whether the text
    is still held (7-day TTL) and the last extraction's status, version, model and item count.
    Never the text itself."""
    return await _read(q.pages, venue_id=venue_id, limit=limit)


@mcp.tool(annotations=READ_ONLY)
async def exhibitions(
    venue_id: VenueId = None,
    source_url: Annotated[
        str | None, Field(description="Only shows this page wrote last (a `pages` url).")
    ] = None,
    title: Annotated[str | None, Field(description="Part of the title, any case.")] = None,
    not_over_on: NotOverOn = None,
    limit: Limit = 100,
) -> list[q.ExhibitionRow]:
    """Exhibitions extraction stored: title, artists, dates, our summary, themes, media,
    confidence and the page that wrote each last. An open-ended show counts as not over."""
    return await _read(
        q.exhibitions,
        venue_id=venue_id,
        source_url=source_url,
        title=title,
        not_over_on=not_over_on,
        limit=limit,
    )


@mcp.tool(annotations=READ_ONLY)
async def art_events(
    venue_id: VenueId = None,
    source_url: Annotated[
        str | None, Field(description="Only events this page wrote last (a `pages` url).")
    ] = None,
    title: Annotated[str | None, Field(description="Part of the title, any case.")] = None,
    kind: Annotated[
        str | None, Field(description="opening, talk, workshop, guided_tour, …")
    ] = None,
    not_over_on: NotOverOn = None,
    limit: Limit = 100,
) -> list[q.ArtEventRow]:
    """Openings, talks, workshops, tours… extraction stored, with the exhibition each belongs
    to when the page said. An event without an end date is over after its start date."""
    return await _read(
        q.art_events,
        venue_id=venue_id,
        source_url=source_url,
        title=title,
        kind=kind,
        not_over_on=not_over_on,
        limit=limit,
    )


@mcp.tool(annotations=READ_ONLY)
async def graf_events(
    venue_id: VenueId = None, not_over_on: NotOverOn = None, limit: Limit = 100
) -> list[q.GrafEventRow]:
    """GRAF's events, one row per occurrence: title, dates, price, URLs and when a sync last
    saw it. Facts only; GRAF's descriptions are never stored."""
    return await _read(q.graf_events, venue_id=venue_id, not_over_on=not_over_on, limit=limit)


@mcp.tool(annotations=READ_ONLY)
async def search_events(
    date_from: Annotated[date, Field(description="First day of the visit (YYYY-MM-DD).")],
    date_to: Annotated[date, Field(description="Last day; date_from again for a single day.")],
    lat: Annotated[
        float | None, Field(ge=-90, le=90, description="Latitude to measure from.")
    ] = None,
    lon: Annotated[
        float | None, Field(ge=-180, le=180, description="Longitude to measure from.")
    ] = None,
    radius_km: Annotated[
        float | None, Field(gt=0, le=200, description="Only within this distance of lat/lon.")
    ] = None,
    kinds: Annotated[list[Kind] | None, Field(description="Only these kinds.")] = None,
    free_only: Annotated[
        bool, Field(description="Drop what is known to cost money; unknown prices stay.")
    ] = False,
    limit: Annotated[int, Field(ge=1, le=100, description="Most rows to return.")] = 40,
) -> q.SearchResult:
    """Exhibitions, art events and GRAF events on at some point in the dates, as one list:
    id, kind (and GRAF's own category), title, venue, start/end, ends_in_days, distance_km,
    artists, media, themes, and
    `total` matched. A GRAF event extraction also found is listed once, under its x:/a: id."""
    return await _read(
        q.search_events,
        date_from=date_from,
        date_to=date_to,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        kinds=kinds,
        free_only=free_only,
        limit=limit,
    )


@mcp.tool(annotations=READ_ONLY)
async def get_events(
    ids: Annotated[
        list[EventId],
        Field(
            min_length=1,
            max_length=q.MAX_HYDRATE,
            description="Ids from search_events: x:12, a:7, g:301.",
        ),
    ],
) -> list[q.EventDetail]:
    """Everything held on each id: our summary, all facts, price (GRAF only), the item's own
    page and the venue's name, address and website. Unknown ids are left out."""
    return await _read(q.get_events, ids=ids)


@mcp.tool(annotations=READ_ONLY)
async def llm_costs(
    since: Annotated[date | None, Field(description="Only calls on or after this day.")] = None,
    purpose: Annotated[
        str | None, Field(description="chat, discover, extract, embed, judge or smoke.")
    ] = None,
) -> list[q.CostRow]:
    """Recorded model calls by purpose and model: calls, errors, tokens (cache included) and
    cost in USD, costliest first."""
    return await _read(q.llm_costs, since=since, purpose=purpose)


if __name__ == "__main__":
    mcp.run()
