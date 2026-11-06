"""`queries/events.py` and the dev MCP server over it: what each query returns and filters on,
and that the server only reads.

Every row is invented (`tests/corpus.py`). That no tool returns page text is checked in
`tests/test_no_verbatim.py`, with the other copyright rules.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date, time
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from art_curator import mcp_server
from art_curator.queries import events as q
from tests.corpus import (
    CHANGED_URL,
    DETAIL_URL,
    LISTING_TEXT,
    LISTING_URL,
    THIN_URL,
    call_tools,
    list_tools,
)

ON = date(2026, 10, 31)


def _query(url: URL, query: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
    async def main() -> Any:
        engine = create_async_engine(url)
        try:
            async with AsyncSession(engine) as session:
                return await query(session, **kwargs)
        finally:
            await engine.dispose()

    return asyncio.run(main())


# --- venues and seeds ----------------------------------------------------------------------------


def test_venues_are_the_pilots_unless_asked(corpus_db):
    [pilot] = _query(corpus_db, q.venues)

    assert (pilot.name, pilot.pages, pilot.exhibitions, pilot.events) == (
        "Galeria Exemple",
        4,
        3,
        3,
    )
    assert (pilot.latitude, pilot.longitude) == pytest.approx((41.38, 2.17))
    assert [v.name for v in _query(corpus_db, q.venues, pilot_only=False)] == [
        "Galeria Exemple",
        "Sala 100% Lluny",
    ]


def test_a_venue_without_a_point_has_no_coordinates(corpus_db):
    [far] = _query(corpus_db, q.venues, pilot_only=False, name="lluny")

    assert (far.latitude, far.longitude, far.pages) == (None, None, 0)


def test_a_venue_name_is_matched_literally(corpus_db):
    """`%` is a character in a name, not a wildcard."""
    assert [v.name for v in _query(corpus_db, q.venues, pilot_only=False, name="%")] == [
        "Sala 100% Lluny"
    ]


def test_seeds_say_what_discovery_concluded(corpus_db):
    [seed] = _query(corpus_db, q.seeds, venue_id=1)

    assert (seed.url, seed.page_type, seed.status, seed.dated_items) == (
        LISTING_URL,
        "current_listing",
        "accepted",
        4,
    )
    assert _query(corpus_db, q.seeds, venue_id=2) == []


# --- pages ---------------------------------------------------------------------------------------


def test_pages_are_described_by_their_metadata(corpus_db):
    rows = {page.url: page for page in _query(corpus_db, q.pages, venue_id=1)}
    listing, detail, changed, thin = (
        rows[LISTING_URL],
        rows[DETAIL_URL],
        rows[CHANGED_URL],
        rows[THIN_URL],
    )

    assert (listing.text_held, listing.words, listing.link_count) == (
        True,
        len(LISTING_TEXT.split()),
        1,
    )
    assert (listing.extract_status, listing.extracted_items, listing.text_extracted) == (
        "ok",
        2,
        True,
    )
    # Purged: the hash and the extraction markers outlive the text.
    assert (detail.text_held, detail.words, detail.link_count) == (False, None, None)
    assert (detail.content_hash, detail.text_extracted) == ("b" * 64, True)
    # The text moved after it was extracted: the next `extract` sends it to the model.
    assert changed.text_extracted is False
    assert (thin.status, thin.content_hash, thin.extract_status, thin.text_extracted) == (
        "thin",
        None,
        None,
        False,
    )


def test_listings_come_before_detail_pages(corpus_db):
    kinds = [page.kind for page in _query(corpus_db, q.pages)]

    assert kinds == ["listing", "listing", "listing", "detail"]


# --- exhibitions and events ----------------------------------------------------------------------


def test_exhibitions_come_earliest_first_with_our_summary(corpus_db):
    rows = _query(corpus_db, q.exhibitions, venue_id=1)

    assert [row.title for row in rows] == ["Col·lecció", "Arxiu", "Prova U"]
    assert rows[2].summary_en == "Drawings on paper, made for this show."
    assert rows[2].artists == ["Artista Inventada"]
    assert rows[2].venue_name == "Galeria Exemple"


def test_an_open_ended_exhibition_is_not_over(corpus_db):
    rows = _query(corpus_db, q.exhibitions, not_over_on=ON)

    assert [row.title for row in rows] == ["Col·lecció", "Prova U"]


def test_exhibitions_by_the_page_that_wrote_them_and_by_title(corpus_db):
    by_page = _query(corpus_db, q.exhibitions, source_url=LISTING_URL)
    by_title = _query(corpus_db, q.exhibitions, title="PROVA")

    assert [row.title for row in by_page] == ["Col·lecció", "Arxiu"]
    assert [row.title for row in by_title] == ["Prova U"]


def test_an_event_without_an_end_is_over_after_its_start(corpus_db):
    rows = _query(corpus_db, q.art_events, not_over_on=ON)

    # Xerrada (10 September) is over; Taller has no date at all and is kept.
    assert [(row.title, row.start_time, row.exhibition_id) for row in rows] == [
        ("Visita", time(19, 0), 1),
        ("Taller", None, None),
    ]


def test_events_by_kind(corpus_db):
    assert [row.title for row in _query(corpus_db, q.art_events, kind="talk")] == ["Xerrada"]


def test_a_limit_caps_the_rows(corpus_db):
    assert len(_query(corpus_db, q.exhibitions, limit=1)) == 1


# --- GRAF ----------------------------------------------------------------------------------------


def test_graf_events_are_one_row_per_occurrence(corpus_db):
    rows = _query(corpus_db, q.graf_events)

    assert [(row.title, row.venue_name, row.price_min) for row in rows] == [
        ("Prova U", "Galeria Exemple", None),
        ("Fira", None, Decimal("12.50")),
    ]


def test_a_graf_occurrence_ends_on_the_day_it_ends_in_barcelona(corpus_db):
    """It ends at 23:30 UTC on 31 October, which is 00:30 on 1 November there (CLAUDE.md § 8)."""
    still_on = _query(corpus_db, q.graf_events, venue_id=1, not_over_on=date(2026, 11, 1))
    over = _query(corpus_db, q.graf_events, venue_id=1, not_over_on=date(2026, 11, 2))

    assert [row.title for row in still_on] == ["Prova U"]
    assert over == []


# --- cost ----------------------------------------------------------------------------------------


def test_costs_are_added_up_by_purpose_and_model(corpus_db):
    extract, discover = _query(corpus_db, q.llm_costs)

    assert (extract.purpose, extract.calls, extract.errors) == ("extract", 2, 1)
    assert (extract.input_tokens, extract.output_tokens) == (9100, 3000)
    assert extract.cost_usd == Decimal("0.024100")
    assert (discover.purpose, discover.calls, discover.cost_usd) == (
        "discover",
        1,
        Decimal("0.006"),
    )


def test_costs_since_a_day_and_for_a_purpose(corpus_db):
    assert [row.purpose for row in _query(corpus_db, q.llm_costs, since=date(2026, 10, 8))] == [
        "extract"
    ]
    assert [row.purpose for row in _query(corpus_db, q.llm_costs, purpose="discover")] == [
        "discover"
    ]


# --- the MCP server ------------------------------------------------------------------------------


def test_the_server_has_a_tool_per_query():
    assert [tool.name for tool in list_tools()] == [
        "venues", "seeds", "pages", "exhibitions", "art_events", "graf_events", "llm_costs",
    ]  # fmt: skip


def test_a_tool_returns_the_query_rows_as_structured_content(corpus_db):
    [result] = call_tools(corpus_db, [("exhibitions", {"not_over_on": "2026-10-31"})])

    assert not result.is_error
    rows = result.structured_content["result"]
    assert [(row["title"], row["end_date"]) for row in rows] == [
        ("Col·lecció", None),
        ("Prova U", "2026-11-30"),
    ]


def test_a_tool_refuses_a_limit_over_the_cap():
    """Refused before a connection is made, so the database here is never reached."""
    [result] = call_tools(
        make_url("postgresql+asyncpg://nobody@localhost/none"), [("pages", {"limit": 501})]
    )

    assert result.is_error


def test_the_server_reads_in_a_read_only_transaction(corpus_db):
    async def write(session: AsyncSession) -> None:
        await session.execute(text("UPDATE venues SET name = 'x'"))

    async def main() -> None:
        engine = create_async_engine(corpus_db)
        try:
            with patch.object(mcp_server, "get_sessionmaker", lambda: async_sessionmaker(engine)):
                await mcp_server._read(write)
        finally:
            await engine.dispose()

    with pytest.raises(DBAPIError, match="read-only transaction"):
        asyncio.run(main())
