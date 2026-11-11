"""`queries/events.py` and the dev MCP server over it: what each query returns and filters on,
and that the server only reads.

Every row is invented (`tests/corpus.py`). That no tool returns page text is checked in
`tests/test_no_verbatim.py`, with the other copyright rules.
"""

import asyncio
import logging
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
from art_curator.ingest.graf import CATEGORIES
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
        4,
        3,
    )
    assert (pilot.latitude, pilot.longitude) == pytest.approx((41.38, 2.17))
    assert [v.name for v in _query(corpus_db, q.venues, pilot_only=False)] == [
        "Annex Exemple",
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

    assert [row.title for row in rows] == ["Col·lecció", "Arxiu", "Prova U", "Stand a la fira"]
    assert rows[2].summary_en == "Drawings on paper, made for this show."
    assert rows[2].artists == ["Artista Inventada"]
    assert rows[2].venue_name == "Galeria Exemple"


def test_an_open_ended_exhibition_is_not_over(corpus_db):
    rows = _query(corpus_db, q.exhibitions, not_over_on=ON)

    assert [row.title for row in rows] == ["Col·lecció", "Prova U", "Stand a la fira"]


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

    # Every occurrence ever stored, superseded and hidden ones included: `search_events` filters.
    assert [row.title for row in rows] == [
        "Prova U", "Fira", "Assaig U", "Retirat", "Arxiu", "Nit Oberta", "Nit Oberta",
    ]  # fmt: skip
    assert (rows[1].venue_name, rows[1].price_min) == (None, Decimal("12.50"))


def test_a_graf_occurrence_ends_on_the_day_it_ends_in_barcelona(corpus_db):
    """It ends at 23:30 UTC on 31 October, which is 00:30 on 1 November there (CLAUDE.md § 8)."""
    still_on = _query(corpus_db, q.graf_events, venue_id=1, not_over_on=date(2026, 11, 1))
    over = _query(corpus_db, q.graf_events, venue_id=1, not_over_on=date(2026, 11, 2))

    assert "Prova U" in [row.title for row in still_on]
    assert "Prova U" not in [row.title for row in over]


# --- search and hydrate --------------------------------------------------------------------------

# Thursday 5 to Sunday 8 November 2026, and the point of Galeria Exemple.
WEEKEND = {"date_from": date(2026, 11, 5), "date_to": date(2026, 11, 8)}
AT_GALLERY = {"lat": 41.38, "lon": 2.17}


def _search(url: URL, **kwargs: Any) -> q.SearchResult:
    return _query(url, q.search_events, **kwargs)


def _ids(result: q.SearchResult) -> list[str]:
    return [row.id for row in result.events]


def test_search_lists_all_three_sources_soonest_to_end_first(corpus_db):
    result = _search(corpus_db, **WEEKEND)

    assert [(row.id, row.title, row.ends_in_days) for row in result.events] == [
        ("a:2", "Visita", 0),  # ends on date_from
        ("g:4", "Nit Oberta", 2),
        ("x:4", "Stand a la fira", 3),
        ("g:7", "Arxiu", 15),  # extraction's Arxiu closed in September: not the same show
        ("x:1", "Prova U", 25),
        ("x:3", "Col·lecció", None),  # open-ended, so last
    ]
    assert result.total == 6
    assert [row.kind for row in result.events] == [
        "guided_tour", "opening", "exhibition", "exhibition", "exhibition", "exhibition",
    ]  # fmt: skip


def test_search_rows_carry_facts_and_no_summary(corpus_db):
    rows = {row.id: row for row in _search(corpus_db, **WEEKEND).events}

    show = rows["x:1"]
    assert (show.venue, show.start, show.end, show.artists) == (
        "Galeria Exemple",
        date(2026, 9, 12),
        date(2026, 11, 30),
        ["Artista Inventada"],
    )
    assert (show.distance_km, show.is_free, show.at_venue, show.category) == (
        None,
        None,
        True,
        None,
    )
    assert not hasattr(show, "summary_en")
    opening = rows["g:4"]
    assert (opening.venue, opening.is_free, opening.artists) == ("Annex Exemple", True, [])
    assert (opening.kind, opening.category) == ("opening", "inauguracio")


def test_an_item_matches_when_its_run_overlaps_the_dates(corpus_db):
    # Visita is on the 5th only; the stand opens on the 6th.
    assert "a:2" not in _ids(
        _search(corpus_db, date_from=date(2026, 11, 6), date_to=date(2026, 11, 8))
    )
    assert "x:4" not in _ids(
        _search(corpus_db, date_from=date(2026, 11, 1), date_to=date(2026, 11, 5))
    )


def test_an_open_ended_exhibition_is_on_whatever_the_dates(corpus_db):
    result = _search(corpus_db, date_from=date(2027, 3, 1), date_to=date(2027, 3, 2))

    assert (_ids(result), result.total) == (["x:3"], 1)


def test_an_event_with_no_date_is_never_searched(corpus_db):
    """Taller (a:3) has no date at all, so no window can hold it."""
    year = _search(corpus_db, date_from=date(2026, 1, 1), date_to=date(2026, 12, 31), limit=100)

    assert "a:3" not in _ids(year)
    assert "a:1" in _ids(year)


def test_nothing_on_is_an_empty_result(corpus_db):
    result = _search(corpus_db, date_from=date(2025, 1, 1), date_to=date(2025, 1, 2))

    assert (result.events, result.total) == ([], 0)


def test_a_graf_event_extraction_also_found_is_listed_once(corpus_db):
    """Both GRAF rows run through 20–25 October: Prova U at the same venue, and Assaig U, whose
    English title `PROVA Ú.` is Prova U, at the organisation's other space."""
    result = _search(corpus_db, date_from=date(2026, 10, 20), date_to=date(2026, 10, 25))

    assert _ids(result) == ["x:1", "x:3"]
    hidden = [row.title for row in _query(corpus_db, q.graf_events, not_over_on=date(2026, 10, 25))]
    assert {"Prova U", "Assaig U"} <= set(hidden)  # stored, only not searched


def test_search_shows_only_what_the_latest_sync_saw(corpus_db):
    """On the 6th GRAF holds a superseded Nit Oberta occurrence (g:5) and Retirat (g:6), which
    left the window early. Neither is listed."""
    result = _search(corpus_db, date_from=date(2026, 11, 6), date_to=date(2026, 11, 6))

    assert _ids(result) == ["x:4", "g:7", "x:1", "x:3"]


def test_a_radius_keeps_what_is_near_and_nothing_without_a_place(corpus_db):
    near = _search(corpus_db, **WEEKEND, **AT_GALLERY, radius_km=1)
    farther = _search(corpus_db, **WEEKEND, **AT_GALLERY, radius_km=5)

    assert _ids(near) == ["a:2", "g:7", "x:1", "x:3"]
    assert {row.distance_km for row in near.events} == {0.0}
    assert _ids(farther) == ["a:2", "g:7", "x:1", "x:3", "g:4"]
    assert farther.events[-1].distance_km == pytest.approx(3.35, abs=0.1)
    # The stand is at a fair, not the gallery: no distance, so never inside a radius.
    assert "x:4" not in _ids(farther)


def test_a_point_without_a_radius_puts_the_nearest_first(corpus_db):
    result = _search(corpus_db, **WEEKEND, **AT_GALLERY)

    assert _ids(result) == ["a:2", "g:7", "x:1", "x:3", "g:4", "x:4"]
    assert result.events[-1].distance_km is None


def test_near_measures_from_the_place_it_names(corpus_db):
    result = _search(corpus_db, **WEEKEND, near="poblenou", radius_km=1)

    assert (result.near.name, result.near.kind, result.near.within) == (
        "el Poblenou",
        "barri",
        "Sant Martí",
    )
    assert _ids(result) == ["g:4"]  # Annex Exemple's opening
    assert result.events[0].distance_km == pytest.approx(0.2, abs=0.05)


def test_near_a_venue_measures_from_its_point(corpus_db):
    by_name = _search(corpus_db, **WEEKEND, near="GALERIA EXEMPLE", radius_km=1)
    by_point = _search(corpus_db, **WEEKEND, **AT_GALLERY, radius_km=1)

    assert (by_name.near.name, by_name.near.kind) == ("Galeria Exemple", "venue")
    assert by_name.events == by_point.events
    assert by_point.near is None


def test_near_a_name_that_is_nowhere_says_so(corpus_db):
    """Sala 100% Lluny is a venue, but it has no point to measure from."""
    for name in ("Atlantis", "Sala 100% Lluny"):
        with pytest.raises(q.UnresolvedPlace, match="No barri, district") as raised:
            _search(corpus_db, **WEEKEND, near=name)
        assert raised.value.candidates == []


def test_search_by_kind_means_the_same_on_every_source(corpus_db):
    exhibitions = _search(corpus_db, **WEEKEND, kinds=["exhibition"])
    openings = _search(corpus_db, **WEEKEND, kinds=["opening"])
    # Fira has no GRAF category, which makes it `other`.
    other = _search(
        corpus_db, date_from=date(2026, 10, 1), date_to=date(2026, 10, 1), kinds=["other"]
    )

    assert _ids(exhibitions) == ["x:4", "g:7", "x:1", "x:3"]
    assert _ids(openings) == ["g:4"]
    assert [(row.id, row.category) for row in other.events] == [("g:2", None)]


def test_free_only_drops_what_is_known_to_cost_money(corpus_db):
    """GRAF's Arxiu is not free. Extracted items have no price, so they stay."""
    result = _search(corpus_db, **WEEKEND, free_only=True)

    assert _ids(result) == ["a:2", "g:4", "x:4", "x:1", "x:3"]


def test_total_counts_what_a_limit_cut(corpus_db):
    result = _search(corpus_db, **WEEKEND, limit=2)

    assert (_ids(result), result.total) == (["a:2", "g:4"], 6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"date_from": date(2026, 11, 8), "date_to": date(2026, 11, 5)},
        {**WEEKEND, "lat": 41.38},
        {**WEEKEND, "radius_km": 2},
        {**WEEKEND, "near": "Poblenou", **AT_GALLERY},
        {**WEEKEND, "kinds": ["exposicio"]},
    ],
    ids=[
        "dates-backwards",
        "lat-without-lon",
        "radius-without-point",
        "near-and-point",
        "unknown-kind",
    ],
)
def test_search_refuses_what_it_cannot_answer(kwargs):
    """Refused before any SQL, so no database is needed."""
    with pytest.raises(ValueError):
        asyncio.run(q.search_events(None, **kwargs))  # type: ignore[arg-type]


def test_every_graf_category_maps_to_a_known_kind_under_its_own_label():
    assert {category.kind for category in CATEGORIES.values()} <= set(q.KINDS)
    # The labels stay one per id even where kinds merge: that is what `category` is for.
    slugs = [category.slug for category in CATEGORIES.values()]
    assert len(set(slugs)) == len(slugs)


def test_get_events_hydrates_each_source_in_the_order_asked(corpus_db):
    # x:999 names nothing; x:01 is x:1 again.
    opening, show, tour = _query(
        corpus_db, q.get_events, ids=["g:4", "x:1", "a:2", "x:999", "x:01"]
    )

    assert (show.id, show.summary_en, show.url) == (
        "x:1",
        "Drawings on paper, made for this show.",
        DETAIL_URL,
    )
    assert (show.venue, show.address, show.website) == (
        "Galeria Exemple",
        "Carrer Inventat 1",
        "https://galeria-exemple.test/",
    )
    assert (tour.id, tour.end, tour.start_time, tour.exhibition_id) == (
        "a:2",
        date(2026, 11, 5),  # no end date: it lasts its start day
        time(19, 0),
        "x:1",
    )
    assert (opening.id, opening.kind, opening.category, opening.start, opening.start_time) == (
        "g:4",
        "opening",
        "inauguracio",
        date(2026, 11, 7),
        time(19, 0),  # 18:00 UTC, on the venue's clock
    )
    assert (opening.is_free, opening.summary_en, opening.url, opening.source_url) == (
        True,
        None,
        "https://annex-exemple.test/nit",
        "https://graf.test/e/9004",
    )


def test_get_events_returns_a_hidden_graf_event_by_id(corpus_db):
    [twin] = _query(corpus_db, q.get_events, ids=["g:3"])

    assert (twin.title, twin.title_en, twin.venue) == ("Assaig U", "PROVA Ú.", "Annex Exemple")


@pytest.mark.parametrize("ids", [["12"], ["y:1"], ["x:"], [f"x:{n}" for n in range(11)]])
def test_get_events_refuses_what_is_not_an_id_or_too_many(ids):
    with pytest.raises(ValueError):
        asyncio.run(q.get_events(None, ids))  # type: ignore[arg-type]


def test_search_events_over_mcp_returns_the_result_and_its_total(corpus_db):
    [result] = call_tools(
        corpus_db,
        [("search_events", {"date_from": "2026-11-05", "date_to": "2026-11-08", "limit": 2})],
    )

    assert not result.is_error
    assert result.structured_content["total"] == 6
    assert [row["id"] for row in result.structured_content["events"]] == ["a:2", "g:4"]


def test_search_events_over_mcp_says_where_near_was_taken_to_be(corpus_db):
    weekend = {"date_from": "2026-11-05", "date_to": "2026-11-08"}
    found, unknown = call_tools(
        corpus_db,
        [
            ("search_events", {**weekend, "near": "Poblenou", "radius_km": 1}),
            ("search_events", {**weekend, "near": "Atlantis"}),
        ],
    )

    assert found.structured_content["near"]["name"] == "el Poblenou"
    assert unknown.is_error
    assert "Atlantis" in unknown.content[0].text


def test_resolve_place_over_mcp(corpus_db):
    [result] = call_tools(corpus_db, [("resolve_place", {"name": "el poblenou"})])

    assert result.structured_content["place"]["name"] == "el Poblenou"
    assert result.structured_content["candidates"] == []


def test_get_events_over_mcp_refuses_a_bad_id():
    """Refused by the tool's schema, so the database here is never reached."""
    [result] = call_tools(
        make_url("postgresql+asyncpg://nobody@localhost/none"), [("get_events", {"ids": ["y:1"]})]
    )

    assert result.is_error


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
        "venues", "seeds", "pages", "exhibitions", "art_events", "graf_events", "search_events",
        "resolve_place", "get_events", "llm_costs",
    ]  # fmt: skip


def test_a_tool_returns_the_query_rows_as_structured_content(corpus_db):
    [result] = call_tools(corpus_db, [("exhibitions", {"not_over_on": "2026-10-31"})])

    assert not result.is_error
    rows = result.structured_content["result"]
    assert [(row["title"], row["end_date"]) for row in rows] == [
        ("Col·lecció", None),
        ("Prova U", "2026-11-30"),
        ("Stand a la fira", "2026-11-08"),
    ]


def test_importing_the_server_leaves_logging_at_the_default_level():
    """`MCPServer` configures the root logger when it is built, so at INFO an import alone made
    `sync-graf` print httpx's request lines into its report."""
    assert mcp_server.mcp.settings.log_level == "WARNING"
    assert logging.getLogger().getEffectiveLevel() >= logging.WARNING


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
