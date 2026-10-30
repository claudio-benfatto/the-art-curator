"""`ingest/extraction.py` and `cli extract`: which pages go to the model, how what it found is kept,
and the report.

No model is called: `StubLlm` fakes the wire. Every page here is invented; the listing is the
synthetic fixture `tests/fixtures/crawl/listing.html`.
"""

import asyncio
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, get_args

import pytest
from alembic import command
from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from typer.testing import CliRunner

from art_curator import cli
from art_curator.config import get_settings
from art_curator.db import models
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.ingest import crawl, extraction, pages
from art_curator.ingest.extraction import (
    PagePlan,
    PageRef,
    PageResult,
    PageToExtract,
    StoredLink,
    merge,
    title_key,
)
from art_curator.llm import extract
from art_curator.llm.client import db_recorder
from art_curator.llm.extract import EXTRACT_VERSION, Extraction
from tests.db import alembic_config, run_sql
from tests.llm_stub import StubLlm
from tests.site_stub import HOST, LISTING

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
VENUE = "Galeria Exemple"
LISTING_URL = f"{HOST}/ca/exposicions/"
AGENDA_URL = f"{HOST}/ca/agenda"
DETAIL_URL = f"{HOST}/ca/exposicions/prova-u"
LISTING_TEXT = pages.page_text(LISTING)
DETAIL_TEXT = "Prova U. Artista Inventada. " + "Una pàgina inventada per a la prova. " * 12
LINKS = [{"number": 1, "url": DETAIL_URL, "text": "Prova U"}]

OUR_SUMMARY = "Drawings and paintings on paper, made for this show."
LONGER_SUMMARY = (
    "Works on paper made for this show, in two rooms: drawings first, then the paintings that "
    "grew out of them."
)


def show(title: str = "Prova U", **changes: Any) -> extract.Exhibition:
    fields = {
        "title": title,
        "artists": ("Artista Inventada",),
        "start_date": date(2026, 9, 12),
        "end_date": date(2026, 11, 30),
        "at_venue": True,
        "detail_url": None,
        "summary_en": None,
        "themes": (),
        "media": (),
        "confidence": "high",
    }
    return extract.Exhibition(**(fields | changes))


def event(title: str = "Visita guiada", **changes: Any) -> extract.ArtEvent:
    fields = show(title, artists=(), end_date=None).model_dump() | {
        "kind": "guided_tour",
        "start_time": None,
        "exhibition_title": None,
    }
    return extract.ArtEvent(**(fields | changes))


def page(url: str = LISTING_URL, *, listing: bool = True, previous: int | None = None):
    return PageToExtract(
        page_id=1,
        venue_id=1,
        venue_name=VENUE,
        url=url,
        listing=listing,
        content_hash="h",
        fetched_on=date(2026, 10, 10),
        previous_items=previous,
        text=LISTING_TEXT,
    )


# --- pure ----------------------------------------------------------------------------------------


def test_database_enums_are_the_model_output_enums():
    """The CHECKs are spelled out in `db/models.py` and migration 0007. A value the model may
    answer with but the table refuses would fail at write time, after the spend."""
    assert get_args(extract.Confidence) == models.EXTRACT_CONFIDENCES
    assert get_args(extract.Medium) == models.MEDIA
    assert get_args(extract.EventKind) == models.EVENT_KINDS


@pytest.mark.parametrize(
    ("a", "b"),
    [("Prova U", "PROVA U"), ("Prova U", "Prova U."), ("Àngels", "angels"), ("L&B", "l and b")],
)
def test_one_show_whatever_the_spelling(a, b):
    assert title_key(a) == title_key(b)


def test_a_title_of_punctuation_still_has_a_key():
    assert title_key("¿?") == "¿?"


def test_merge_takes_the_newest_value_that_is_not_empty():
    old = {"title": "Prova U", "end_date": date(2026, 11, 30), "artists": ["A"], "detail_url": "u"}
    new = {"title": "PROVA U", "end_date": date(2027, 1, 10), "artists": [], "detail_url": None}

    assert merge(old, new) == {
        "title": "PROVA U",
        "end_date": date(2027, 1, 10),  # an extension on the listing lands
        "artists": ["A"],  # an empty list says nothing
        "detail_url": "u",  # nor does a page without links
    }


@pytest.mark.parametrize(
    ("old", "new", "kept"),
    [
        (LONGER_SUMMARY, OUR_SUMMARY, LONGER_SUMMARY),
        (OUR_SUMMARY, LONGER_SUMMARY, LONGER_SUMMARY),
        (OUR_SUMMARY, None, OUR_SUMMARY),
        (None, OUR_SUMMARY, OUR_SUMMARY),
    ],
)
def test_merge_keeps_the_longer_summary(old, new, kept):
    assert merge({"summary_en": old}, {"summary_en": new}) == {"summary_en": kept}


def test_broken_is_a_listing_that_stopped_yielding():
    empty = Extraction((), ())

    assert PageResult(page(previous=3), empty).broken
    assert not PageResult(page(previous=0), empty).broken  # it never yielded
    assert not PageResult(page(previous=None), empty).broken  # never extracted, or failed
    assert not PageResult(page(previous=3), Extraction((show(),), ())).broken
    assert not PageResult(page(DETAIL_URL, listing=False, previous=3), empty).broken
    assert not PageResult(page(previous=3), error="failed").broken


def test_page_and_link_text_stay_out_of_repr():
    p = PageToExtract(**{**page().__dict__, "links": (StoredLink(1, DETAIL_URL, "Enllaç ocult"),)})

    assert "Fictici" in p.text and "Fictici" not in repr(p)
    assert "Enllaç ocult" not in repr(p)


def test_a_capped_listing_with_nothing_current_is_reported_not_failed():
    """The page is fetched on 2026-10-10: shows that ended before are over."""
    over = show(start_date=date(2025, 1, 10), end_date=date(2025, 3, 1))
    current = show(end_date=date(2026, 11, 30))
    undated = show(start_date=None, end_date=None)

    def result(*shows, capped=True, listing=True):
        return PageResult(page(listing=listing), Extraction(shows, (), capped=capped))

    assert result(over, over).capped_with_nothing_current
    assert not result(over, current).capped_with_nothing_current
    assert not result(over, undated).capped_with_nothing_current  # may be on: not flagged
    assert not result(over, over, capped=False).capped_with_nothing_current
    lines = extraction.report(PagePlan(to_extract=(page(),)), [result(over, over)])
    assert lines[-1] == (
        f"capped      {LISTING_URL}   kept the first 40 in page order and none is on or coming: "
        "if the page runs oldest first, its current shows were cut"
    )


def test_report_lists_each_page_under_its_venue():
    other = PageToExtract(**{**page().__dict__, "venue_name": "Sala Prova", "url": "https://s.t/a"})
    plan = PagePlan(
        to_extract=(page(previous=2), page(AGENDA_URL), other, page(DETAIL_URL, listing=False)),
        unchanged=(PageRef(VENUE, "https://u.t/"),),
        expired=(PageRef(VENUE, "https://e.t/"),),
    )
    results = [
        PageResult(page(previous=2), Extraction((), ())),
        PageResult(page(AGENDA_URL), Extraction((show(),), (event(),), summaries_dropped=1)),
        PageResult(other, error="record_programme: no usable answer after one retry (x)"),
    ]

    assert extraction.report(plan, results) == [
        "pages       6   4 to extract   1 unchanged   1 expired",
        "extracted   3   2 ok   1 failed   1 left by --limit",
        "items       1 exhibitions   1 events   1 summaries dropped by the verbatim guard",
        "site        Galeria Exemple",
        f"ok          {LISTING_URL}   0 exhibitions   0 events",
        f"broken      {LISTING_URL}   no items, after 2 last time: re-run discover, or fix its "
        "pin in seeds.yaml",
        f"ok          {AGENDA_URL}   1 exhibitions   1 events",
        "site        Sala Prova",
        "failed      https://s.t/a   record_programme: no usable answer after one retry (x)",
        "expired     https://e.t/   its text was purged before it was extracted: crawl again",
    ]


def test_a_page_that_fails_does_not_stop_the_rest():
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, {"exhibitions": "none"})
    stub.tool_reply(extract.TOOL, {"exhibitions": "none"})
    stub.tool_reply(extract.TOOL, {"exhibitions": [], "events": []})

    extracted = asyncio.run(
        extraction.extract_pages(stub.client, HAIKU, [page(), page(AGENDA_URL)])
    )

    assert [r.error.startswith("record_programme") for r in extracted.results] == [True, False]
    assert extracted.results[1].extraction == Extraction((), ())
    assert extracted.stopped == ""
    assert len(stub.calls) == 3


def test_a_refused_call_stops_the_run_but_keeps_what_was_extracted():
    """The first page is paid for: losing it to the second page's refusal would pay twice."""
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, {"exhibitions": [], "events": []})
    stub.converse_stub.add_client_error(
        "converse", service_error_code="ThrottlingException", service_message="Too many requests"
    )

    extracted = asyncio.run(
        extraction.extract_pages(
            stub.client, HAIKU, [page(), page(AGENDA_URL), page(DETAIL_URL, listing=False)]
        )
    )

    assert [r.page.url for r in extracted.results] == [LISTING_URL]
    assert extracted.stopped.startswith("ThrottlingException: An error occurred")
    assert [row.error_type for row in stub.calls] == [None, "ThrottlingException"]  # both recorded

    lines = extraction.report(PagePlan(to_extract=(page(), page(AGENDA_URL))), extracted.results,
                              extracted.stopped)  # fmt: skip
    assert "extracted   1   1 ok   1 left for the next run" in lines
    assert lines[-1].startswith("stopped     the endpoint refused a call: ThrottlingException:")


def test_a_listing_gives_facts_and_a_show_page_its_description():
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, _answer())
    described = _show_input(summary_en=OUR_SUMMARY, themes=["paper"], media=["drawing"])
    stub.tool_reply(extract.TOOL, _answer([described]))

    listing, detail = asyncio.run(
        extraction.extract_pages(stub.client, HAIKU, [page(), page(DETAIL_URL, listing=False)])
    ).results

    assert [r.page.listing for r in (listing, detail)] == [True, False]
    assert [s.summary_en for s in listing.extraction.exhibitions] == [None]
    assert [s.summary_en for s in detail.extraction.exhibitions] == [OUR_SUMMARY]
    systems = [request["system"][0]["text"] for request in stub.converse_requests]
    assert systems == [extract.LISTING_SYSTEM, extract.DETAIL_SYSTEM]


# --- real database -------------------------------------------------------------------------------

VENUE_ROWS = f"""
INSERT INTO venues (id, source_venue_id, name, slug, website_url, crawl_enabled, is_pilot)
OVERRIDING SYSTEM VALUE VALUES
  (1, 101, 'Galeria Exemple', 'galeria-exemple', '{HOST}/', true, true),
  (2, 102, 'MACBA', 'macba', 'http://macba.test', true, true)
"""

SHOWS = (
    "SELECT title, start_date, end_date, artists, detail_url, summary_en, source_url, "
    "first_seen_at, last_seen_at FROM exhibitions ORDER BY id"
)
MARKS = (
    "SELECT url, extract_status, extracted_hash = content_hash, extract_version, extract_model, "
    "extracted_items FROM venue_pages ORDER BY id"
)


@pytest.fixture(scope="module")
def extract_db(fresh_database) -> Iterator[URL]:
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@pytest.fixture
def db(extract_db) -> URL:
    run_sql(extract_db, "TRUNCATE venue_pages, venues, llm_calls CASCADE", VENUE_ROWS)
    return extract_db


def _in_session(url: URL, fn: Callable[[AsyncSession], Any], *, commit: bool = True) -> Any:
    async def main() -> Any:
        engine = create_async_engine(url)
        try:
            async with AsyncSession(engine) as session:
                result = await fn(session)
                await (session.commit() if commit else session.rollback())
                return result
        finally:
            await engine.dispose()

    return asyncio.run(main())


def marks(text: str, *, items: int | None = 3, **changes: Any) -> dict[str, Any]:
    """What an extraction of `text` with this version and model left on its row."""
    return {
        "extract_status": "ok",
        "extracted_hash": crawl.text_hash(text),
        "extract_version": EXTRACT_VERSION,
        "extract_model": HAIKU,
        "extracted_items": items,
        "extracted_at": func.now(),
        **changes,
    }


def _add_page(
    url: URL, page_url: str, text: str, *, age: timedelta = timedelta(0), **columns: Any
) -> None:
    row = {
        "venue_id": 1,
        "url": page_url,
        "status": "ok",
        "http_status": 200,
        "content_hash": crawl.text_hash(text),
        "raw_text": text,
        "fetched_at": func.now() - age,
        **columns,
    }
    _in_session(url, lambda s: s.execute(insert(crawl.PAGES).values(row)))


def _plan(url: URL, model: str = HAIKU) -> PagePlan:
    return _in_session(url, lambda s: extraction.load_pages(s, model))


def _result(url: URL, page_url: str, *shows: extract.Exhibition, events=()) -> PageResult:
    [found] = [p for p in _plan(url).to_extract if p.url == page_url]
    return PageResult(found, Extraction(shows, tuple(events)))


def _write(url: URL, results: list[PageResult], *, commit: bool = True) -> extraction.Written:
    return _in_session(url, lambda s: extraction.write_results(s, results, HAIKU), commit=commit)


def _shows(url: URL) -> list[tuple]:
    return run_sql(url, SHOWS)[0]


def test_pages_are_sorted_by_what_this_run_does_with_them(db):
    _add_page(db, DETAIL_URL, DETAIL_TEXT, kind="detail")
    _add_page(db, LISTING_URL, LISTING_TEXT, links=LINKS)
    _add_page(db, AGENDA_URL, "agenda", **marks("agenda"))
    _add_page(db, f"{HOST}/v", "v", **marks("v", extract_version=EXTRACT_VERSION - 1))
    _add_page(db, f"{HOST}/m", "m", **marks("m", extract_model="another-model"))
    _add_page(db, f"{HOST}/h", "h", **marks("h", extracted_hash="0" * 64))
    _add_page(db, f"{HOST}/f", "f", **marks("f", extract_status="failed", items=None))
    _add_page(db, f"{HOST}/purged", "x", raw_text=None)
    _add_page(db, f"{HOST}/thin", "y", status="thin", content_hash=None, raw_text=None)

    plan = _plan(db)

    # Listings first; a version, model or hash that moved sends a page back to the model.
    assert [p.url.removeprefix(HOST) for p in plan.to_extract] == [
        "/ca/exposicions/", "/v", "/m", "/h", "/ca/exposicions/prova-u",
    ]  # fmt: skip
    # A page that failed waits for one of the three to move, like any other.
    assert [p.url.removeprefix(HOST) for p in plan.unchanged] == ["/ca/agenda", "/f"]
    assert plan.expired == (PageRef(VENUE, f"{HOST}/purged"),)
    listing = plan.to_extract[0]
    assert (listing.venue_name, listing.listing, listing.text) == (VENUE, True, LISTING_TEXT)
    assert listing.links == (StoredLink(1, DETAIL_URL, "Prova U"),)
    assert plan.to_extract[-1].listing is False
    assert [p.previous_items for p in plan.to_extract[1:4]] == [3, 3, 3]


def test_the_fetch_day_is_the_venues_day(db):
    """23:30 UTC on 14 October is already the 15th in Barcelona."""
    _add_page(db, LISTING_URL, LISTING_TEXT, fetched_at=datetime(2026, 10, 14, 23, 30, tzinfo=UTC))

    [listing] = _plan(db).to_extract

    assert listing.fetched_on == date(2026, 10, 15)


def test_shows_and_events_are_written_and_the_page_is_marked(db):
    _add_page(db, LISTING_URL, LISTING_TEXT, links=LINKS)
    events = (
        event("Inauguració", kind="opening", start_time=time(19), exhibition_title="PROVA U."),
        event("Xerrada", kind="talk", exhibition_title="Una altra"),
    )
    shown = show(detail_url=DETAIL_URL, summary_en=OUR_SUMMARY, media=("painting",))

    written = _write(db, [_result(db, LISTING_URL, shown, events=events)])

    assert written == extraction.Written(pages=1, exhibitions=1, events=2)
    [stored] = _shows(db)
    assert stored[:7] == (
        "Prova U",
        shown.start_date,
        shown.end_date,
        ["Artista Inventada"],
        DETAIL_URL,
        OUR_SUMMARY,
        LISTING_URL,
    )
    linked = run_sql(
        db,
        "SELECT e.title, e.kind, e.start_time, x.title FROM art_events e "
        "LEFT JOIN exhibitions x ON x.id = e.exhibition_id ORDER BY e.id",
    )[0]
    assert linked == [
        ("Inauguració", "opening", time(19), "Prova U"),
        ("Xerrada", "talk", None, None),  # names an exhibition nobody stored
    ]
    assert run_sql(db, MARKS)[0] == [(LISTING_URL, "ok", True, EXTRACT_VERSION, HAIKU, 3)]


def test_a_show_seen_on_its_own_page_and_again_on_the_listing_is_one_row(db):
    _add_page(db, LISTING_URL, LISTING_TEXT, links=LINKS)
    _add_page(db, DETAIL_URL, DETAIL_TEXT, kind="detail")
    listing = show(detail_url=DETAIL_URL, summary_en=OUR_SUMMARY)
    # The show's own page: more to say, but no dates and no links.
    detail = show(
        "PROVA U",
        artists=("Artista Inventada", "Segona Artista"),
        start_date=None,
        end_date=None,
        summary_en=LONGER_SUMMARY,
    )
    _write(db, [_result(db, LISTING_URL, listing), _result(db, DETAIL_URL, detail)])
    [first] = _shows(db)

    assert first[:7] == (
        "PROVA U",
        listing.start_date,  # the detail page gave none, so the listing's stay
        listing.end_date,
        ["Artista Inventada", "Segona Artista"],
        DETAIL_URL,
        LONGER_SUMMARY,
        DETAIL_URL,
    )

    # The listing changes: the run is extended, and its summary is still the short one.
    edited = LISTING_TEXT + " Prorrogada."
    _in_session(
        db,
        lambda s: s.execute(
            update(crawl.PAGES)
            .where(crawl.PAGES.c.url == LISTING_URL)
            .values(raw_text=edited, content_hash=crawl.text_hash(edited))
        ),
    )
    _write(db, [_result(db, LISTING_URL, show(end_date=date(2027, 1, 10), summary_en="Short."))])
    [second] = _shows(db)

    assert (second[2], second[5]) == (date(2027, 1, 10), LONGER_SUMMARY)
    assert second[7] == first[7]  # first_seen_at
    assert second[8] > first[8]  # last_seen_at


def test_a_page_that_names_a_show_twice_writes_one_row(db):
    """Postgres refuses an upsert that touches one row twice in a statement."""
    _add_page(db, LISTING_URL, LISTING_TEXT)

    written = _write(db, [_result(db, LISTING_URL, show(summary_en=OUR_SUMMARY), show("Prova U."))])

    assert written.exhibitions == 1
    assert [row[5] for row in _shows(db)] == [OUR_SUMMARY]


def test_a_recurring_event_is_one_row_per_date_and_a_missing_date_is_one_too(db):
    _add_page(db, AGENDA_URL, "agenda")
    _add_page(db, f"{HOST}/ca/agenda?p=2", "agenda 2")
    tours = [event(start_date=date(2026, 10, d)) for d in (17, 24)] + [event(start_date=None)]

    _write(db, [_result(db, AGENDA_URL, events=tours)])
    _write(db, [_result(db, f"{HOST}/ca/agenda?p=2", events=[event(start_date=None)])])

    [[(rows,)]] = run_sql(db, "SELECT count(*) FROM art_events")
    assert rows == 3


def test_a_failed_page_is_marked_and_writes_nothing(db):
    _add_page(db, LISTING_URL, LISTING_TEXT)
    [found] = _plan(db).to_extract

    _write(db, [PageResult(found, error="no usable answer")])

    assert run_sql(db, MARKS)[0] == [(LISTING_URL, "failed", True, EXTRACT_VERSION, HAIKU, None)]
    assert _shows(db) == []
    assert _plan(db).to_extract == ()


def test_a_marker_needs_its_hash_version_model_and_time(db):
    with pytest.raises(IntegrityError, match="ck_venue_pages_extract_marker"):
        _add_page(db, LISTING_URL, LISTING_TEXT, extract_status="ok")


def test_rolled_back_write_leaves_nothing(db):
    _add_page(db, LISTING_URL, LISTING_TEXT)

    _write(db, [_result(db, LISTING_URL, show())], commit=False)

    assert _shows(db) == []
    assert [p.url for p in _plan(db).to_extract] == [LISTING_URL]


# --- cli -----------------------------------------------------------------------------------------


def _show_input(title: str = "Prova U", **changes: Any) -> dict:
    """A show as a listing records it: facts only."""
    return {
        "title": title,
        "artists": ["Artista Inventada"],
        "start": {"day": 12, "month": 9, "year": 2026},
        "end": {"day": 30, "month": 11, "year": 2026},
        "at_venue": True,
        "link": None,
        "confidence": "high",
        **changes,
    }


def _answer(shows: list[dict] | None = None, events: list[dict] | None = None) -> dict:
    return {"exhibitions": [_show_input()] if shows is None else shows, "events": events or []}


def _clear_caches() -> None:
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


@pytest.fixture
def run(db, monkeypatch) -> Iterator[Callable[..., Any]]:
    """`run(*args)` invokes `extract` against `db`. `run.stub` is the model stub: queue its
    answers first."""
    monkeypatch.setenv("DATABASE_URL", db.render_as_string(hide_password=False))
    for var in ("EXTRACT_MODEL", "LANGFUSE_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    _clear_caches()
    stub = StubLlm(record=db_recorder(get_sessionmaker()))
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)

    def _run(*args: str):
        return CliRunner().invoke(cli.app, ["extract", *args])

    _run.stub = stub
    yield _run
    _clear_caches()


def _calls(url: URL) -> int:
    [[(calls,)]] = run_sql(url, "SELECT count(*) FROM llm_calls WHERE purpose = 'extract'")
    return calls


def test_extract_writes_and_reports_and_a_second_run_calls_no_model(run, db):
    """PR 22's done-when: a second run on unchanged pages makes zero model calls."""
    _add_page(db, LISTING_URL, LISTING_TEXT, links=LINKS)
    run.stub.tool_reply(
        extract.TOOL,
        _answer([_show_input(link=1)], [{**_show_input("Inauguració", end=None, link=None),
                                         "kind": "opening", "start_time": "19:00",
                                         "exhibition_title": "Prova U"}]),
        input_tokens=3300,
        output_tokens=600,
    )  # fmt: skip

    first = run()

    assert first.exit_code == 0, first.output
    assert "Extract\n" in first.output
    assert "pages       1   1 to extract   0 unchanged" in first.output
    assert "items       1 exhibitions   1 events" in first.output
    assert f"ok          {LISTING_URL}   1 exhibitions   1 events" in first.output
    assert "written     1 exhibitions   1 events   1 pages marked" in first.output
    # 3300 in and 600 out on Haiku 4.5 ($1 / $5 per million).
    assert "cost        $0.006300   1 model calls   $0.006300 a page" in first.output
    assert [row[4] for row in _shows(db)] == [DETAIL_URL]

    second = run()

    assert second.exit_code == 0, second.output
    assert "pages       1   0 to extract   1 unchanged" in second.output
    assert "cost        $0   0 model calls" in second.output
    assert _calls(db) == 1


def test_extract_prints_no_page_text(run, db):
    _add_page(db, LISTING_URL, LISTING_TEXT, links=LINKS)
    run.stub.tool_reply(extract.TOOL, _answer())

    result = run()

    assert "Fictici" not in result.output
    assert "Prova U" not in result.output  # nor titles: counts only


def test_dry_run_rolls_back_but_keeps_the_spend(run, db):
    _add_page(db, LISTING_URL, LISTING_TEXT)
    run.stub.tool_reply(extract.TOOL, _answer())

    result = run("--dry-run")

    assert result.exit_code == 0, result.output
    assert "Extract (dry run, rolled back)" in result.output
    assert _shows(db) == []
    assert run_sql(db, MARKS)[0] == [(LISTING_URL, None, None, None, None, None)]
    assert _calls(db) == 1


def test_failed_page_is_marked_reported_and_exits_1_then_waits(run, db):
    _add_page(db, LISTING_URL, LISTING_TEXT)
    run.stub.tool_reply(extract.TOOL, {"exhibitions": "none"})
    run.stub.tool_reply(extract.TOOL, {"exhibitions": "none"})

    first = run()

    assert first.exit_code == 1
    assert f"failed      {LISTING_URL}   record_programme: no usable answer" in first.output
    assert run_sql(db, MARKS)[0][0][1] == "failed"
    second = run()
    assert second.exit_code == 0, second.output
    assert _calls(db) == 2


def test_listing_that_stops_yielding_is_a_broken_seed(run, db):
    _add_page(db, LISTING_URL, LISTING_TEXT, **marks(LISTING_TEXT, extracted_hash="0" * 64))
    run.stub.tool_reply(extract.TOOL, _answer([]))

    result = run()

    assert result.exit_code == 1
    assert f"broken      {LISTING_URL}   no items, after 3 last time" in result.output
    assert run_sql(db, MARKS)[0][0][5] == 0


def test_purge_runs_first_and_a_purged_page_is_reported_not_sent(run, db):
    _add_page(db, LISTING_URL, LISTING_TEXT, age=timedelta(days=7, minutes=1))

    result = run()

    assert result.exit_code == 0, result.output
    assert "purged      1 pages of text older than 7 days" in result.output
    assert f"expired     {LISTING_URL}   its text was purged" in result.output
    assert _calls(db) == 0


def test_limit_leaves_the_rest_for_the_next_run(run, db):
    _add_page(db, LISTING_URL, LISTING_TEXT)
    _add_page(db, AGENDA_URL, "agenda")
    run.stub.tool_reply(extract.TOOL, _answer())

    result = run("--limit", "1")

    assert result.exit_code == 0, result.output
    assert "extracted   1   1 ok   1 left by --limit" in result.output
    assert [p.url for p in _plan(db).to_extract] == [AGENDA_URL]


def test_venue_selects_its_pages_and_an_unknown_one_names_the_known(run, db):
    _add_page(db, "http://macba.test/ara", "macba", venue_id=2)
    _add_page(db, LISTING_URL, LISTING_TEXT)
    run.stub.tool_reply(extract.TOOL, _answer([]))

    unknown = run("--venue", "Galeria Exmple")
    macba = run("--venue", "macba")

    assert unknown.exit_code == 1
    assert "Known: Galeria Exemple, MACBA" in unknown.output
    assert macba.exit_code == 0, macba.output
    assert "pages       1   1 to extract   0 unchanged" in macba.output
    assert "ok          http://macba.test/ara   0 exhibitions   0 events" in macba.output
    assert _calls(db) == 1


def test_no_crawled_pages_fails(run):
    result = run()

    assert result.exit_code == 1
    assert "Run crawl --pilot first" in result.output
