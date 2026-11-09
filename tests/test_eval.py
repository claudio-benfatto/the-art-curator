"""`eval/extraction.py`, `eval/runner.py` and `cli eval-extraction`: the gold set, how items are
matched and scored, discovery against the labelled listings, where a labelled page's text comes
from, and the drafts.

No model is called: `StubLlm` fakes the wire. Every page and label here is invented; the listing
is the synthetic fixture `tests/fixtures/crawl/listing.html`.
"""

import asyncio
from collections.abc import Callable, Iterator
from datetime import date
from typing import Any

import pytest
from alembic import command
from pydantic import ValidationError
from sqlalchemy.engine import URL
from typer.testing import CliRunner

from art_curator import cli
from art_curator.config import get_settings
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.eval import extraction as evaluation
from art_curator.eval import runner
from art_curator.eval.extraction import (
    GoldItem,
    GoldPage,
    GoldSet,
    Item,
    Outcome,
    SeedVerdict,
    SkipReason,
    contained_title,
    load_gold,
    match_items,
    on_or_coming,
    same_title,
    score_discovery,
    score_page,
    totals,
)
from art_curator.eval.runner import PageText, StoredPage
from art_curator.ingest import pages
from art_curator.ingest.crawl import text_hash
from art_curator.ingest.discover import url_key
from art_curator.ingest.extraction import StoredLink
from art_curator.ingest.http import RobotsPolicy, open_client
from art_curator.llm import extract
from art_curator.llm.client import db_recorder
from art_curator.llm.extract import MAX_LISTED_ITEMS, Extraction
from tests.db import alembic_config, run_sql
from tests.llm_stub import StubLlm
from tests.site_stub import HOST, LISTING, POLICY, FakeSite, html, no_sleep

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
VENUE = "Galeria Exemple"
LISTING_URL = f"{HOST}/ca/exposicions/"
DETAIL_URL = f"{HOST}/ca/exposicions/prova-u"
LISTING_TEXT = pages.page_text(LISTING)
LISTING_HASH = text_hash(LISTING_TEXT)
FETCHED_ON = date(2026, 10, 10)


def gold_page(url: str = LISTING_URL, *items: GoldItem, **changes: Any) -> GoldPage:
    fields = {
        "url": url,
        "venue": VENUE,
        "kind": "listing",
        "content_hash": LISTING_HASH,
        "confirmed": True,
        "items": items,
    }
    return GoldPage(**(fields | changes))


def label(title: str = "Prova U", **changes: Any) -> GoldItem:
    fields = {
        "title": title,
        "artists": ("Artista Inventada",),
        "start": date(2026, 9, 12),
        "end": date(2026, 11, 30),
    }
    return GoldItem(**(fields | changes))


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


def item(title: str, kind: str = "exhibition", start: date | None = None) -> Item:
    return Item(kind, title, (), start, None, True)  # type: ignore[arg-type]


# --- the gold set --------------------------------------------------------------------------------


def test_the_committed_gold_set_is_valid():
    gold = load_gold()

    assert all(len(page.content_hash) == 64 for page in gold.pages)


def test_a_page_is_labelled_with_facts_only_and_the_defaults_spelled_out():
    gold = load_gold(
        f"""
listings:
  Galeria Exemple: {LISTING_URL}
  Galeria Buida: []
pages:
  - url: {LISTING_URL}
    venue: Galeria Exemple
    kind: listing
    content_hash: {LISTING_HASH}
    items:
      - title: Prova U
        start: 2026-09-12
      - kind: event
        title: Inauguració
        at_venue: false
"""
    )

    assert gold.listings == {"Galeria Exemple": (LISTING_URL,), "Galeria Buida": ()}
    [page] = gold.pages
    assert page.confirmed is False  # a person says so, explicitly
    assert page.items == (
        GoldItem(title="Prova U", start=date(2026, 9, 12)),
        GoldItem(kind="event", title="Inauguració", at_venue=False),
    )
    assert page.items[0].artists == () and page.items[0].at_venue is True


def test_an_empty_gold_set_is_valid():
    assert load_gold("listings:\npages:\n") == GoldSet()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"content_hash": "abc"}, "content_hash"),
        ({"url": "salapares.com/exposiciones"}, r"not an http\(s\) URL"),
        ({"items": [{"title": "Prova U", "start": "2026-11-30", "end": "2026-09-12"}]}, "ends"),
        ({"items": [{"title": "Prova U", "dates": "12.09"}]}, "Extra inputs"),
        ({"items": None}, "items"),
    ],
)
def test_a_bad_label_is_refused(changes, message):
    fields = {"url": LISTING_URL, "venue": VENUE, "kind": "listing", "content_hash": LISTING_HASH}

    with pytest.raises(ValidationError, match=message):
        GoldPage(**(fields | {"items": []} | changes))


def test_a_listing_is_labelled_only_as_far_as_extraction_reads_it():
    items = [label(f"Mostra {n}") for n in range(MAX_LISTED_ITEMS + 1)]

    with pytest.raises(ValidationError, match=f"Label only the first {MAX_LISTED_ITEMS}"):
        gold_page(LISTING_URL, *items)
    assert len(gold_page(DETAIL_URL, *items, kind="detail").items) == MAX_LISTED_ITEMS + 1


def test_only_a_listing_with_items_can_be_labelled_partially():
    assert gold_page(LISTING_URL, label(), partial=True).partial
    with pytest.raises(ValidationError, match="only a listing"):
        gold_page(DETAIL_URL, label(), kind="detail", partial=True)
    with pytest.raises(ValidationError, match="needs its first items"):
        gold_page(LISTING_URL, partial=True)


def test_one_label_per_page_and_one_listings_entry_per_venue():
    with pytest.raises(ValidationError, match="more than one label"):
        GoldSet(pages=(gold_page(LISTING_URL), gold_page(LISTING_URL.rstrip("/"))))
    with pytest.raises(ValidationError, match="more than one listings entry"):
        GoldSet(listings={"Galería Alegría": (), "galeria alegria": ()})


def test_venue_keeps_that_venues_pages_and_listings():
    gold = GoldSet(
        listings={VENUE: (LISTING_URL,), "MACBA": ()},
        pages=(gold_page(), gold_page("http://macba.test/ara", venue="MACBA")),
    )

    only = gold.for_venue("galeria exemple")

    assert list(only.listings) == [VENUE]
    assert [page.url for page in only.pages] == [LISTING_URL]
    assert gold.venues == [VENUE, "MACBA"]


# --- matching ------------------------------------------------------------------------------------


@pytest.mark.parametrize(("a", "b"), [("Prova U", "PROVA U."), ("Àngels", "angels")])
def test_titles_are_compared_as_title_keys(a, b):
    assert same_title(a, b)


@pytest.mark.parametrize(
    ("a", "b", "matched"),
    [
        ("Prova U", "Prova U. Obra recent", True),  # a dropped subtitle
        ("Obra recent", "Prova U. Obra recent", True),
        ("Prova", "Provant", False),  # words, not letters
        ("Art", "Art i ciència", False),  # too short to be told apart
    ],
)
def test_a_title_inside_another_matches_by_whole_words(a, b, matched):
    assert contained_title(a, b) is matched


def test_exact_titles_are_matched_before_contained_ones():
    labels = [item("Prova U"), item("Prova U. Obra recent")]
    found = [item("Prova U. Obra recent"), item("Prova U")]

    matches, missed, extra = match_items(labels, found)

    assert [(m.label.title, m.found.title) for m in matches] == [
        ("Prova U", "Prova U"),
        ("Prova U. Obra recent", "Prova U. Obra recent"),
    ]
    assert missed == extra == []


def test_the_same_kind_then_the_same_start_breaks_a_tie():
    tour = "Visita guiada"
    labels = [item(tour, "event", date(2026, 10, 17))]
    found = [
        item(tour, "exhibition", date(2026, 10, 17)),
        item(tour, "event", date(2026, 10, 10)),
        item(tour, "event", date(2026, 10, 17)),
    ]

    [match], _, extra = match_items(labels, found)

    assert match.found == found[2]
    assert extra == found[:2]


def test_what_is_left_is_missed_or_extra():
    _, missed, extra = match_items([item("Prova U"), item("Tercera")], [item("Segona mostra")])

    assert [i.title for i in missed] == ["Prova U", "Tercera"]
    assert [i.title for i in extra] == ["Segona mostra"]


# --- scoring -------------------------------------------------------------------------------------


def test_a_page_is_scored_field_by_field_on_matched_items():
    page = gold_page(
        LISTING_URL,
        label("Prova U"),
        label("Segona mostra", artists=("Col·lectiu Fictici", "Una Altra")),
        label("Tercera", at_venue=False),
        label("Inauguració", kind="event", artists=(), end=None),
        label("No trobada"),
    )
    found = Extraction(
        exhibitions=(
            show("PROVA U"),
            show("Segona mostra", artists=("una altra", "COL·LECTIU FICTICI")),  # names, any order
            show("Tercera", end_date=date(2026, 12, 31)),  # wrong end, wrong at_venue
            show("Inventada"),
        ),
        events=(),
    )

    score = score_page(page, found)
    total = totals([score])

    assert [i.title for i in score.missed] == ["Inauguració", "No trobada"]
    assert [i.title for i in score.extra] == ["Inventada"]
    assert (total.recall.hits, total.recall.total) == (3, 5)
    assert (total.precision.hits, total.precision.total) == (3, 4)
    assert (total.dates.hits, total.artists.hits, total.at_venue.hits) == (2, 3, 2)
    assert str(total.recall) == "0.60 (3/5)"


def test_an_event_recorded_as_an_exhibition_is_found_but_the_kind_is_wrong():
    page = gold_page(LISTING_URL, label("Inauguració", kind="event", artists=(), end=None))
    found = Extraction(exhibitions=(show("Inauguració", artists=(), end_date=None),), events=())

    total = totals([score_page(page, found)])

    assert (total.recall.hits, total.kind.hits, total.kind.total) == (1, 0, 1)


def test_a_failed_page_counts_as_finding_nothing():
    score = score_page(gold_page(LISTING_URL, label()), None, "no usable answer")
    total = totals([score])

    assert total.failed == 1
    assert (total.recall.hits, total.recall.total) == (0, 1)
    assert str(total.precision) == "n/a (0/0)"


def test_items_on_an_empty_page_are_invented():
    empty = score_page(gold_page(f"{HOST}/ca/agenda"), Extraction((show(),), (event(),)))
    full = score_page(gold_page(LISTING_URL, label()), Extraction((show(),), ()))

    total = totals([empty, full])

    assert (total.empty_pages, total.invented) == (1, 2)


def test_a_partial_listing_counts_towards_recall_but_not_precision():
    partial = gold_page(LISTING_URL, label("Prova U"), label("Tercera"), partial=True)
    full = gold_page(DETAIL_URL, label("Prova U"), kind="detail")
    found = Extraction((show("Prova U"), show("Més enllà del primer"), show("Ni aquesta")), ())

    total = totals([score_page(partial, found), score_page(full, Extraction((show(),), ()))])

    assert (total.recall.hits, total.recall.total) == (2, 3)
    assert (total.precision.hits, total.precision.total) == (1, 1)


@pytest.mark.parametrize(
    ("changes", "current"),
    [
        ({"end": date(2026, 10, 10)}, True),  # its last day
        ({"end": date(2026, 10, 9)}, False),
        ({"end": None}, True),  # an exhibition with no end may be on
        ({"kind": "event", "start": date(2026, 10, 9), "end": None}, False),
        ({"at_venue": False}, False),
    ],
)
def test_a_current_show_is_one_not_over_at_the_venue(changes, current):
    assert on_or_coming(label(**changes), FETCHED_ON) is current


# --- discovery -----------------------------------------------------------------------------------


def verdict(venue: str, url: str, status: str, page_type="current_listing", confidence="high"):
    return SeedVerdict(venue, url, page_type, confidence, status)


def test_discovery_is_scored_per_labelled_venue():
    listings = {
        "Correcta": ("https://correcta.test/exposicions",),
        "Equivocada": ("https://equivocada.test/exposicions",),
        "Dubtosa": ("https://dubtosa.test/agenda",),
        "Fixada": ("https://fixada.test/ara",),
        "Sense llistat": (),
    }
    verdicts = [
        verdict("Correcta", "https://www.correcta.test/exposicions/", "accepted"),  # spelling
        verdict("Correcta", "https://correcta.test/arxiu", "rejected", "past_archive"),
        verdict("Equivocada", "https://equivocada.test/exposicions", "accepted"),
        verdict("Equivocada", "https://equivocada.test/noticies", "accepted", "agenda"),
        verdict("Dubtosa", "https://dubtosa.test/agenda", "ambiguous", "agenda", "medium"),
        verdict("Sense llistat", "https://sensellistat.test/", "accepted"),
        verdict("No etiquetada", "https://noetiquetada.test/", "accepted"),
    ]

    score = score_discovery(listings, verdicts)

    assert {v.venue: v.outcome for v in score.venues} == {
        "Correcta": Outcome.CORRECT,
        "Dubtosa": Outcome.AMBIGUOUS,
        "Equivocada": Outcome.WRONG,
        "Fixada": Outcome.UNDISCOVERED,  # pinned before discovery ran
        "Sense llistat": Outcome.WRONG,  # it has no listing, so anything accepted is wrong
    }
    [wrong] = [v for v in score.venues if v.venue == "Equivocada"]
    assert wrong.wrong == ("https://equivocada.test/noticies",)
    assert str(score.share(Outcome.CORRECT)) == "0.25 (1/4)"
    assert str(score.share(Outcome.AMBIGUOUS)) == "0.25 (1/4)"
    assert str(score.share(Outcome.WRONG)) == "0.50 (2/4)"
    assert [(c.page_type, c.confidence, c.candidates, c.listings) for c in score.calibration] == [
        ("current_listing", "high", 3, 2),
        ("agenda", "high", 1, 0),
        ("agenda", "medium", 1, 1),
        ("past_archive", "high", 1, 0),
    ]
    assert score.unlabelled == ("No etiquetada",)


# --- the report ----------------------------------------------------------------------------------


def test_the_report_names_what_is_wrong_and_what_was_not_scored():
    page = gold_page(LISTING_URL, label("Prova U"), label("Tercera"))
    found = Extraction((show("Prova U", end_date=date(2026, 12, 30)), show("Inventada")), ())
    gold = GoldSet(
        listings={VENUE: (LISTING_URL,)},
        pages=(
            page,
            gold_page(f"{HOST}/vella", confirmed=False),
            gold_page(f"{HOST}/canviada"),
        ),
    )
    skipped = [
        evaluation.Skipped(gold.pages[1], SkipReason.UNCONFIRMED),
        evaluation.Skipped(gold.pages[2], SkipReason.STALE),
    ]
    discovery = score_discovery(gold.listings, [verdict(VENUE, LISTING_URL, "accepted")])

    lines = evaluation.report(gold, FETCHED_ON, [score_page(page, found)], skipped, discovery)

    assert lines[:3] == [
        "gold        2 pages   1 venues   1 with no current show   1 empty   1 unconfirmed",
        "scored      1 pages   1 unconfirmed   1 stale",
        "recall      0.50 (1/2)",
    ]
    assert "invented    0 items on 0 empty pages" in lines
    assert f"ok          {LISTING_URL}   1 of 2 labelled   2 extracted" in lines
    assert "            missed    'Tercera'" in lines
    assert "            extra     'Inventada'" in lines
    assert (
        "            dates     'Prova U'   2026-09-12 – 2026-11-30, got 2026-09-12 – 2026-12-30"
    ) in lines
    assert f"stale       {HOST}/canviada   changed since it was labelled: label it again" in lines
    assert "discovery   1 labelled venues   1 correct" in lines
    assert "verdict     current_listing high   1 candidates   1 labelled listings" in lines


def test_the_report_does_not_call_what_a_partial_listing_found_past_its_labels_extra():
    page = gold_page(LISTING_URL, label("Prova U"), partial=True)
    found = Extraction((show("Prova U"), show("Més enllà del primer")), ())
    gold = GoldSet(pages=(page,))
    nothing = evaluation.DiscoveryScore()

    lines = evaluation.report(gold, FETCHED_ON, [score_page(page, found)], [], nothing)

    assert lines[0].endswith("0 empty   1 partial")
    assert (
        f"ok          {LISTING_URL}   1 of 1 labelled   2 extracted   partial: extras not judged"
    ) in lines
    assert not any("extra " in line for line in lines)


# --- the labelled pages' text --------------------------------------------------------------------


def stored(text: str | None = LISTING_TEXT, url: str = LISTING_URL) -> dict[str, StoredPage]:
    page = None
    if text is not None:
        links = (StoredLink(1, DETAIL_URL, "Prova U"),)
        # Stored under another name: the label's venue is the one extraction is told.
        page = PageText(url, "Altre nom", True, text_hash(text), FETCHED_ON, text, links)
    hash_ = text_hash(text) if text is not None else LISTING_HASH
    return {url_key(url): StoredPage(page, url, VENUE, True, hash_)}


def _find(gold: list[GoldPage], rows: dict[str, StoredPage], site: FakeSite | None = None):
    site = site or FakeSite()

    async def main():
        async with open_client(POLICY, transport=site.transport, sleep=no_sleep) as client:
            return await runner.find_texts(
                client, RobotsPolicy(client), gold, rows, date(2026, 11, 6)
            )

    ready, skipped = asyncio.run(main())
    return ready, skipped, site


def test_the_stored_text_is_used_when_it_is_the_labelled_one():
    [(_, text)], skipped, site = _find([gold_page()], stored())

    assert skipped == []
    assert site.requests == []
    assert (text.fetched_on, text.venue_name, len(text.links)) == (FETCHED_ON, VENUE, 1)


def test_a_page_whose_text_was_purged_is_fetched_again_with_its_links():
    [(_, text)], skipped, site = _find([gold_page()], stored(text=None))

    assert skipped == []
    assert site.paths()[-1] == "/ca/exposicions/"
    assert text.fetched_on == date(2026, 11, 6)  # read against today, as fetched today
    assert text.text == LISTING_TEXT
    assert [link.url for link in text.links][:2] == [f"{HOST}/ca/exposicions/", f"{HOST}/ca/agenda"]


def test_a_detail_page_fetched_again_has_no_links():
    detail = gold_page(LISTING_URL, kind="detail")

    [(_, text)], _, _ = _find([detail], {})

    assert text.links == ()


def test_a_page_changed_since_it_was_labelled_is_stale():
    """Changed in the database and on the site: a fresh fetch is tried before giving up."""
    site = FakeSite({"/ca/exposicions/": html(LISTING.replace(b"Prova U", b"Prova V"))})

    ready, [skip], _ = _find([gold_page()], stored(text="Galeria Exemple. Canviada."), site)

    assert ready == []
    assert skip.reason is SkipReason.STALE
    assert site.paths()[-1] == "/ca/exposicions/"


def test_a_page_that_cannot_be_fetched_is_unreadable_and_a_draft_is_not_scored():
    site = FakeSite({"/ca/exposicions/": html("", status=403)})
    draft = gold_page(f"{HOST}/ca/agenda", confirmed=False)

    ready, skipped, _ = _find([gold_page(), draft], {}, site)

    assert ready == []
    assert [(s.reason, s.detail) for s in skipped] == [
        (SkipReason.UNREADABLE, "blocked (HTTP 403)"),
        (SkipReason.UNCONFIRMED, ""),
    ]
    assert site.paths().count("/ca/agenda") == 0  # a draft is not even fetched


# --- the model and the drafts --------------------------------------------------------------------


def _show_input(title: str = "Prova U", **changes: Any) -> dict:
    return {
        "title": title,
        "artists": ["Artista Inventada"],
        "start": {"day": 12, "month": 9, "year": None},
        "end": {"day": 30, "month": 11, "year": None},
        "at_venue": True,
        "link": 1,
        "confidence": "high",
        **changes,
    }


def _text(url: str = LISTING_URL) -> PageText:
    [row] = stored(url=url).values()
    assert row.page is not None
    return row.page


def test_pages_are_extracted_as_cli_extract_does_and_a_failed_one_is_kept():
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, {"exhibitions": [_show_input()], "events": []})
    stub.tool_reply(extract.TOOL, {"exhibitions": "none"})
    stub.tool_reply(extract.TOOL, {"exhibitions": "none"})

    done = asyncio.run(runner.extract_all(stub.client, HAIKU, [_text(), _text(f"{HOST}/b")]))

    ok, failed = done.results
    assert ok.extraction is not None
    [found] = ok.extraction.exhibitions
    assert (found.start_date, found.detail_url) == (date(2026, 9, 12), DETAIL_URL)
    assert failed.extraction is None and "no usable answer" in failed.error
    assert done.stopped == ""


def test_a_draft_is_a_gold_set_of_unconfirmed_labels():
    found = Extraction((show(),), (event(start_date=date(2026, 10, 17)),))
    results = [
        runner.Result(_text(), found),
        runner.Result(_text(f"{HOST}/b"), error="no usable answer"),
    ]

    text = runner.draft_yaml(results, HAIKU, date(2026, 11, 6))
    gold = load_gold(text)

    assert text.startswith("# Drafted by `eval-extraction --draft` on 2026-11-06 with ")
    [page] = gold.pages  # nothing to draft from a page that failed
    assert (page.url, page.confirmed, page.content_hash) == (LISTING_URL, False, LISTING_HASH)
    assert page.items == (
        label(),
        GoldItem(kind="event", title="Visita guiada", start=date(2026, 10, 17)),
    )
    assert "summary" not in text


def test_a_long_listing_is_drafted_partially_and_a_show_page_whole():
    many = Extraction(
        tuple(show(f"Mostra {n}") for n in range(runner.DRAFTED_ITEMS + 1)), (event(),)
    )
    detail = PageText(DETAIL_URL, VENUE, False, LISTING_HASH, FETCHED_ON, "", ())
    results = [runner.Result(_text(), many), runner.Result(detail, many)]

    listing, show_page = load_gold(runner.draft_yaml(results, HAIKU, FETCHED_ON)).pages

    assert listing.partial and len(listing.items) == runner.DRAFTED_ITEMS + 1
    assert listing.items[-2].title == f"Mostra {runner.DRAFTED_ITEMS - 1}"
    assert listing.items[-1].kind == "event"
    assert not show_page.partial and len(show_page.items) == runner.DRAFTED_ITEMS + 2


def test_drafts_are_for_pages_not_labelled_yet_that_still_hold_text():
    rows = stored() | stored(url=f"{HOST}/b") | stored(text=None, url=f"{HOST}/purgada")

    drafts = runner.draft_pages(rows, GoldSet(pages=(gold_page(),)), None)

    assert [page.url for page in drafts] == [f"{HOST}/b"]
    assert runner.draft_pages(rows, GoldSet(), "MACBA") == []


# --- cli (database) ------------------------------------------------------------------------------

VENUE_ROWS = f"""
INSERT INTO venues (id, source_venue_id, name, slug, website_url, crawl_enabled, is_pilot)
OVERRIDING SYSTEM VALUE VALUES (1, 101, 'Galeria Exemple', 'galeria-exemple', '{HOST}/', true, true)
"""
PAGE_ROW = f"""
INSERT INTO venue_pages (venue_id, url, kind, status, http_status, content_hash, raw_text, links,
                         fetched_at)
VALUES (1, '{LISTING_URL}', 'listing', 'ok', 200, '{LISTING_HASH}', $${LISTING_TEXT}$$,
        '[{{"number": 1, "url": "{DETAIL_URL}", "text": "Prova U"}}]', now())
"""
SEED_ROW = f"""
INSERT INTO venue_seeds (venue_id, url, page_type, confidence, dated_items, language, status,
                         model)
VALUES (1, '{LISTING_URL}', 'current_listing', 'high', 3, 'ca', 'accepted', '{HAIKU}')
"""
UNTOUCHED = (
    "SELECT (SELECT count(*) FROM exhibitions) + (SELECT count(*) FROM art_events), "
    "(SELECT count(*) FROM venue_pages WHERE extract_status IS NOT NULL)"
)


@pytest.fixture(scope="module")
def eval_db(fresh_database) -> Iterator[URL]:
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@pytest.fixture
def db(eval_db) -> URL:
    run_sql(eval_db, "TRUNCATE venue_pages, venue_seeds, venues, llm_calls CASCADE", VENUE_ROWS)
    return eval_db


def _clear_caches() -> None:
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


@pytest.fixture
def run(db, monkeypatch) -> Iterator[Callable[..., Any]]:
    """`run(*args, gold=GoldSet)` invokes `eval-extraction` against `db` with that gold set.
    `run.stub` is the model stub, `run.site` the fake venue site."""
    monkeypatch.setenv("DATABASE_URL", db.render_as_string(hide_password=False))
    for var in ("EXTRACT_MODEL", "LANGFUSE_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    _clear_caches()
    stub = StubLlm(record=db_recorder(get_sessionmaker()))
    site = FakeSite()
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)
    monkeypatch.setattr(
        cli,
        "open_client",
        lambda: open_client(POLICY, transport=site.transport, sleep=no_sleep),
    )

    def _run(*args: str, gold: GoldSet | None = None):
        monkeypatch.setattr(evaluation, "load_gold", lambda: gold or GoldSet())
        return CliRunner().invoke(cli.app, ["eval-extraction", *args])

    _run.stub = stub
    _run.site = site
    yield _run
    _clear_caches()


def test_eval_scores_the_stored_text_and_discovery_and_writes_nothing(run, db):
    run_sql(db, PAGE_ROW, SEED_ROW)
    labels = (label("Prova U", start=date(2026, 9, 12)), label("Segona mostra"))
    gold = GoldSet(listings={VENUE: (LISTING_URL,)}, pages=(gold_page(LISTING_URL, *labels),))
    run.stub.tool_reply(
        extract.TOOL,
        {"exhibitions": [_show_input()], "events": []},
        input_tokens=3300,
        output_tokens=600,
    )

    result = run(gold=gold)

    assert result.exit_code == 0, result.output
    assert "Eval extraction\n" in result.output
    assert "recall      0.50 (1/2)" in result.output
    assert "missed    'Segona mostra'" in result.output
    assert "discovery   1 labelled venues   1 correct" in result.output
    assert "cost        $0.006300   1 model calls" in result.output
    assert run.site.requests == []  # the stored text was the labelled one
    assert run_sql(db, UNTOUCHED) == [[(0, 0)]]
    assert "Fictici" not in result.output  # page text is never printed


def test_eval_of_a_venue_the_gold_set_does_not_label_names_the_known(run):
    result = run("--venue", "MACBA", gold=GoldSet(pages=(gold_page(),)))

    assert result.exit_code == 1
    assert "labels nothing for MACBA. Known: Galeria Exemple" in result.output


def test_draft_writes_unconfirmed_labels_and_never_overwrites(run, db, tmp_path):
    run_sql(db, PAGE_ROW)
    run.stub.tool_reply(extract.TOOL, {"exhibitions": [_show_input()], "events": []})
    path = tmp_path / "draft.yaml"

    result = run("--draft", str(path))
    again = run("--draft", str(path))

    assert result.exit_code == 0, result.output
    assert f"drafted     1 pages to {path}, every one unconfirmed" in result.output
    [page] = load_gold(path.read_text(encoding="utf-8")).pages
    assert (page.url, page.confirmed, [i.title for i in page.items]) == (
        LISTING_URL,
        False,
        ["Prova U"],
    )
    assert again.exit_code == 1
    assert "exists" in again.output
    assert run_sql(db, UNTOUCHED) == [[(0, 0)]]
