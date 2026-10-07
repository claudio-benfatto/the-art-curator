"""`ingest/seeds.py`, `ingest/pages.py` and `cli discover`: overrides, one site end to end over a
fake website and a stubbed model, the `venue_seeds` write, and the report.

No model is called and no venue site is fetched: `StubLlm` and `FakeSite` fake the wire only.
"""

import asyncio
import functools
from collections.abc import Callable, Iterator
from typing import Any, get_args

import httpx2
import pytest
from alembic import command
from pydantic import ValidationError
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from typer.testing import CliRunner

from art_curator import cli
from art_curator.config import get_settings
from art_curator.db import models
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.ingest import discover, pages, seeds
from art_curator.ingest.discover import (
    PASS1_TOOL,
    PASS2_TOOL,
    Candidate,
    GrafHints,
    PageVerdict,
    SeedStatus,
)
from art_curator.ingest.http import RobotsPolicy, open_client
from art_curator.ingest.matching import PILOT_VENUES
from art_curator.ingest.pages import PageStatus
from art_curator.ingest.seeds import (
    SeedOverrides,
    Site,
    SiteOutcome,
    SiteResult,
    Skipped,
    load_overrides,
)
from art_curator.llm.client import db_recorder
from tests.db import alembic_config, run_sql
from tests.llm_stub import StubLlm
from tests.site_stub import HOST, LISTING, POLICY, FakeSite, html, no_sleep

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
VENUE = "Galeria Exemple"
LISTING_URL = f"{HOST}/ca/exposicions/"
AGENDA_URL = f"{HOST}/ca/agenda"
# The homepage fixture's links, as `collect_links` numbers them (tests/test_discover.py).
LISTING_LINK, AGENDA_LINK, ARTISTS_LINK = 2, 3, 4

SITE = Site(venue_id=1, name=VENUE, url=f"{HOST}/")


def _verdict(page_type: str, confidence: str = "high", dated_items: int = 3) -> PageVerdict:
    return PageVerdict(
        page_type=page_type, dated_items=dated_items, language="ca", confidence=confidence
    )


def _prompt(request: dict) -> str:
    blocks = [*request["system"], *(b for m in request["messages"] for b in m["content"])]
    return "\n".join(b["text"] for b in blocks if "text" in b)


def _discover(site: Site, fake: FakeSite, stub: StubLlm) -> SiteResult:
    async def main() -> SiteResult:
        async with open_client(POLICY, transport=fake.transport, sleep=no_sleep) as client:
            return await seeds.discover_site(client, RobotsPolicy(client), stub.client, HAIKU, site)

    return asyncio.run(main())


def _fetch(fake: FakeSite, url: str) -> pages.Fetched:
    async def main() -> pages.Fetched:
        async with open_client(POLICY, transport=fake.transport, sleep=no_sleep) as client:
            return await pages.fetch_page(client, RobotsPolicy(client), url)

    return asyncio.run(main())


# --- seeds.yaml ----------------------------------------------------------------------------------


def test_committed_overrides_load_and_name_only_pilots():
    """A typo in a venue name would never apply. `discover` reports it; this catches it first."""
    overrides = load_overrides()
    assert overrides.unmatched(PILOT_VENUES) == []


def test_override_is_found_whatever_the_case_or_accents():
    overrides = load_overrides(
        "venues:\n"
        "  àngels barcelona:\n"
        "    pin: https://angels.test/exhibitions\n"
        "    reject:\n"
        "      - https://angels.test/news\n"
        "      - https://angels.test/artists\n"
    )

    found = overrides.for_venue("Angels Barcelona")
    assert found.pin == ("https://angels.test/exhibitions",)  # one URL need not be a list
    assert found.reject == ("https://angels.test/news", "https://angels.test/artists")
    assert overrides.for_venue("MACBA") == seeds.VenueOverride()


@pytest.mark.parametrize("text", ["", "venues:\n", "venues: {}\n"])
def test_empty_overrides_are_valid(text):
    assert load_overrides(text) == SeedOverrides()


@pytest.mark.parametrize(
    "text",
    [
        "venues:\n  MACBA:\n    pin: macba.cat/exposicions\n",  # no scheme
        "venues:\n  MACBA:\n    pin: mailto:info@macba.cat\n",
        "venues:\n  MACBA:\n    note: the page says ...\n",  # no free text (CLAUDE.md § 1)
        "venues:\n  MACBA:\n    pin: https://m.test/a\n    reject: https://m.test/a/\n",
        "venues:\n  MACBA: {}\n  macba: {}\n",  # two entries for one venue
        "sites: {}\n",
    ],
)
def test_bad_overrides_are_refused(text):
    with pytest.raises(ValidationError):
        load_overrides(text)


def test_unmatched_override_is_named():
    overrides = load_overrides("venues:\n  Galeria Exmple:\n    reject: https://e.test/x\n")
    assert overrides.unmatched([VENUE]) == ["Galeria Exmple"]


# --- pages ---------------------------------------------------------------------------------------


def test_page_text_is_visible_text_on_one_line():
    text = pages.page_text(LISTING)

    assert "Prova U Artista Inventada 12.09.2026 – 30.11.2026" in text  # the dates survive
    assert text.startswith("Exposicions Agenda Artistes Visita")  # and so does the navigation
    # trafilatura's default cleaning: no scripts, and no <footer> either.
    assert "MARCA-SCRIPT" not in text and "De dimarts a dissabte" not in text and "<" not in text
    assert "\n" not in text and "  " not in text


def test_page_text_is_capped():
    body = "<html><body><p>" + "paraula " * 10_000 + "</p></body></html>"
    assert len(pages.page_text(body)) == pages.MAX_TEXT_CHARS


@pytest.mark.parametrize(
    ("response", "status"),
    [
        (html(LISTING), PageStatus.OK),
        (html("<html><body><div id='app'></div><p>Carregant</p></body></html>"), PageStatus.THIN),
        (html("no", 403), PageStatus.BLOCKED),
        (html("no", 401), PageStatus.BLOCKED),
        (html(LISTING, 503, **{"cf-mitigated": "challenge"}), PageStatus.BLOCKED),
        (html("gone", 404), PageStatus.ERROR),
        (html("oops", 500), PageStatus.ERROR),
        (httpx2.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}),
         PageStatus.ERROR),
    ],
)  # fmt: skip
def test_fetch_page_classifies_what_came_back(response, status):
    fetched = _fetch(FakeSite({"/p": response}), f"{HOST}/p")

    assert fetched.status is status
    assert fetched.http_status == response.status_code
    assert bool(fetched.text) == (status in (PageStatus.OK, PageStatus.THIN))


def test_fetch_page_honours_robots_without_fetching():
    robots = httpx2.Response(200, text="User-agent: *\nDisallow: /ca/\n")
    fake = FakeSite({"/robots.txt": robots})

    assert _fetch(fake, LISTING_URL).status is PageStatus.ROBOTS
    assert fake.paths() == ["/robots.txt"]


def test_fetch_page_waits_for_a_robots_txt_it_could_not_read():
    fake = FakeSite({"/robots.txt": httpx2.Response(500)})

    fetched = _fetch(fake, LISTING_URL)

    assert fetched.status is PageStatus.UNAVAILABLE
    assert fetched.detail == "unavailable"
    assert fake.paths() == ["/robots.txt"]


def test_fetch_page_reports_a_dead_host_as_unavailable():
    def refuse(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        raise httpx2.ConnectError("refused", request=request)

    async def main() -> pages.Fetched:
        transport = httpx2.MockTransport(refuse)
        async with open_client(POLICY, transport=transport, sleep=no_sleep) as client:
            return await pages.fetch_page(client, RobotsPolicy(client), LISTING_URL)

    assert asyncio.run(main()).status is PageStatus.UNAVAILABLE


# --- one site ------------------------------------------------------------------------------------


def test_site_with_a_plain_listing_is_accepted():
    fake, stub = FakeSite(), StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [LISTING_LINK, AGENDA_LINK]})
    stub.tool_reply(PASS2_TOOL, _verdict("current_listing").model_dump())

    result = _discover(SITE, fake, stub)

    assert result.outcome is SiteOutcome.ACCEPTED
    assert result.candidates == (
        Candidate(LISTING_URL, _verdict("current_listing"), SeedStatus.ACCEPTED),
    )
    # The agenda link 404s: reported, never classified.
    assert result.skipped == (Skipped(AGENDA_URL, "error (HTTP 404)"),)
    assert fake.paths() == ["/robots.txt", "/", "/ca/exposicions/", "/ca/agenda"]

    pass1, pass2 = (_prompt(request) for request in stub.converse_requests)
    assert f"[{LISTING_LINK}] (nav) Exposicions — {LISTING_URL}" in pass1
    assert "Segona mostra Col·lectiu Fictici 03.10.2026 – 10.01.2027" in pass2
    assert {call.purpose for call in stub.calls} == {"discover"}


def test_page_text_stays_off_the_spans():
    """Discovery reads third-party text, so its bodies never reach Langfuse (CLAUDE.md § 1)."""
    fake, stub = FakeSite(), StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [LISTING_LINK]})
    stub.tool_reply(PASS2_TOOL, _verdict("current_listing").model_dump())

    _discover(SITE, fake, stub)

    assert len(stub.spans) == 2
    assert "Col·lectiu Fictici" not in repr([dict(span.attributes) for span in stub.spans])


def test_pinned_site_is_not_discovered():
    fake, stub = FakeSite(), StubLlm()

    result = _discover(Site(1, VENUE, f"{HOST}/", pinned=(LISTING_URL,)), fake, stub)

    assert result.outcome is SiteOutcome.PINNED
    assert fake.requests == [] and stub.converse_requests == []


def test_rejected_url_is_never_shown_to_the_model():
    fake, stub = FakeSite(), StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": []})
    rejected = "https://exemple.cat/ca/agenda/"  # another spelling of AGENDA_URL

    result = _discover(Site(1, VENUE, f"{HOST}/", rejected=(rejected,)), fake, stub)

    [pass1] = (_prompt(request) for request in stub.converse_requests)
    assert "/ca/agenda" not in pass1
    assert f"[{AGENDA_LINK}] (nav) Artistes" in pass1  # renumbered: no gap to choose
    # The model chose nothing: a person decides, and nothing is stored.
    assert (result.outcome, result.candidates) == (SiteOutcome.AMBIGUOUS, ())


def test_a_different_proposal_never_replaces_the_seed_in_use():
    fake, stub = FakeSite(), StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [LISTING_LINK]})
    stub.tool_reply(PASS2_TOOL, _verdict("current_listing").model_dump())

    result = _discover(Site(1, VENUE, f"{HOST}/", seeds_in_use=(AGENDA_URL,)), fake, stub)

    assert result.outcome is SiteOutcome.AMBIGUOUS
    assert [c.status for c in result.candidates] == [SeedStatus.AMBIGUOUS]


def test_thin_candidate_is_skipped_without_a_model_call():
    fake = FakeSite({"/ca/exposicions/": html("<html><body><div id='app'></div></body></html>")})
    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [LISTING_LINK]})

    result = _discover(SITE, fake, stub)

    assert result.outcome is SiteOutcome.AMBIGUOUS
    assert result.skipped == (Skipped(LISTING_URL, "thin (HTTP 200)"),)
    assert len(stub.converse_requests) == 1


@pytest.mark.parametrize(
    ("pages_", "outcome", "detail"),
    [
        ({"/": html("no", 403)}, SiteOutcome.BLOCKED, "blocked (HTTP 403)"),
        ({"/": html("x", 503, **{"cf-mitigated": "challenge"})}, SiteOutcome.BLOCKED,
         "blocked (HTTP 503)"),
        ({"/robots.txt": httpx2.Response(200, text="User-agent: *\nDisallow: /\n")},
         SiteOutcome.ROBOTS, "robots"),
        ({"/robots.txt": httpx2.Response(503)}, SiteOutcome.UNAVAILABLE, "unavailable"),
        ({"/": html("gone", 404)}, SiteOutcome.ERROR, "error (HTTP 404)"),
        ({"/": html("<html><body><div id='app'></div></body></html>")}, SiteOutcome.THIN,
         "no links on the homepage"),
    ],
)  # fmt: skip
def test_unreadable_homepage_is_an_outcome_and_costs_nothing(pages_, outcome, detail):
    stub = StubLlm()

    result = _discover(SITE, FakeSite(pages_), stub)

    assert (result.outcome, result.detail) == (outcome, detail)
    assert result.candidates == () and stub.converse_requests == []


def test_model_that_keeps_inventing_links_is_an_error_not_a_guess():
    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [99]})
    stub.tool_reply(PASS1_TOOL, {"links": [99]})

    result = _discover(SITE, FakeSite(), stub)

    assert (result.outcome, result.detail) == (SiteOutcome.ERROR, seeds.INVALID_ANSWER)


def test_unclassifiable_candidate_is_skipped_and_the_rest_still_count():
    fake = FakeSite({"/ca/agenda": html(LISTING)})
    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [LISTING_LINK, AGENDA_LINK]})
    stub.tool_reply(PASS2_TOOL, None)
    stub.tool_reply(PASS2_TOOL, None)
    stub.tool_reply(PASS2_TOOL, _verdict("agenda").model_dump())

    result = _discover(SITE, fake, stub)

    assert result.skipped == (Skipped(LISTING_URL, seeds.INVALID_ANSWER),)
    assert [(c.url, c.status) for c in result.candidates] == [(AGENDA_URL, SeedStatus.ACCEPTED)]
    assert result.outcome is SiteOutcome.ACCEPTED


# --- the report ----------------------------------------------------------------------------------


def test_report_groups_sites_by_outcome_and_marks_each_candidate():
    results = [
        SiteResult(
            Site(2, "Sala Prova", "https://sala.test/", seeds_in_use=("https://sala.test/ara",)),
            SiteOutcome.AMBIGUOUS,
            (
                Candidate(
                    "https://sala.test/mostra",
                    _verdict("single_show", "medium", 1),
                    SeedStatus.AMBIGUOUS,
                ),
                Candidate("https://sala.test/botiga", _verdict("other", "high", 0),
                          SeedStatus.REJECTED),
            ),
            (Skipped("https://sala.test/agenda", "error (HTTP 404)"),),
        ),
        SiteResult(
            SITE,
            SiteOutcome.ACCEPTED,
            (Candidate(LISTING_URL, _verdict("current_listing"), SeedStatus.ACCEPTED),),
        ),
        SiteResult(Site(3, "Museu Mur", "https://mur.test/"), SiteOutcome.BLOCKED,
                   detail="blocked (HTTP 403)"),
        SiteResult(Site(4, "Espai Fix", "https://fix.test/", pinned=("https://fix.test/expo",)),
                   SiteOutcome.PINNED),
    ]  # fmt: skip

    assert seeds.report(results, ["Galeria Exmple"]) == [
        "sites       4   1 accepted   1 ambiguous   1 pinned   1 blocked",
        "accepted    Galeria Exemple",
        f"            + {LISTING_URL}   current_listing high   3 dated   ca",
        "ambiguous   Sala Prova",
        "            ? https://sala.test/mostra   single_show medium   1 dated   ca",
        "            - https://sala.test/botiga   other high   0 dated   ca",
        "            ! https://sala.test/agenda   error (HTTP 404)",
        "            in use https://sala.test/ara",
        "pinned      Espai Fix",
        "            pin https://fix.test/expo",
        "blocked     Museu Mur   blocked (HTTP 403)",
        "override    Galeria Exmple: no crawlable pilot venue has this name",
    ]


# --- schema --------------------------------------------------------------------------------------


def test_database_enums_are_the_model_output_enums():
    """`venue_seeds` CHECKs are spelled out in `db/models.py` and migration 0004. A value the
    model may answer with but the table refuses would fail at write time, after the spend."""
    assert get_args(discover.PageType) == models.SEED_PAGE_TYPES
    assert get_args(discover.Confidence) == models.SEED_CONFIDENCES
    assert get_args(discover.Language) == models.SEED_LANGUAGES
    assert tuple(SeedStatus) == models.SEED_STATUSES


def test_venue_seeds_has_nowhere_to_put_a_sentence():
    # Adding a text column here is a copyright decision (CLAUDE.md § 1), not a refactor.
    assert set(models.VenueSeed.__table__.columns.keys()) == {
        "id", "venue_id", "url", "page_type", "confidence", "dated_items", "language", "status",
        "model", "discovered_at", "created_at",
    }  # fmt: skip


# --- real database -------------------------------------------------------------------------------

MACBA_URL = "http://macba.test"

# Two pilots. MACBA is two terms on one URL, with its events on the term that is *not* the pilot.
SEED_ROWS = f"""
INSERT INTO venues (id, source_venue_id, name, slug, website_url, crawl_enabled, is_pilot)
OVERRIDING SYSTEM VALUE VALUES
  (1, 101, 'Galeria Exemple', 'galeria-exemple', '{HOST}/', true, true),
  (2, 102, 'MACBA', 'macba', '{MACBA_URL}', true, true),
  (3, 103, 'MACBA, Museu', 'macba-museu', '{MACBA_URL}', true, false),
  (4, 104, 'Sense Web', 'sense-web', NULL, false, true),
  (5, 105, 'No Pilot', 'no-pilot', 'https://nopilot.test/', true, false);
INSERT INTO event_snapshots (id, source_event_id, venue_id, title, source_url, web_url_ca)
OVERRIDING SYSTEM VALUE VALUES
  (1, 9001, 3, 'Mostra viva', 'https://graf.test/e/1', '{MACBA_URL}/ca/mostra-viva'),
  (2, 9002, 3, 'Mostra acabada', 'https://graf.test/e/2', NULL),
  (3, 9003, 3, 'Vespre únic', 'https://graf.test/e/3', NULL),
  (4, 9004, 5, 'D''un altre lloc', 'https://graf.test/e/4', NULL);
INSERT INTO event_occurrences (event_id, source_occurrence_id, starts_at, ends_at) VALUES
  (1, 1, now() - interval '10 days', now() + interval '10 days'),
  (2, 2, now() - interval '30 days', now() - interval '1 day'),
  (3, 3, now() + interval '2 days', NULL),
  (4, 4, now() - interval '1 day', now() + interval '1 day');
"""


@pytest.fixture(scope="module")
def seeds_db(fresh_database) -> Iterator[URL]:
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@pytest.fixture
def db(seeds_db) -> URL:
    run_sql(
        seeds_db,
        "TRUNCATE venue_seeds, venues, event_snapshots, event_occurrences, llm_calls CASCADE",
        *(s for s in SEED_ROWS.split(";") if s.strip()),
    )
    return seeds_db


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


def _load(url: URL, overrides: SeedOverrides | None = None) -> dict[str, Site]:
    sites = _in_session(url, lambda s: seeds.load_sites(s, overrides or SeedOverrides()))
    return {site.name: site for site in sites}


def _write(url: URL, results: list[SiteResult], *, commit: bool = True) -> int:
    return _in_session(url, lambda s: seeds.write_seeds(s, results, HAIKU), commit=commit)


def _result(url: str, page_type: str = "current_listing", confidence: str = "high") -> SiteResult:
    verdict = _verdict(page_type, confidence)
    return SiteResult(
        SITE, SiteOutcome.ACCEPTED, (Candidate(url, verdict, discover.route(verdict)),)
    )


SEED_STATE = "SELECT url, id, status, page_type, discovered_at, created_at FROM venue_seeds"


def _seed_state(url: URL) -> dict[str, tuple]:
    [rows] = run_sql(url, SEED_STATE)
    return {r[0]: r[1:] for r in rows}


def test_one_site_per_crawlable_pilot_url(db):
    sites = _load(db)

    # Not `Sense Web` (no URL), not `No Pilot`, and MACBA once, on the pilot's own row.
    assert {name: (s.venue_id, s.url) for name, s in sites.items()} == {
        VENUE: (1, f"{HOST}/"),
        "MACBA": (2, MACBA_URL),
    }


def test_hints_are_the_current_events_of_every_space_on_the_site(db):
    sites = _load(db)

    # Hung off the sibling term; the ended show and the other venue's event are left out.
    assert sites["MACBA"].hints == GrafHints(
        titles=("Mostra viva", "Vespre únic"), urls=(f"{MACBA_URL}/ca/mostra-viva",)
    )
    assert sites[VENUE].hints == GrafHints()


def test_written_seeds_come_back_as_seeds_in_use_unless_rejected_by_hand(db):
    assert _write(db, [_result(LISTING_URL), _result(AGENDA_URL, "other")]) == 2

    assert _load(db)[VENUE].seeds_in_use == (LISTING_URL,)  # the rejected one is not in use

    overrides = load_overrides(f"venues:\n  {VENUE}:\n    reject: {HOST}/ca/exposicions\n")
    site = _load(db, overrides)[VENUE]
    assert site.seeds_in_use == ()
    assert site.rejected == (f"{HOST}/ca/exposicions",)


def test_second_write_updates_the_same_row(db):
    _write(db, [_result(LISTING_URL)])
    before = _seed_state(db)

    # Another spelling of the same page, and a verdict that changed.
    _write(db, [_result(f"{HOST}/ca/exposicions", "past_archive")])
    after = _seed_state(db)

    assert after.keys() == before.keys() == {LISTING_URL}
    seed_id, status, page_type, discovered_at, created_at = after[LISTING_URL]
    assert (seed_id, created_at) == (before[LISTING_URL][0], before[LISTING_URL][4])
    assert (status, page_type) == ("rejected", "past_archive")
    assert discovered_at > before[LISTING_URL][3]


def test_rolled_back_write_leaves_nothing(db):
    assert _write(db, [_result(LISTING_URL)], commit=False) == 1
    assert _seed_state(db) == {}


def test_site_without_candidates_writes_nothing(db):
    assert _write(db, [SiteResult(SITE, SiteOutcome.BLOCKED, detail="blocked (HTTP 403)")]) == 0
    assert _seed_state(db) == {}


# --- cli discover --------------------------------------------------------------------------------


def _clear_caches() -> None:
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


@pytest.fixture
def run(db, monkeypatch) -> Iterator[Callable[..., Any]]:
    """`run(*args, site=FakeSite(), overrides="")` invokes `discover` against `db`. `run.stub` is
    the model stub: queue its answers first."""
    monkeypatch.setenv("DATABASE_URL", db.render_as_string(hide_password=False))
    for var in ("EXTRACT_MODEL", "LANGFUSE_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    _clear_caches()
    stub = StubLlm(record=db_recorder(get_sessionmaker()))
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)

    def _run(*args: str, site: FakeSite | None = None, overrides: str = ""):
        site = site or FakeSite()
        monkeypatch.setattr(
            cli,
            "open_client",
            functools.partial(open_client, POLICY, transport=site.transport, sleep=no_sleep),
        )
        monkeypatch.setattr(seeds, "load_overrides", lambda: load_overrides(overrides))
        return CliRunner().invoke(cli.app, ["discover", *args])

    _run.stub = stub
    yield _run
    _clear_caches()


def _queue_one_accepted_site(stub: StubLlm) -> None:
    stub.tool_reply(PASS1_TOOL, {"links": [LISTING_LINK]}, input_tokens=5000, output_tokens=20)
    stub.tool_reply(
        PASS2_TOOL, _verdict("current_listing").model_dump(), input_tokens=4000, output_tokens=30
    )


def test_discover_writes_seeds_and_reports_cost(run, db):
    _queue_one_accepted_site(run.stub)

    result = run("--venue", "galeria exemple")

    assert result.exit_code == 0, result.output
    assert "Discover\n" in result.output
    assert "accepted    Galeria Exemple" in result.output
    assert f"+ {LISTING_URL}   current_listing high   3 dated   ca" in result.output
    assert "written     1 seeds" in result.output
    # 9000 in and 50 out on Haiku 4.5 ($1 / $5 per million).
    assert "cost        $0.009250   2 model calls" in result.output
    assert {u: s[1] for u, s in _seed_state(db).items()} == {LISTING_URL: "accepted"}
    [[(model,)]] = run_sql(db, "SELECT DISTINCT model FROM venue_seeds")
    assert model == HAIKU


def test_discover_runs_every_site_and_one_bad_site_does_not_stop_the_rest(run, db):
    # MACBA's host serves nothing here, so its homepage 404s; Galeria Exemple follows.
    _queue_one_accepted_site(run.stub)

    result = run()

    assert result.exit_code == 0, result.output
    assert "sites       2   1 accepted   1 error" in result.output
    assert "error       MACBA   error (HTTP 404)" in result.output


def test_dry_run_reports_and_keeps_only_the_spend(run, db):
    _queue_one_accepted_site(run.stub)

    result = run("--dry-run", "--venue", VENUE)

    assert result.exit_code == 0, result.output
    assert "Discover (dry run, rolled back)" in result.output
    assert "written     1 seeds" in result.output
    assert _seed_state(db) == {}
    [[(calls,)]] = run_sql(db, "SELECT count(*) FROM llm_calls WHERE purpose = 'discover'")
    assert calls == 2  # the calls were made, so they stay recorded


def test_pinned_venue_makes_no_calls(run, db):
    overrides = f"venues:\n  {VENUE}:\n    pin: {LISTING_URL}\n  Galeria Exmple: {{}}\n"

    result = run("--venue", VENUE, overrides=overrides)

    assert result.exit_code == 0, result.output
    assert f"pinned      {VENUE}" in result.output
    assert f"pin {LISTING_URL}" in result.output
    assert "override    Galeria Exmple: no crawlable pilot venue has this name" in result.output
    assert "cost        $0   0 model calls" in result.output


def test_unknown_venue_fails_and_names_the_known_ones(run):
    result = run("--venue", "Galeria Exmple")

    assert result.exit_code == 1
    assert "Known: Galeria Exemple, MACBA" in result.output


def test_no_pilots_fails_before_spending(run, db):
    run_sql(db, "UPDATE venues SET is_pilot = false")

    result = run()

    assert result.exit_code == 1
    assert "Run sync-graf first" in result.output


def test_invalid_overrides_fail_before_spending(run):
    site = FakeSite()

    result = run(site=site, overrides="venues:\n  MACBA:\n    pin: not-a-url\n")

    assert result.exit_code == 1
    assert "seeds.yaml is invalid" in result.output
    assert site.requests == []
