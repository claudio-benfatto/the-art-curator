"""`ingest/crawl.py`, `cli crawl` and `cli purge-pages`: which pages a site is crawled at, one site
over a fake website, the `venue_pages` write, the 7-day purge, and the report.

No venue site is fetched and no model exists on this path: `FakeSite` fakes the wire only.
"""

import asyncio
import functools
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import pytest
from alembic import command
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from typer.testing import CliRunner

from art_curator import cli
from art_curator.config import get_settings
from art_curator.db import models
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.ingest import crawl, pages, seeds
from art_curator.ingest.crawl import Broken, Change, CrawledPage, SiteCrawl, Target
from art_curator.ingest.http import RobotsPolicy, open_client
from art_curator.ingest.pages import PageStatus
from art_curator.ingest.seeds import Site, load_overrides
from tests.db import alembic_config, run_sql
from tests.site_stub import HOST, LISTING, POLICY, FakeSite, html, no_sleep

VENUE = "Galeria Exemple"
LISTING_URL = f"{HOST}/ca/exposicions/"
AGENDA_URL = f"{HOST}/ca/agenda"
LISTING_TEXT = pages.page_text(LISTING)
LISTING_HASH = crawl.text_hash(LISTING_TEXT)
LISTING_WORDS = len(LISTING_TEXT.split())
THIN_PAGE = "<html><body><div id='app'></div><p>Carregant</p></body></html>"

SITE = Site(venue_id=1, name=VENUE, url=f"{HOST}/")


def _crawl(target: Target, fake: FakeSite) -> SiteCrawl:
    async def main() -> SiteCrawl:
        async with open_client(POLICY, transport=fake.transport, sleep=no_sleep) as client:
            return await crawl.crawl_site(client, RobotsPolicy(client), target)

    return asyncio.run(main())


def _page(target: Target, fake: FakeSite) -> CrawledPage:
    [page] = _crawl(target, fake).pages
    return page


# --- one site ------------------------------------------------------------------------------------


def test_ok_page_carries_its_text_and_the_hash_of_that_text():
    page = _page(Target(SITE, (LISTING_URL,)), FakeSite())

    assert (page.url, page.status, page.http_status) == (LISTING_URL, PageStatus.OK, 200)
    assert page.text == LISTING_TEXT
    assert (page.content_hash, page.words) == (LISTING_HASH, LISTING_WORDS)
    assert (page.change, page.broken) == (Change.NEW, None)


def test_page_text_stays_out_of_repr():
    """A `CrawledPage` in a log line or a failed assertion must not print the page."""
    page = _page(Target(SITE, (LISTING_URL,)), FakeSite())

    assert "Col·lectiu Fictici" in page.text
    assert "Col·lectiu Fictici" not in repr(page)


@pytest.mark.parametrize(
    ("stored", "change"),
    [
        ({}, Change.NEW),
        ({LISTING_URL: None}, Change.NEW),  # a row from a fetch that was not `ok`
        ({LISTING_URL: LISTING_HASH}, Change.UNCHANGED),
        ({LISTING_URL: "0" * 64}, Change.CHANGED),
    ],
)
def test_change_is_the_text_hash_against_the_stored_one(stored, change):
    assert _page(Target(SITE, (LISTING_URL,), hashes=stored), FakeSite()).change is change


def test_hash_ignores_markup_that_varies_between_fetches():
    """HTML carries nonces; the hash is of the text, so two fetches of one page agree."""
    nonce = LISTING.replace(b"<script>", b'<script nonce="a1b2c3">')
    assert nonce != LISTING

    page = _page(Target(SITE, (LISTING_URL,)), FakeSite({"/ca/exposicions/": html(nonce)}))

    assert page.content_hash == LISTING_HASH


@pytest.mark.parametrize(
    ("response", "status", "detail"),
    [
        (html(THIN_PAGE), PageStatus.THIN, "1 words"),
        (html("no", 403), PageStatus.BLOCKED, "HTTP 403"),
        (html(LISTING, 503, **{"cf-mitigated": "challenge"}), PageStatus.BLOCKED, "HTTP 503"),
        (html("oops", 500), PageStatus.ERROR, "HTTP 500"),
    ],
)
def test_unusable_page_is_a_status_with_no_text(response, status, detail):
    page = _page(Target(SITE, (LISTING_URL,)), FakeSite({"/ca/exposicions/": response}))

    assert (page.status, page.detail) == (status, detail)
    assert (page.text, page.content_hash, page.change, page.broken) == ("", None, None, None)


def test_robots_and_unavailable_are_statuses_too():
    disallow = FakeSite({"/robots.txt": httpx2.Response(200, text="User-agent: *\nDisallow: /\n")})
    down = FakeSite({"/robots.txt": httpx2.Response(503)})

    assert _page(Target(SITE, (LISTING_URL,)), disallow).status is PageStatus.ROBOTS
    assert _page(Target(SITE, (LISTING_URL,)), down).status is PageStatus.UNAVAILABLE
    assert disallow.paths() == ["/robots.txt"]


@pytest.mark.parametrize("status", [404, 410])
def test_listing_that_is_gone_is_a_broken_seed(status):
    page = _page(Target(SITE, (AGENDA_URL,)), FakeSite({"/ca/agenda": html("gone", status)}))

    assert (page.status, page.broken) == (PageStatus.ERROR, Broken.GONE)


def test_listing_that_redirects_to_the_homepage_is_a_broken_seed():
    redirect = httpx2.Response(301, headers={"location": "https://exemple.cat/"})

    page = _page(Target(SITE, (AGENDA_URL,)), FakeSite({"/ca/agenda": redirect}))

    # The homepage itself fetched fine: the status is about the page, `broken` about the seed.
    assert (page.url, page.broken) == (AGENDA_URL, Broken.HOMEPAGE)
    assert page.status in (PageStatus.OK, PageStatus.THIN)


def test_homepage_pinned_as_the_listing_is_not_broken():
    assert _page(Target(SITE, (f"{HOST}/",)), FakeSite()).broken is None


def test_detail_page_that_is_gone_is_not_a_broken_seed():
    target = Target(SITE, (LISTING_URL,), details=(f"{HOST}/ca/exposicions/prova-u",))

    _, detail = _crawl(target, FakeSite()).pages

    assert (detail.status, detail.http_status, detail.broken) == (PageStatus.ERROR, 404, None)


def test_site_is_capped_at_eight_pages_listings_first():
    details = tuple(f"{HOST}/ca/exposicions/mostra-{n}" for n in range(10))
    target = Target(SITE, (LISTING_URL, AGENDA_URL), details=details)
    fake = FakeSite()

    result = _crawl(target, fake)

    assert [page.url for page in result.pages] == [LISTING_URL, AGENDA_URL, *details[:6]]
    assert len(fake.paths()) == 1 + crawl.MAX_PAGES_PER_SITE  # robots.txt once, then the pages


# --- the report ----------------------------------------------------------------------------------


def test_report_lists_each_site_with_its_pages_and_flags_broken_seeds():
    crawls = [
        SiteCrawl(Site(3, "Museu Mur", "https://mur.test/")),
        SiteCrawl(
            Site(2, "Sala Prova", "https://sala.test/"),
            (
                CrawledPage("https://sala.test/ara", PageStatus.THIN, 200, 12),
                CrawledPage("https://sala.test/agenda", PageStatus.ERROR, 404, broken=Broken.GONE),
                CrawledPage("https://sala.test/x", PageStatus.ROBOTS),
            ),
        ),
        SiteCrawl(
            SITE,
            (
                CrawledPage(LISTING_URL, PageStatus.OK, 200, 87, "a" * 64, "t", Change.UNCHANGED),
                CrawledPage(AGENDA_URL, PageStatus.OK, 200, 140, "b" * 64, "t", Change.CHANGED),
            ),
        ),
    ]

    assert crawl.report(crawls) == [
        "sites       3   2 crawled   1 without a seed",
        "pages       5   2 ok   1 thin   1 robots   1 error",
        "text        0 new   1 changed   1 unchanged",
        "site        Galeria Exemple",
        f"ok          {LISTING_URL}   87 words   unchanged",
        f"ok          {AGENDA_URL}   140 words   changed",
        "site        Sala Prova",
        "thin        https://sala.test/ara   12 words",
        "error       https://sala.test/agenda   HTTP 404",
        "broken      https://sala.test/agenda   the page is gone: re-run discover, or fix its pin "
        "in seeds.yaml",
        "robots      https://sala.test/x",
        "no seed     Museu Mur   run discover, or pin its listing in seeds.yaml",
    ]


# --- schema --------------------------------------------------------------------------------------


def test_database_statuses_are_the_page_statuses_that_get_written():
    """`unavailable` is the one status never stored; the CHECK must allow every other."""
    assert set(models.PAGE_STATUSES) == set(PageStatus) - {PageStatus.UNAVAILABLE}


# --- real database -------------------------------------------------------------------------------

MACBA_URL = "http://macba.test"

VENUE_ROWS = f"""
INSERT INTO venues (id, source_venue_id, name, slug, website_url, crawl_enabled, is_pilot)
OVERRIDING SYSTEM VALUE VALUES
  (1, 101, 'Galeria Exemple', 'galeria-exemple', '{HOST}/', true, true),
  (2, 102, 'MACBA', 'macba', '{MACBA_URL}', true, true)
"""

PAGE_STATE = (
    "SELECT url, id, venue_id, status, http_status, content_hash, raw_text FROM venue_pages"
)


@pytest.fixture(scope="module")
def crawl_db(fresh_database) -> Iterator[URL]:
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@pytest.fixture
def db(crawl_db) -> URL:
    run_sql(crawl_db, "TRUNCATE venue_pages, venue_seeds, venues CASCADE", VENUE_ROWS)
    return crawl_db


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


def _targets(url: URL, overrides: str = "") -> dict[str, Target]:
    found = _in_session(url, lambda s: crawl.load_targets(s, load_overrides(overrides)))
    return {target.site.name: target for target in found}


def _write(url: URL, crawls: list[SiteCrawl], *, commit: bool = True) -> int:
    return _in_session(url, lambda s: crawl.write_pages(s, crawls), commit=commit)


def _page_state(url: URL) -> dict[str, tuple]:
    [rows] = run_sql(url, PAGE_STATE)
    return {r[0]: r[1:] for r in rows}


def _accept(url: URL, *seed_urls: str, status: str = "accepted", venue_id: int = 1) -> None:
    run_sql(
        url,
        *(
            "INSERT INTO venue_seeds (venue_id, url, page_type, confidence, dated_items, "
            f"language, status, model) VALUES ({venue_id}, '{seed}', 'current_listing', 'high', "
            f"3, 'ca', '{status}', 'm')"
            for seed in seed_urls
        ),
    )


def _crawl_and_write(url: URL, fake: FakeSite, overrides: str = "") -> SiteCrawl:
    result = _crawl(_targets(url, overrides)[VENUE], fake)
    _write(url, [result])
    return result


def test_site_is_crawled_at_its_accepted_seeds(db):
    _accept(db, LISTING_URL)
    _accept(db, AGENDA_URL, status="rejected")
    _accept(db, f"{HOST}/ca/arxiu", status="ambiguous")

    targets = _targets(db)

    assert targets[VENUE].listings == (LISTING_URL,)
    assert targets["MACBA"].listings == ()  # no seed yet: still a target, so the report names it


def test_pin_replaces_the_accepted_seeds_and_a_rejected_url_is_dropped(db):
    _accept(db, LISTING_URL, AGENDA_URL)

    pinned = _targets(db, f"venues:\n  {VENUE}:\n    pin: {HOST}/ca/ara\n")[VENUE]
    rejected = _targets(db, f"venues:\n  {VENUE}:\n    reject: {HOST}/ca/exposicions\n")[VENUE]

    assert pinned.listings == (f"{HOST}/ca/ara",)
    assert rejected.listings == (AGENDA_URL,)


def test_page_is_stored_with_its_text_and_hash(db):
    _accept(db, LISTING_URL)

    _crawl_and_write(db, FakeSite())

    [(url, (_, venue_id, status, http_status, content_hash, raw_text))] = _page_state(db).items()
    assert (url, venue_id, status, http_status) == (LISTING_URL, 1, "ok", 200)
    assert (content_hash, raw_text) == (LISTING_HASH, LISTING_TEXT)


def test_second_crawl_of_an_unchanged_page_changes_no_hash_or_id(db):
    _accept(db, LISTING_URL)
    first = _crawl_and_write(db, FakeSite())
    before = _page_state(db)

    second = _crawl_and_write(db, FakeSite())

    assert [page.change for page in first.pages] == [Change.NEW]
    assert [page.change for page in second.pages] == [Change.UNCHANGED]
    assert _page_state(db) == before


def test_changed_page_keeps_its_row_and_takes_the_new_hash(db):
    _accept(db, LISTING_URL)
    _crawl_and_write(db, FakeSite())
    before = _page_state(db)[LISTING_URL]
    edited = LISTING.replace(b"Tercera", b"Quarta")

    result = _crawl_and_write(db, FakeSite({"/ca/exposicions/": html(edited)}))

    after = _page_state(db)[LISTING_URL]
    assert [page.change for page in result.pages] == [Change.CHANGED]
    assert after[0] == before[0]
    assert after[4] == crawl.text_hash(pages.page_text(edited)) != before[4]


def test_another_spelling_of_a_stored_url_updates_the_same_row(db):
    _accept(db, LISTING_URL)
    _crawl_and_write(db, FakeSite())

    # The pin drops the trailing slash; the row keeps the spelling it was stored under.
    result = _crawl_and_write(
        db, FakeSite(), f"venues:\n  {VENUE}:\n    pin: {HOST}/ca/exposicions\n"
    )

    assert [(page.url, page.change) for page in result.pages] == [(LISTING_URL, Change.UNCHANGED)]
    assert _page_state(db).keys() == {LISTING_URL}


def test_page_that_stops_working_loses_its_text_and_hash(db):
    _accept(db, LISTING_URL)
    _crawl_and_write(db, FakeSite())

    _crawl_and_write(db, FakeSite({"/ca/exposicions/": html(THIN_PAGE)}))

    assert _page_state(db)[LISTING_URL][2:] == ("thin", 200, None, None)


def test_unavailable_site_keeps_what_the_last_crawl_stored(db):
    _accept(db, LISTING_URL)
    _crawl_and_write(db, FakeSite())
    before = _page_state(db)

    result = _crawl(_targets(db)[VENUE], FakeSite({"/robots.txt": httpx2.Response(503)}))

    assert [page.status for page in result.pages] == [PageStatus.UNAVAILABLE]
    assert _write(db, [result]) == 0
    assert _page_state(db) == before


def test_rolled_back_write_leaves_nothing(db):
    result = _crawl(Target(SITE, (LISTING_URL,)), FakeSite())

    assert _write(db, [result], commit=False) == 1
    assert _page_state(db) == {}


# --- the purge -----------------------------------------------------------------------------------

OLD_AND_NEW_PAGES = f"""
INSERT INTO venue_pages (venue_id, url, status, http_status, content_hash, raw_text, fetched_at)
VALUES
  (1, '{HOST}/vella', 'ok', 200, 'h-vella', 'text vell', now() - interval '7 days 1 minute'),
  (1, '{HOST}/recent', 'ok', 200, 'h-recent', 'text recent', now() - interval '6 days 23 hours'),
  (1, '{HOST}/buida', 'thin', 200, NULL, NULL, now() - interval '30 days')
"""
TEXT_PAST_TTL = (
    "SELECT count(*) FROM venue_pages "
    "WHERE raw_text IS NOT NULL AND fetched_at < now() - interval '7 days'"
)


def _texts(url: URL) -> dict[str, tuple]:
    [rows] = run_sql(url, "SELECT url, content_hash, raw_text FROM venue_pages")
    return {r[0].removeprefix(HOST): r[1:] for r in rows}


def test_purge_drops_text_past_seven_days_and_keeps_the_row_and_hash(db):
    run_sql(db, OLD_AND_NEW_PAGES)

    assert _in_session(db, crawl.purge_pages) == 1

    assert _texts(db) == {
        "/vella": ("h-vella", None),  # the hash still lets extraction skip an unchanged page
        "/recent": ("h-recent", "text recent"),
        "/buida": (None, None),
    }
    assert run_sql(db, TEXT_PAST_TTL) == [[(0,)]]
    assert _in_session(db, crawl.purge_pages) == 0


# --- cli -----------------------------------------------------------------------------------------


def _clear_caches() -> None:
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


@pytest.fixture
def run(db, monkeypatch) -> Iterator[Callable[..., Any]]:
    """`run(*args, site=FakeSite(), overrides="")` invokes the CLI against `db`."""
    monkeypatch.setenv("DATABASE_URL", db.render_as_string(hide_password=False))
    _clear_caches()

    def _run(*args: str, site: FakeSite | None = None, overrides: str = ""):
        site = site or FakeSite()
        monkeypatch.setattr(
            cli,
            "open_client",
            functools.partial(open_client, POLICY, transport=site.transport, sleep=no_sleep),
        )
        monkeypatch.setattr(seeds, "load_overrides", lambda: load_overrides(overrides))
        return CliRunner().invoke(cli.app, list(args))

    yield _run
    _clear_caches()


def test_crawl_purges_then_stores_pages_and_reports(run, db):
    _accept(db, LISTING_URL)
    run_sql(db, OLD_AND_NEW_PAGES)

    result = run("crawl", "--pilot")

    assert result.exit_code == 0, result.output
    assert "Crawl\n" in result.output
    assert "purged      1 pages of text older than 7 days" in result.output
    assert "sites       2   1 crawled   1 without a seed" in result.output
    assert f"ok          {LISTING_URL}   {LISTING_WORDS} words   new" in result.output
    assert "no seed     MACBA" in result.output
    assert "written     1 pages" in result.output
    assert _page_state(db)[LISTING_URL][2:] == ("ok", 200, LISTING_HASH, LISTING_TEXT)
    assert run_sql(db, TEXT_PAST_TTL) == [[(0,)]]


def test_crawl_prints_no_page_text(run, db):
    _accept(db, LISTING_URL)

    result = run("crawl", "--pilot")

    assert "Fictici" not in result.output


def test_second_crawl_reports_unchanged(run, db):
    _accept(db, LISTING_URL)
    run("crawl", "--pilot")
    before = _page_state(db)

    result = run("crawl", "--pilot", "--venue", "galeria exemple")

    assert result.exit_code == 0, result.output
    assert "text        0 new   0 changed   1 unchanged" in result.output
    assert _page_state(db) == before


def test_dry_run_rolls_back_the_pages_but_not_the_purge(run, db):
    _accept(db, LISTING_URL)
    run_sql(db, OLD_AND_NEW_PAGES)

    result = run("crawl", "--pilot", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "Crawl (dry run, rolled back)" in result.output
    assert "written     1 pages" in result.output
    assert LISTING_URL not in _page_state(db)
    assert _texts(db)["/vella"] == ("h-vella", None)


def test_broken_seed_is_stored_reported_and_exits_1(run, db):
    result = run("crawl", "--pilot", overrides=f"venues:\n  {VENUE}:\n    pin: {AGENDA_URL}\n")

    assert result.exit_code == 1
    assert f"broken      {AGENDA_URL}   the page is gone" in result.output
    assert _page_state(db)[AGENDA_URL][2:] == ("error", 404, None, None)


def test_crawl_without_a_scope_fetches_nothing(run):
    site = FakeSite()

    result = run("crawl", site=site)

    assert result.exit_code == 1
    assert "Pass --pilot" in result.output
    assert site.requests == []


def test_unknown_venue_fails_and_names_the_known_ones(run):
    result = run("crawl", "--pilot", "--venue", "Galeria Exmple")

    assert result.exit_code == 1
    assert "Known: Galeria Exemple, MACBA" in result.output


def test_no_pilots_fails(run, db):
    run_sql(db, "UPDATE venues SET is_pilot = false")

    result = run("crawl", "--pilot")

    assert result.exit_code == 1
    assert "Run sync-graf first" in result.output


def test_purge_pages_command(run, db):
    run_sql(db, OLD_AND_NEW_PAGES)

    result = run("purge-pages")

    assert result.exit_code == 0, result.output
    assert result.output == "Purged 1 pages of text older than 7 days\n"
    assert _texts(db)["/recent"] == ("h-recent", "text recent")
