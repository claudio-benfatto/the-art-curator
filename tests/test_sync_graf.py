"""Writing a GRAF fetch: `plan_sync` (pure), `write_graf` (real database), `report` (pure), and
`cli sync-graf` end to end over a fake GRAF (`tests/graf_stub.py`).

The recorded fixtures already hold the cases that matter, so the tests name real rows:

- MACBA is one profile (24) and two terms here: `MACBA` (159, slug round) and `MACBA, Museu d'Art
  Contemporani de Barcelona` (487, containment) — the many-to-one join migration 0002 allows.
- `Chiquita Room` (472) is a pilot, joined by name, with two events.
- 486, 251 and 752 sit at `0,0` and must have no geometry.
- Event 57221 names term 594, which is not in the fixture.

Variants are built by editing the raw JSON and re-parsing, so `write_graf` never sees a shape GRAF
cannot produce.
"""

import asyncio
import copy
import functools
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from typer.testing import CliRunner

from art_curator import cli
from art_curator.config import get_settings
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.ingest.graf import banned_keys, parse_events, parse_profiles, parse_terms
from art_curator.ingest.http import open_client
from art_curator.ingest.sync import (
    GrafSnapshot,
    SyncResult,
    VenueRow,
    plan_sync,
    report,
    write_graf,
)
from tests.db import alembic_config, run_sql
from tests.graf_stub import BASE_URL, POLICY, FakeGraf, full_corpus, no_sleep

FIXTURES = Path(__file__).parent / "fixtures" / "graf"

MACBA_PROFILE = 24
MACBA_TERMS = (159, 487)
CHIQUITA = 472
NULL_ISLAND = {486, 251, 752}
UNKNOWN_TERM_EVENT = 57221


def raw(name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURES / name).read_text())


def snapshot(
    venues: Callable[[list[dict]], list[dict]] = lambda rows: rows,
    profiles: Callable[[list[dict]], list[dict]] = lambda rows: rows,
    events: Callable[[list[dict]], list[dict]] = lambda rows: rows,
) -> GrafSnapshot:
    """The recorded fixture, optionally edited. Each callable gets a deep copy of the raw rows."""
    return GrafSnapshot(
        terms=parse_terms(venues(copy.deepcopy(raw("venues.json")))),
        profiles=parse_profiles(profiles(copy.deepcopy(raw("profiles.json")))),
        events=parse_events(events(copy.deepcopy(raw("events.json")))),
    )


def plan(snap: GrafSnapshot) -> list[VenueRow]:
    return plan_sync(snap).venues


def without(key: str, value: int) -> Callable[[list[dict]], list[dict]]:
    return lambda rows: [r for r in rows if r[key] != value]


def edited(row_id: int, **changes: Any) -> Callable[[list[dict]], list[dict]]:
    return lambda rows: [{**r, **changes} if r["id"] == row_id else r for r in rows]


# --- plan_sync: pure ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rows() -> dict[int, VenueRow]:
    return {r.source_venue_id: r for r in plan(snapshot())}


def test_one_row_per_term(rows):
    assert set(rows) == {487, 159, 472, 601, 895, 486, 251, 752}


def test_one_profile_serves_several_terms(rows):
    for term_id in MACBA_TERMS:
        assert rows[term_id].source_profile_id == MACBA_PROFILE
        assert rows[term_id].website_url == "http://macba.cat"
        assert rows[term_id].crawl_enabled


def test_pilots_are_flagged(rows):
    # Only two pilots have a term in the trimmed fixture; the full list is test_matching's job.
    assert {t for t, r in rows.items() if r.is_pilot} == {159, CHIQUITA}


def test_unjoined_term_has_no_url(rows):
    lb = rows[601]  # L&B Gallery: its profile is not in the trimmed fixture
    assert (lb.source_profile_id, lb.website_url, lb.crawl_enabled) == (None, None, False)


def test_null_island_has_no_geometry(rows):
    assert {t for t, r in rows.items() if r.geom is None} == NULL_ISLAND
    assert rows[159].geom == "SRID=4326;POINT(2.16694 41.383152)"


@pytest.mark.parametrize(
    ("name", "slug", "profile_id", "website", "instagram"),
    [
        # Profile 173's only URL is Instagram: joined, facts only.
        ("AGUAS", "aguas", 173, None, "https://www.instagram.com/aguasssssss"),
        # Profile 118 has `url: ""`: joined, nothing to crawl.
        ("ProjecteSD", "projectesd", 118, None, None),
    ],
)
def test_joined_but_not_crawlable(name, slug, profile_id, website, instagram):
    term = {"id": 9001, "name": name, "slug": slug, "latitude": "41.4", "longtitude": "2.1"}
    [row] = [r for r in plan(snapshot(venues=lambda rs: [*rs, term])) if r.source_venue_id == 9001]
    assert (row.source_profile_id, row.website_url, row.instagram_url) == (
        profile_id,
        website,
        instagram,
    )
    assert not row.crawl_enabled


def test_repeated_term_collapses_to_the_last():
    # Pagination over a list that changes mid-fetch can serve a term twice; one ON CONFLICT
    # statement cannot touch the same row twice.
    def repeat(rs):
        return [*rs, {**next(r for r in rs if r["id"] == 895), "address": "Later, 1"}]

    planned = [r for r in plan(snapshot(venues=repeat)) if r.source_venue_id == 895]
    assert [r.address for r in planned] == ["Later, 1"]


# --- write_graf: real database -------------------------------------------------------------------


@pytest.fixture(scope="module")
def sync_db(fresh_database) -> Iterator[URL]:
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@pytest.fixture
def db(sync_db) -> URL:
    run_sql(sync_db, "TRUNCATE venues, event_snapshots, event_occurrences CASCADE")
    return sync_db


def sync(url: URL, snap: GrafSnapshot | None = None, *, commit: bool = True) -> SyncResult:
    snap = snap or snapshot()

    async def _run() -> SyncResult:
        engine = create_async_engine(url)
        try:
            async with AsyncSession(engine) as session:
                result = await write_graf(session, snap, plan(snap))
                await (session.commit() if commit else session.rollback())
                return result
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def query(url: URL, sql: str) -> list[tuple]:
    [rows] = run_sql(url, sql)
    return rows


def by_key(url: URL, sql: str) -> dict:
    """`{first column: rest of the row}`."""
    return {r[0]: r[1:] for r in query(url, sql)}


VENUE_STATE = "SELECT source_venue_id, id, created_at, updated_at FROM venues"
EVENT_STATE = "SELECT source_event_id, id, first_seen_at, last_seen_at FROM event_snapshots"
OCCURRENCE_STATE = (
    "SELECT source_occurrence_id, id, event_id, first_seen_at, last_seen_at FROM event_occurrences"
)


def test_first_sync_writes_the_facts(db):
    result = sync(db)

    assert (result.venues, result.venues_inserted, result.venues_updated) == (8, 8, 0)
    assert (result.events, result.occurrences) == (4, 4)
    assert result.joins_before == {}
    assert result.joins_after == {159: MACBA_PROFILE, 487: MACBA_PROFILE, CHIQUITA: 83}

    # Many terms, one profile: what migration 0002 exists to allow.
    assert {
        r[0] for r in query(db, "SELECT source_venue_id FROM venues WHERE source_profile_id = 24")
    } == set(MACBA_TERMS)
    assert {r[0] for r in query(db, "SELECT source_venue_id FROM venues WHERE geom IS NULL")} == (
        NULL_ISLAND
    )
    assert {r[0] for r in query(db, "SELECT source_venue_id FROM venues WHERE is_pilot")} == {
        159,
        CHIQUITA,
    }


def test_event_at_an_unknown_venue_is_kept_without_one(db):
    result = sync(db)

    assert result.unknown_venue_events == (UNKNOWN_TERM_EVENT,)
    assert query(
        db, f"SELECT venue_id FROM event_snapshots WHERE source_event_id = {UNKNOWN_TERM_EVENT}"
    ) == [(None,)]
    # The other three resolve to the surrogate id of the term they name.
    assert by_key(
        db,
        "SELECT e.source_event_id, v.source_venue_id FROM event_snapshots e "
        "JOIN venues v ON v.id = e.venue_id",
    ) == {60390: (895,), 59179: (CHIQUITA,), 59371: (CHIQUITA,)}


def test_second_run_changes_nothing_but_last_seen(db):
    sync(db)
    venues, events, occurrences = (
        by_key(db, VENUE_STATE),
        by_key(db, EVENT_STATE),
        by_key(db, OCCURRENCE_STATE),
    )

    result = sync(db)

    assert (result.venues_inserted, result.venues_updated, result.venues_unjoined) == (0, 0, 0)
    assert result.joins_before == result.joins_after
    # Venues: same ids, and `updated_at` untouched because nothing changed.
    assert by_key(db, VENUE_STATE) == venues
    # Events and occurrences: same ids and first_seen_at, later last_seen_at.
    for before, after in [
        (events, by_key(db, EVENT_STATE)),
        (occurrences, by_key(db, OCCURRENCE_STATE)),
    ]:
        assert after.keys() == before.keys()
        for key, (*same, last_seen) in after.items():
            assert same == list(before[key][:-1])
            assert last_seen > before[key][-1]


def test_one_run_is_one_timestamp(db):
    # `now()` is fixed at transaction start: "seen in this run" is an equality, not a window.
    sync(db)
    assert query(
        db,
        "SELECT count(DISTINCT last_seen_at) FROM ("
        " SELECT last_seen_at FROM event_snapshots UNION ALL"
        " SELECT last_seen_at FROM event_occurrences) t",
    ) == [(1,)]


def test_event_leaving_the_window_is_kept_and_frozen(db):
    sync(db)
    first = by_key(db, EVENT_STATE)
    first_occ = by_key(db, OCCURRENCE_STATE)

    sync(db, snapshot(events=without("id", 60390)))
    second = by_key(db, EVENT_STATE)
    assert second.keys() == first.keys()  # nothing deleted
    assert second[60390] == first[60390]  # last_seen_at frozen
    assert by_key(db, OCCURRENCE_STATE)[23161] == first_occ[23161]
    assert second[59179][-1] > first[59179][-1]  # the rest moved on

    sync(db)
    third = by_key(db, EVENT_STATE)
    assert third[60390][:-1] == first[60390][:-1]  # same id and first_seen_at
    assert third[60390][-1] > second[60390][-1]


def test_vanished_term_is_unjoined_not_deleted(db):
    sync(db)
    chiquita_id = by_key(db, VENUE_STATE)[CHIQUITA][0]

    result = sync(db, snapshot(venues=without("id", CHIQUITA)))

    assert result.venues_unjoined == 1
    assert CHIQUITA in result.joins_before
    assert CHIQUITA not in result.joins_after
    assert query(
        db,
        "SELECT id, name, source_profile_id, website_url, crawl_enabled, is_pilot "
        f"FROM venues WHERE source_venue_id = {CHIQUITA}",
    ) == [(chiquita_id, "Chiquita Room", None, None, False, False)]
    # Its events still resolve: the venue map is read back from the table, not from this fetch.
    assert query(
        db, "SELECT DISTINCT venue_id FROM event_snapshots WHERE source_event_id IN (59179, 59371)"
    ) == [(chiquita_id,)]

    # A third run without it again finds nothing left to unjoin.
    assert sync(db, snapshot(venues=without("id", CHIQUITA))).venues_unjoined == 0


def test_changed_venue_bumps_only_its_own_updated_at(db):
    sync(db)
    before = by_key(db, VENUE_STATE)

    result = sync(db, snapshot(venues=edited(895, address="Carrer Nou, 1")))

    assert (result.venues_inserted, result.venues_updated) == (0, 1)
    after = by_key(db, VENUE_STATE)
    assert [t for t in after if after[t] != before[t]] == [895]
    assert after[895][:2] == before[895][:2]  # same id and created_at


def test_profile_url_change_reaches_every_space(db):
    sync(db)
    result = sync(db, snapshot(profiles=edited(MACBA_PROFILE, url="https://www.macba.cat")))

    assert result.venues_updated == 2
    assert {
        r[0] for r in query(db, "SELECT website_url FROM venues WHERE source_profile_id = 24")
    } == {"https://www.macba.cat"}


def test_small_move_is_a_change(db):
    # Guards the ST_AsBinary comparison: geography `=` may only compare bounding boxes.
    sync(db)
    result = sync(db, snapshot(venues=edited(895, longtitude="2.171491")))

    assert result.venues_updated == 1
    assert query(db, "SELECT ST_X(geom::geometry) FROM venues WHERE source_venue_id = 895") == [
        (pytest.approx(2.171491, abs=1e-9),)
    ]


def test_repeated_post_is_one_snapshot_with_two_occurrences(db):
    # `/events` is per occurrence: a recurring post appears once per date.
    def recurring(rs):
        again = next(r for r in rs if r["id"] == 59371)
        week_later = {"start": "2026-11-25T19:00:00+01:00", "end": "2026-11-25T21:00:00+01:00"}
        return [*rs, {**again, "occurrence_id": "99999", **week_later}]

    result = sync(db, snapshot(events=recurring))

    assert (result.events, result.occurrences) == (4, 5)
    assert query(
        db,
        "SELECT o.source_occurrence_id FROM event_occurrences o "
        "JOIN event_snapshots e ON e.id = o.event_id WHERE e.source_event_id = 59371 ORDER BY 1",
    ) == [(22821,), (99999,)]


def test_distance_filter_skips_null_island(db):
    # CLAUDE.md § 8. Within 2 km of MACBA: both MACBA terms, Chiquita Room, Biblioteca Sofia Barat;
    # not L&B Gallery (~2.7 km), and none of the 0,0 terms in the Gulf of Guinea.
    sync(db)
    near = query(
        db,
        "SELECT source_venue_id FROM venues "
        "WHERE ST_DWithin(geom, ST_GeogFromText('SRID=4326;POINT(2.1669 41.3833)'), 2000)",
    )
    assert {r[0] for r in near} == {159, 487, CHIQUITA, 895}


def test_rolled_back_run_writes_nothing(db):
    # What `sync-graf --dry-run` relies on: the real write, undone, still reports real numbers.
    result = sync(db, commit=False)

    assert (result.venues_inserted, result.events) == (8, 4)
    assert query(
        db,
        "SELECT (SELECT count(*) FROM venues) + (SELECT count(*) FROM event_snapshots)"
        " + (SELECT count(*) FROM event_occurrences)",
    ) == [(0,)]


def test_zero_venues_is_refused(db):
    # An empty fetch would otherwise unjoin every venue in the table.
    sync(db)
    with pytest.raises(ValueError, match="zero venues"):
        sync(db, GrafSnapshot(terms=[], profiles=[], events=[]))
    assert query(db, "SELECT count(*) FROM venues WHERE website_url IS NOT NULL") == [(3,)]


# --- report: pure --------------------------------------------------------------------------------


def _result(snap: GrafSnapshot, joins_before: dict[int, int]) -> SyncResult:
    venues = plan(snap)
    return SyncResult(
        venues=len(venues),
        venues_inserted=0,
        venues_updated=0,
        venues_unjoined=0,
        joins_before=joins_before,
        joins_after={v.source_venue_id: v.source_profile_id for v in venues if v.source_profile_id},
        events=4,
        occurrences=4,
    )


def test_report_names_every_join_change():
    # The join is recomputed each run, so a rename can move a URL between venues silently; this
    # block is the only place it shows.
    snap = snapshot()
    before = {159: MACBA_PROFILE, CHIQUITA: 14, 777: MACBA_PROFILE}  # 777: no longer served
    lines = report(snap, plan_sync(snap), _result(snap, before))

    start = next(i for i, line in enumerate(lines) if line.startswith("changes"))
    assert lines[start : start + 4] == [
        "changes    +1 joined   -1 unjoined   ~1 moved",
        "           + MACBA, Museu d'Art Contemporani de Barcelona -> MACBA",
        "           - term 777 (not in this fetch), was MACBA",
        "           ~ Chiquita Room: ADN Galeria -> Chiquita Room",
    ]


def test_report_lists_containment_joins_for_review():
    snap = snapshot()
    lines = report(snap, plan_sync(snap), _result(snap, {}))

    assert "contains   MACBA, Museu d'Art Contemporani de Barcelona -> MACBA" in lines
    assert "changes    first sync: 3 joined" in lines


# --- cli sync-graf --------------------------------------------------------------------------------


@pytest.fixture
def run(db, monkeypatch) -> Callable[..., Any]:
    """`run(*args, graf=FakeGraf())` invokes `sync-graf` against `db` and a fake GRAF."""
    monkeypatch.setenv("DATABASE_URL", db.render_as_string(hide_password=False))
    monkeypatch.setenv("GRAF_BASE_URL", BASE_URL)
    _clear_caches()

    def _run(*args: str, graf: FakeGraf | None = None):
        graf = graf or FakeGraf(**full_corpus())
        monkeypatch.setattr(
            cli,
            "open_client",
            functools.partial(open_client, POLICY, transport=graf.transport, sleep=no_sleep),
        )
        return CliRunner().invoke(cli.app, ["sync-graf", *args])

    yield _run
    _clear_caches()


def _clear_caches() -> None:
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


VENUE_COUNTS = (
    "SELECT count(*), count(*) FILTER (WHERE is_pilot), count(*) FILTER (WHERE crawl_enabled) "
    "FROM venues"
)


def test_sync_graf_writes_and_reports(run, db):
    result = run()

    assert result.exit_code == 0, result.output
    assert result.output.startswith("GRAF sync\n")
    for line in [
        "  venues     570 terms   0 with geometry   570 without (missing or 0,0)",
        "  joined     156 terms   (slug 92, base-slug 2, name 34, contains 28)   1 ambiguous",
        "  changes    first sync: 156 joined",
        "  written    570 new   0 changed   0 unjoined (no longer served)",
        "  pilots     20/20 matched   18 with a crawlable url",
        "             no url: ProjecteSD (blank url), #plantauno (no profile)",
        "  events     4 posts   4 occurrences   0 naming an unknown venue",
    ]:
        assert line in result.output.splitlines()
    assert query(db, VENUE_COUNTS) == [(570, 20, 149)]


def test_second_sync_reports_no_changes(run):
    run()
    result = run()

    assert result.exit_code == 0, result.output
    assert "  changes    +0 joined   -0 unjoined   ~0 moved" in result.output
    assert "  written    0 new   0 changed   0 unjoined (no longer served)" in result.output


def test_dry_run_reports_and_writes_nothing(run, db):
    result = run("--dry-run")

    assert result.exit_code == 0, result.output
    assert result.output.startswith("GRAF sync (dry run, rolled back)\n")
    assert "  written    570 new   0 changed   0 unjoined (no longer served)" in result.output
    assert query(
        db, "SELECT (SELECT count(*) FROM venues) + (SELECT count(*) FROM event_snapshots)"
    ) == [(0,)]


def test_truncated_fetch_fails_before_writing(run, db):
    # A blocked or cut-short fetch: without the floor it would unjoin every venue it missed.
    run()
    result = run(graf=FakeGraf())  # the trimmed fixture: 8 terms

    assert result.exit_code == 1
    assert (
        "FAIL: GRAF served 8 venue terms, expected at least 500. Nothing written." in result.output
    )
    assert query(db, VENUE_COUNTS) == [(570, 20, 149)]


def test_unmatched_pilot_exits_1_after_committing(run, db):
    result = run("--min-venues", "1", graf=FakeGraf())

    assert result.exit_code == 1
    assert "PILOT UNMATCHED: Fundació Joan Miró" in result.output
    assert len([line for line in result.output.splitlines() if "PILOT UNMATCHED" in line]) == 18
    assert query(db, "SELECT count(*) FROM venues") == [(8,)]  # the facts are kept

    allowed = run("--min-venues", "1", "--allow-unmatched-pilots", graf=FakeGraf())
    assert allowed.exit_code == 0, allowed.output


def test_deep_occurrences_asks_once_per_post(run):
    graf = FakeGraf(**full_corpus())
    result = run("--deep-occurrences", graf=graf)

    assert result.exit_code == 0, result.output
    assert sum("occurrences" in p for p in graf.paths()) == 4


def test_record_writes_scrubbed_fixtures(run, tmp_path):
    corpus = full_corpus()
    corpus["profiles"] = [{**p, "description": "Prosa de tercers."} for p in corpus["profiles"]]
    result = run("--dry-run", "--record", str(tmp_path), graf=FakeGraf(**corpus))

    assert result.exit_code == 0, result.output
    assert f"recorded {tmp_path / 'identity.json'}" in result.output
    for name in ("venues", "profiles", "events", "identity"):
        assert banned_keys(json.loads((tmp_path / f"{name}.json").read_text())) == set()
