"""Writing a GRAF fetch: `plan_venues` (pure) and `write_graf` (real database).

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
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from art_curator.ingest.graf import parse_events, parse_profiles, parse_terms
from art_curator.ingest.matching import match_pilots, match_profiles
from art_curator.ingest.sync import GrafSnapshot, SyncResult, VenueRow, plan_venues, write_graf
from tests.db import alembic_config, run_sql

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
    pilots, _ = match_pilots(snap.terms)
    return plan_venues(snap, match_profiles(snap.terms, snap.profiles), pilots.values())


def without(key: str, value: int) -> Callable[[list[dict]], list[dict]]:
    return lambda rows: [r for r in rows if r[key] != value]


def edited(row_id: int, **changes: Any) -> Callable[[list[dict]], list[dict]]:
    return lambda rows: [{**r, **changes} if r["id"] == row_id else r for r in rows]


# --- plan_venues: pure ---------------------------------------------------------------------------


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
