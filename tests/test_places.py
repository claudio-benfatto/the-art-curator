"""Loading `places` (`ingest/places.py`, `cli load-places`) and resolving a name to a point
(`queries/events.resolve_place`).

The downloads are small CSVs built here in the shape the sources serve, with real names and
invented polygons (squares around a plausible centroid). Names and points are facts, so nothing
here is third-party prose. Venues are real GRAF terms, at the points GRAF gives them:

- `MACBA` and `MACBA, Museu d'Art Contemporani de Barcelona` are two terms of one museum.
- `Barcelona` is one of GRAF's catch-all terms named after a town, 0.4 km from the city's seat.
- `Palau Victòria Eugènia` is two terms, spelt with and without an accent, 140 m apart.
"""

import asyncio
import csv
import functools
import io
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
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.ingest import places
from art_curator.ingest.http import FetchError, open_client
from art_curator.queries import events as q
from tests.db import alembic_config, run_sql
from tests.graf_stub import POLICY, no_sleep


def square(longitude: float, latitude: float, half: float = 0.002) -> str:
    """A WGS84 square centred on the point, as the Barcelona files serve an area."""
    w, e, s, n = longitude - half, longitude + half, latitude - half, latitude + half
    return f"POLYGON (({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


# The files also carry each area in projected ETRS89 metres. The loader must never read those.
PROJECTED = (
    "POLYGON ((430162.1 4581936.9, 430102.8 4581878.5, 430018.6 4581140.8, 430162.1 4581936.9))"
)


def csv_text(header: list[str], rows: list[tuple]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


BARRIS_HEADER = [
    "codi_districte", "nom_districte", "codi_barri", "nom_barri", "geometria_etrs89",
    "geometria_wgs84",
]  # fmt: skip
DISTRICTS_HEADER = ["Codi_Districte", "nom_districte", "geometria_etrs89", "geometria_wgs84"]
MUNICIPALITIES_HEADER = ["codi", "nom", "nom_comarca", "latitud", "longitud"]

# The barri les Corts and its district are ~1.2 km apart here (1.4 km in the real files).
BARRIS = csv_text(
    BARRIS_HEADER,
    [
        ("10", "Sant Martí", "68", "el Poblenou", PROJECTED, square(2.2024, 41.4000)),
        ("04", "Les Corts", "19", "les Corts", PROJECTED, square(2.1290, 41.3850)),
        ("08", "Nou Barris", "49", "Canyelles", PROJECTED, square(2.1660, 41.4410)),
        (
            "01",
            "Ciutat Vella",
            "04",
            "Sant Pere, Santa Caterina i la Ribera",
            PROJECTED,
            square(2.1790, 41.3880),
        ),
    ],
)
DISTRICTS = csv_text(
    DISTRICTS_HEADER, [("04", "Les Corts", PROJECTED, square(2.1150, 41.3880, half=0.01))]
)
MUNICIPALITIES = csv_text(
    MUNICIPALITIES_HEADER,
    [
        ("080193", "Barcelona", "Barcelonès", "41.382405390000002", "2.1772411059999999"),
        ("080431", "Canyelles", "Garraf", "41.285755979999998", "1.722358067"),
        ("082981", "Vic", "Osona", "41.929967609999999", "2.254264193"),
        ("081017", "Hospitalet de Llobregat, l'", "Barcelonès", "41.35945967", "2.099712098"),
        ("999998", "No consta", "No consta", "0", "0"),
    ],
)


def records() -> list[places.PlaceRecord]:
    return (
        places.parse_barris(BARRIS)
        + places.parse_districts(DISTRICTS)
        + places.parse_municipalities(MUNICIPALITIES)
    )


def by_name(name: str) -> places.PlaceRecord:
    [record] = [r for r in records() if r.name == name]
    return record


# --- parsing -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "extra", "expected"),
    [
        ("el Poblenou", (), ("Poblenou",)),
        ("l'Antiga Esquerra de l'Eixample", (), ("Antiga Esquerra de l'Eixample",)),
        ("Les Corts", (), ("Corts",)),
        ("Vic", (), ()),
        ("Sant Pere, Santa Caterina i la Ribera", ("el Born",), ("el Born", "Born")),
    ],
)
def test_a_place_answers_to_its_name_without_the_article(name, extra, expected):
    assert places.aliases(name, extra) == expected


@pytest.mark.parametrize(
    ("filed", "name"),
    [
        ("Hospitalet de Llobregat, l'", "l'Hospitalet de Llobregat"),
        ("Garriga, la", "la Garriga"),
        ("Bòrdes, Es", "Es Bòrdes"),
        ("Castell d'Aro, Platja d'Aro i s'Agaró", "Castell d'Aro, Platja d'Aro i s'Agaró"),
        ("Vic", "Vic"),
    ],
)
def test_a_municipality_filed_under_its_noun_gets_its_article_back(filed, name):
    assert places.article_first(filed) == name


def test_each_kind_is_read_with_what_it_lies_within():
    poblenou, district, vic = by_name("el Poblenou"), by_name("Les Corts"), by_name("Vic")

    assert (poblenou.kind, poblenou.code, poblenou.within) == ("barri", "68", "Sant Martí")
    assert poblenou.wkt == square(2.2024, 41.4000)  # the WGS84 column, not the projected one
    assert (district.kind, district.code, district.within) == ("district", "04", "Barcelona")
    assert (vic.kind, vic.code, vic.within) == ("municipality", "082981", "Osona")
    assert vic.wkt == "POINT(2.254264193 41.92996761)"


def test_a_municipality_answers_to_its_noun_alone():
    assert by_name("l'Hospitalet de Llobregat").aliases == ("Hospitalet de Llobregat",)


def test_extra_aliases_reach_their_barri():
    assert by_name("Sant Pere, Santa Caterina i la Ribera").aliases == ("el Born", "Born")


def test_the_not_recorded_municipality_is_left_out():
    assert "No consta" not in {r.name for r in records()}


@pytest.mark.parametrize(
    ("parse", "text", "error"),
    [
        (
            places.parse_barris,
            csv_text(BARRIS_HEADER, [("10", "Sant Martí", "68", "el Poblenou", "", PROJECTED)]),
            "outside Catalonia",
        ),
        (
            places.parse_barris,
            csv_text(BARRIS_HEADER, [("10", "Sant Martí", "68", "el Poblenou", "", "")]),
            "not a WKT polygon",
        ),
        (
            places.parse_municipalities,
            csv_text(MUNICIPALITIES_HEADER, [("082981", "Vic", "Osona", "2.25", "41.93")]),
            "outside Catalonia",
        ),
        (
            places.parse_municipalities,
            csv_text(MUNICIPALITIES_HEADER, [("082981", "Vic", "Osona", "", "")]),
            "no usable point",
        ),
        (places.parse_districts, csv_text(["nom_districte"], [("Gràcia",)]), "missing"),
    ],
    ids=["projected-polygon", "no-polygon", "swapped-axes", "no-point", "missing-column"],
)
def test_a_download_that_cannot_be_loaded_is_refused(parse, text, error):
    with pytest.raises(places.PlacesError, match=error):
        parse(text)


def test_problems_name_what_would_make_a_bad_table():
    found = places.problems([*records(), by_name("Vic")])

    assert "4 barri rows, expected at least 73" in found
    assert "municipality code 082981 appears 2 times" in found
    assert "no barri named 'el Barri Gòtic' for EXTRA_ALIASES" in found


@pytest.fixture
def small_download(monkeypatch) -> None:
    """Take the sample for a whole download: its counts, and only the alias it has a barri for."""
    monkeypatch.setattr(places, "MIN_PLACES", {"barri": 4, "district": 1, "municipality": 3})
    monkeypatch.setattr(
        places, "EXTRA_ALIASES", {"Sant Pere, Santa Caterina i la Ribera": ("el Born",)}
    )


def test_problems_are_none_for_a_whole_download(small_download):
    assert places.problems(records()) == []


# --- fetching ------------------------------------------------------------------------------------


class FakeSources:
    """The three downloads over an httpx2 `MockTransport`, keyed by URL path."""

    def __init__(self, **overrides: httpx2.Response) -> None:
        self.responses = {
            httpx2.URL(places.BARRIS_URL).path: httpx2.Response(200, text=BARRIS),
            httpx2.URL(places.DISTRICTS_URL).path: httpx2.Response(200, text=DISTRICTS),
            httpx2.URL(places.MUNICIPALITIES_URL).path: httpx2.Response(
                200, text=MUNICIPALITIES, headers={"Content-Type": "text/csv; charset=UTF-8"}
            ),
            **overrides,
        }
        self.transport = httpx2.MockTransport(lambda request: self.responses[request.url.path])


def fetch(sources: FakeSources) -> list[places.PlaceRecord]:
    async def main() -> list[places.PlaceRecord]:
        async with open_client(POLICY, transport=sources.transport, sleep=no_sleep) as client:
            return await places.fetch(client)

    return asyncio.run(main())


def test_fetch_reads_every_source():
    assert fetch(FakeSources()) == records()


def test_fetch_refuses_a_failed_download():
    path = httpx2.URL(places.DISTRICTS_URL).path
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch(FakeSources(**{path: httpx2.Response(404)}))


# --- writing and resolving -----------------------------------------------------------------------

# Real GRAF terms, at GRAF's points; ethall is a pilot. Sala Sense Lloc is invented and has no
# point, so it is nowhere to measure from.
VENUES = """
INSERT INTO venues (source_venue_id, name, slug, city, geom) VALUES
  (159, 'MACBA', 'macba', 'Barcelona', ST_GeogFromText('POINT(2.166940 41.383152)')),
  (487, 'MACBA, Museu d''Art Contemporani de Barcelona', 'macba-museu', 'Barcelona',
   ST_GeogFromText('POINT(2.166867 41.383191)')),
  (248, 'Barcelona', 'barcelona', 'Barcelona', ST_GeogFromText('POINT(2.173403 41.385064)')),
  (300, 'Fundació Joan Miró', 'fundacio-joan-miro', 'Barcelona',
   ST_GeogFromText('POINT(2.1600 41.3685)')),
  (708, 'Palau Victòria Eugènia', 'palau-victoria-eugenia', 'Barcelona',
   ST_GeogFromText('POINT(2.150689 41.370763)')),
  (942, 'Palau Victoria Eugènia', 'palau-victoria-eugenia-2', 'Barcelona',
   ST_GeogFromText('POINT(2.151776 41.369747)')),
  (102, 'ethall', 'ethall-2', 'L''Hospitalet de Llobregat',
   ST_GeogFromText('POINT(2.124170 41.367572)')),
  (689, 'Aparador 22', 'aparador-22', 'Girona', ST_GeogFromText('POINT(2.822232 41.984441)')),
  (301, 'Sala Sense Lloc', 'sala-sense-lloc', NULL, NULL)
"""


def write(url: URL, rows: list[places.PlaceRecord]) -> int:
    async def main() -> int:
        engine = create_async_engine(url)
        try:
            async with AsyncSession(engine) as session:
                written = await places.write_places(session, rows)
                await session.commit()
                return written
        finally:
            await engine.dispose()

    return asyncio.run(main())


def resolve(url: URL, name: str) -> q.PlaceMatch:
    async def main() -> q.PlaceMatch:
        engine = create_async_engine(url)
        try:
            async with AsyncSession(engine) as session:
                return await q.resolve_place(session, name)
        finally:
            await engine.dispose()

    return asyncio.run(main())


@pytest.fixture(scope="module")
def places_db(fresh_database) -> Iterator[URL]:
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        run_sql(url, VENUES)
        write(url, records())
        yield url


def test_write_stores_an_area_at_its_centroid_and_a_town_at_its_seat(places_db):
    [rows] = run_sql(
        places_db,
        "SELECT name, round(ST_Y(geom::geometry)::numeric, 4), round(ST_X(geom::geometry)::numeric,"
        " 4), aliases FROM places WHERE name IN ('el Poblenou', 'Vic') ORDER BY name",
    )

    assert [(name, float(lat), float(lon), aliases) for name, lat, lon, aliases in rows] == [
        ("el Poblenou", 41.4, 2.2024, ["Poblenou"]),
        ("Vic", 41.93, 2.2543, []),
    ]


@pytest.mark.parametrize(
    ("asked", "name", "kind"),
    [
        ("Poblenou", "el Poblenou", "barri"),
        ("poblenou", "el Poblenou", "barri"),
        ("EL POBLENOU", "el Poblenou", "barri"),
        ("MACBA", "MACBA", "venue"),
        ("Vic", "Vic", "municipality"),
        ("vic", "Vic", "municipality"),
        ("Born", "Sant Pere, Santa Caterina i la Ribera", "barri"),
        ("Fundacio Joan Miro", "Fundació Joan Miró", "venue"),
    ],
)
def test_a_name_resolves_ignoring_case_accents_and_article(places_db, asked, name, kind):
    match = resolve(places_db, asked)

    assert (match.place.name, match.place.kind, match.candidates) == (name, kind, [])


def test_vic_resolves_to_its_seat(places_db):
    vic = resolve(places_db, "Vic").place

    assert (vic.within, vic.latitude, vic.longitude) == pytest.approx(
        ("Osona", 41.92996761, 2.254264193)
    )


def test_a_name_far_apart_in_two_places_returns_both(places_db):
    match = resolve(places_db, "Canyelles")

    assert match.place is None
    assert [(c.kind, c.qualified()) for c in match.candidates] == [
        ("barri", "Canyelles, Nou Barris"),
        ("municipality", "Canyelles, Garraf"),
    ]


def test_naming_what_it_lies_within_picks_one(places_db):
    match = resolve(places_db, "canyelles, garraf")

    assert (match.place.kind, match.place.within) == ("municipality", "Garraf")


def test_one_name_close_together_is_one_spot_the_most_specific(places_db):
    """The barri les Corts lies in its district, and GRAF's catch-all `Barcelona` term at the
    city's seat: neither is a real choice to put to anyone."""
    corts, city = resolve(places_db, "Les Corts"), resolve(places_db, "Barcelona")

    assert (corts.place.name, corts.place.kind) == ("les Corts", "barri")
    assert (city.place.name, city.place.kind) == ("Barcelona", "municipality")


def test_one_venue_under_two_terms_is_one_spot(places_db):
    """Equal once accents go, and 140 m apart: whichever comes first by name is the place."""
    match = resolve(places_db, "Palau Victoria Eugenia")

    assert (match.place.name, match.candidates) == ("Palau Victoria Eugènia", [])


def test_a_part_of_a_name_resolves_when_only_one_spot_holds_it(places_db):
    """No name is `Miró` or `Eugènia`. One venue holds the first; the second is in both Palau
    terms, which are one spot."""
    miro, palau = resolve(places_db, "Miró"), resolve(places_db, "Eugènia")

    assert miro.place.name == "Fundació Joan Miró"
    assert palau.place.name == "Palau Victoria Eugènia"


def test_a_town_is_not_every_venue_in_it(places_db):
    """ethall answers to "ethall, L'Hospitalet de Llobregat", but only as a whole: a part of the
    town's name is the town."""
    for asked in ("Hospitalet", "L'Hospitalet de Llobregat"):
        match = resolve(places_db, asked)
        assert (match.place.name, match.place.kind) == ("l'Hospitalet de Llobregat", "municipality")
    assert resolve(places_db, "ethall, l'hospitalet de llobregat").place.name == "ethall"


def test_an_exact_name_wins_over_names_that_contain_it(places_db):
    """`MACBA` is a whole name, and also part of another: the whole name is the answer."""
    assert resolve(places_db, "MACBA").place.name == "MACBA"


@pytest.mark.parametrize("asked", ["Atlantis", "Sala Sense Lloc", "", " - ", "22@"])
def test_a_name_for_nowhere_resolves_to_nothing(places_db, asked):
    """`22@` is no place we hold, and "22" is too short to be part of Aparador 22's name."""
    assert resolve(places_db, asked) == q.PlaceMatch(place=None, candidates=[])


def test_a_second_write_replaces_the_table(fresh_database):
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        write(url, records())
        write(url, [by_name("Vic")])

        assert run_sql(url, "SELECT name FROM places") == [[("Vic",)]]


# --- cli load-places -----------------------------------------------------------------------------


@pytest.fixture
def run(fresh_database, monkeypatch) -> Iterator[Callable[..., Any]]:
    """`run(*args, sources=FakeSources())` invokes `load-places` against a fresh database."""
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        monkeypatch.setenv("DATABASE_URL", url.render_as_string(hide_password=False))
        _clear_caches()

        def _run(*args: str, sources: FakeSources | None = None):
            sources = sources or FakeSources()
            monkeypatch.setattr(
                cli,
                "open_client",
                functools.partial(open_client, POLICY, transport=sources.transport, sleep=no_sleep),
            )
            return CliRunner().invoke(cli.app, ["load-places", *args])

        _run.url = url
        yield _run
        _clear_caches()


def _clear_caches() -> None:
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


def test_load_places_writes_and_credits_its_sources(run, small_download):
    result = run()

    assert result.exit_code == 0, result.output
    assert "municipality  4" in result.output
    assert "CC BY 4.0" in result.output
    assert "Llicència oberta" in result.output
    assert run_sql(run.url, "SELECT count(*) FROM places") == [[(9,)]]


def test_load_places_dry_run_writes_nothing(run, small_download):
    result = run("--dry-run")

    assert result.exit_code == 0, result.output
    assert "dry run, rolled back" in result.output
    assert run_sql(run.url, "SELECT count(*) FROM places") == [[(0,)]]


def test_load_places_refuses_a_short_download(run):
    result = run()

    assert result.exit_code == 1
    assert "4 barri rows, expected at least 73" in result.output
    assert "Nothing written" in result.output
    assert run_sql(run.url, "SELECT count(*) FROM places") == [[(0,)]]


def test_load_places_refuses_a_failed_download(run, small_download):
    path = httpx2.URL(places.MUNICIPALITIES_URL).path
    result = run(sources=FakeSources(**{path: httpx2.Response(503)}))

    assert result.exit_code == 1
    assert "Nothing written" in result.output
