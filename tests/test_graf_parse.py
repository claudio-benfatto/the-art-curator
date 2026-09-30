"""GRAF payload parsing. Every assertion here pins a trap observed in a live probe (2026-09-28).

The fixtures are recorded responses, scrubbed at capture (CLAUDE.md § 1) but otherwise faithful —
`_links`, `class_list` and `meta` are left in deliberately, because they are what proves the models
survive fields we have no column for.
"""

import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from art_curator.ingest.graf import (
    Event,
    Occurrence,
    Point,
    Profile,
    Term,
    banned_keys,
    clean_text,
    parse_datetime,
    parse_decimal,
    parse_events,
    parse_int,
    parse_point,
    parse_profiles,
    parse_terms,
    scrub,
)

FIXTURES = Path(__file__).parent / "fixtures" / "graf"


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def terms() -> list[Term]:
    return parse_terms(load("venues.json"))


@pytest.fixture(scope="module")
def profiles() -> list[Profile]:
    return parse_profiles(load("profiles.json"))


@pytest.fixture(scope="module")
def events() -> list[Event]:
    return parse_events(load("events.json"))


# --- the fixtures parse at all -------------------------------------------------------------------


def test_fixtures_parse_to_typed_objects(terms, profiles, events):
    assert len(terms) == 8
    assert len(profiles) == 5
    assert len(events) == 4
    assert {t.source_venue_id for t in terms} == {487, 159, 472, 601, 895, 486, 251, 752}


def test_identity_fixture_covers_the_whole_corpus():
    identity = load("identity.json")
    # Floors, not equalities: the term count moved 568 -> 570 in ten days (plan § Risks).
    assert len(identity["terms"]) >= 560
    assert len(identity["profiles"]) >= 150
    assert all({"id", "name", "slug"} <= set(t) for t in identity["terms"])
    assert all({"id", "name", "slug", "url"} <= set(p) for p in identity["profiles"])


# --- geometry ------------------------------------------------------------------------------------


def test_longtitude_is_the_real_field_name():
    """Their API misspells it. A correct `longitude` must yield no geometry, rather than crash."""
    correct = Term.model_validate(
        {"id": 1, "name": "n", "slug": "s", "latitude": "41.38", "longitude": "2.17"}
    )
    assert correct.point is None

    misspelled = Term.model_validate(
        {"id": 1, "name": "n", "slug": "s", "latitude": "41.38", "longtitude": "2.17"}
    )
    assert misspelled.point == Point(41.38, 2.17)


def test_null_island_terms_have_no_geometry(terms):
    """14 of 570 terms sit at 0,0. As a POINT they land in the Gulf of Guinea and silently corrupt
    the distance math of CLAUDE.md § 8."""
    by_slug = {t.slug: t for t in terms}
    assert by_slug["mar-mediterrani"].point is None
    assert by_slug["madrid"].point is None
    # 0,0 yet carrying a city — the coordinate is missing, the rest of the facts are not.
    girona = by_slug["museu-dhistoria-de-girona-sala-dexposicions"]
    assert girona.point is None and girona.city == "Girona"


@pytest.mark.parametrize(
    ("lat", "lon", "expected"),
    [
        ("41.383152", "2.166940", Point(41.383152, 2.16694)),
        ("0.000000", "0.000000", None),
        (0, 0, None),
        ("", "", None),
        (None, None, None),
        ("not a number", "2.17", None),
        ("41.38", "", None),
        ("91.0", "2.17", None),  # out of range: a corrupt value, not a place
        ("41.38", "181.0", None),
        ("0.000000", "2.166940", Point(0.0, 2.16694)),  # only *both* zero means missing
    ],
)
def test_parse_point(lat, lon, expected):
    assert parse_point(lat, lon) == expected


def test_ewkt_puts_longitude_first():
    """The one place WKT's x-y order is applied. MACBA is at lat 41.38, lon 2.17 — a swap would
    put it in Somalia, and nothing downstream would complain."""
    assert Point(41.383152, 2.16694).ewkt == "SRID=4326;POINT(2.16694 41.383152)"


# --- prices, ids, timestamps ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", None),  # what GRAF actually sends, for every event in the live window
        ("   ", None),
        (None, None),
        ("12", Decimal("12")),
        ("12.50", Decimal("12.50")),
        ("12,50", Decimal("12.50")),  # comma decimal, plausible from a Catalan admin
        (8, Decimal("8")),
        ("free", None),
        (True, None),  # a bool is not a price
    ],
)
def test_parse_decimal(value, expected):
    assert parse_decimal(value) == expected


def test_blank_prices_are_absent_not_zero(events):
    """`_event_price-min` is `""`, never absent. Reading it as 0 would make every event look free
    at a price of nothing, which is a different claim from "no price recorded"."""
    assert all(e.price_min is None and e.price_max is None for e in events)
    assert all(e.is_free is True for e in events)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("22815", 22815), (22815, 22815), (" 22815 ", 22815), ("", None), (None, None), ("x", None)],
)
def test_parse_int(value, expected):
    assert parse_int(value) == expected


def test_list_row_occurrence_id_is_a_string(events):
    """The `/events` list row serves `"23161"`; `/events/{id}/occurrences` serves `23161`. Both
    parse to the same int, because they key the same row."""
    raw = load("events.json")
    assert all(isinstance(row["occurrence_id"], str) for row in raw)
    assert all(isinstance(e.occurrence.source_occurrence_id, int) for e in events)

    from_list = Occurrence.model_validate(
        {"occurrence_id": "23161", "start": "2026-09-29T18:30:00+02:00", "end": None}
    )
    from_endpoint = Occurrence.model_validate(
        {"occurrence_id": 23161, "start": "2026-09-29T18:30:00+02:00", "end": None}
    )
    assert from_list == from_endpoint
    assert from_list.ends_at is None


def test_occurrence_dates_keep_their_offset():
    occ = Occurrence.model_validate(
        {
            "occurrence_id": 1,
            "start": "2026-09-29T18:30:00+02:00",
            "end": "2026-09-29T20:00:00+02:00",
        }
    )
    assert occ.starts_at == datetime(2026, 9, 29, 16, 30, tzinfo=UTC)
    assert occ.ends_at == datetime(2026, 9, 29, 18, 0, tzinfo=UTC)


def test_modified_gmt_is_naive_utc_and_modified_is_site_local(events):
    """`modified_gmt` 15:35:16 and `modified` 17:35:16 are the same instant. Taking `modified` and
    attaching UTC would record the event as edited two hours late."""
    raw = next(row for row in load("events.json") if row["id"] == 60390)
    assert raw["modified_gmt"] == "2026-09-24T15:35:16"  # naive, but UTC
    assert raw["modified"] == "2026-09-24T17:35:16"  # naive, and site-local
    event = next(e for e in events if e.source_event_id == 60390)
    assert event.source_modified_at == datetime(2026, 9, 24, 15, 35, 16, tzinfo=UTC)


def test_naive_timestamps_are_rejected_unless_declared_utc():
    """The single-event endpoint serves `start` as `"2026-09-29 18:30:00"` — site-local and naive.
    Guessing UTC there would shift every date, so a naive value is simply not a timestamp."""
    assert parse_datetime("2026-09-29 18:30:00") is None
    assert parse_datetime("2026-09-24T15:35:16", assume_utc=True) == datetime(
        2026, 9, 24, 15, 35, 16, tzinfo=UTC
    )
    # An explicit offset always wins over `assume_utc`.
    madrid = timezone(timedelta(hours=2))
    assert parse_datetime("2026-09-24T17:35:16+02:00", assume_utc=True) == datetime(
        2026, 9, 24, 17, 35, 16, tzinfo=madrid
    )
    assert parse_datetime("", assume_utc=True) is None
    assert parse_datetime(None) is None
    assert parse_datetime("not a date") is None


def test_event_dates_come_from_start_not_date(events):
    """Event `date` is absent from the payload (CLAUDE.md records it as null). Either way the
    dates live on `start`/`end`, and every event in the fixture has an occurrence."""
    assert all("date" not in row for row in load("events.json"))
    assert all(e.occurrence is not None for e in events)
    assert all(e.occurrence.starts_at.tzinfo is not None for e in events)


def test_event_without_occurrence_id_still_parses():
    """The single-event endpoint omits `occurrence_id`. That costs the dates, not the facts."""
    event = Event.model_validate(
        {"id": 1, "title": {"rendered": "T"}, "link": "https://graf.cat/x", "acf": {}}
    )
    assert event.occurrence is None
    assert event.title == "T"


# --- titles and entities -------------------------------------------------------------------------


def test_html_entities_are_unescaped_in_persisted_strings(events, terms, profiles):
    """Titles are dedup identifiers (CLAUDE.md § 1), so `d&#8217;estiu` and `d’estiu` must not be
    two identities for one event. `L&amp;B Gallery` is both a term and a profile name, and the
    matcher compares the two."""
    raw = next(row for row in load("events.json") if row["id"] == 57221)
    assert raw["title"]["rendered"] == "Col·lectiva d&#8217;estiu"
    assert next(e for e in events if e.source_event_id == 57221).title == "Col·lectiva d’estiu"

    assert next(row for row in load("venues.json") if row["id"] == 601)["name"] == "L&amp;B Gallery"
    assert next(t for t in terms if t.source_venue_id == 601).name == "L&B Gallery"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("L&amp;B Gallery", "L&B Gallery"),
        ("d&#8217;estiu", "d’estiu"),
        ("  padded  ", "padded"),
        ("", None),
        ("   ", None),
        (None, None),
        (42, None),
    ],
)
def test_clean_text(value, expected):
    assert clean_text(value) == expected


def test_blank_strings_become_absent(terms, profiles):
    """GRAF sends `""` for an unset city, postcode or profile url, not null."""
    assert next(t for t in terms if t.slug == "madrid").city is None
    # ProjecteSD's profile exists but carries `url: ""` — P2 must see no URL, not an empty one.
    assert next(p for p in profiles if p.slug == "projectesd").url is None


# --- the venue/profile entity split --------------------------------------------------------------


def test_a_term_has_no_url_and_a_profile_has_no_geography(terms, profiles):
    """The reason P1 joins them at all: neither entity is sufficient on its own."""
    assert not any(hasattr(t, "url") for t in terms)
    assert not any(hasattr(p, "point") for p in profiles)
    assert {p.source_profile_id for p in profiles} == {24, 83, 118, 173, 14}


def test_unknown_fields_are_ignored_not_fatal():
    """GRAF adds fields (`class_list`, `country`, `abe_event_venue`); a new one must not fail."""
    term = Term.model_validate(
        {
            "id": 1,
            "name": "n",
            "slug": "s",
            "country": "Espanya",
            "something_new_upstream": {"nested": True},
        }
    )
    assert term.source_venue_id == 1


def test_event_names_zero_or_one_term():
    assert (
        Event.model_validate(
            {"id": 1, "title": {"rendered": "T"}, "link": "u", "acf": {}, "event-venues": []}
        ).source_venue_id
        is None
    )
    assert (
        Event.model_validate(
            {
                "id": 1,
                "title": {"rendered": "T"},
                "link": "u",
                "acf": {},
                "event-venues": [895, 487],
            }
        ).source_venue_id
        == 895
    )


def test_event_venue_is_null_when_the_term_is_outside_the_fixture(events):
    """Event 57221 names term 594, which `venues.json` does not carry — the PR 14 case where an
    event gets `venue_id IS NULL` rather than aborting the sync."""
    event = next(e for e in events if e.source_event_id == 57221)
    assert event.source_venue_id == 594
    assert 594 not in {t["id"] for t in load("venues.json")}


# --- scrub ---------------------------------------------------------------------------------------


def test_scrub_drops_prose_at_any_depth():
    payload = {
        "id": 1,
        "name": "keep",
        "description": "third-party prose",
        "content": {"rendered": "<p>prose</p>"},
        "excerpt": {"rendered": "prose"},
        "guid": {"rendered": "https://graf.cat/?p=1"},
        "yoast_head": "<meta>",
        "yoast_head_json": {"title": "x"},
        "acf": {"title_ca": "keep", "desc_ca": "prosa", "desc_es": "prosa", "desc_en": "prose"},
        "nested": [{"deeper": [{"desc_ca": "prosa", "keep": 1}]}],
    }
    clean = scrub(payload)
    assert clean == {
        "id": 1,
        "name": "keep",
        "acf": {"title_ca": "keep"},
        "nested": [{"deeper": [{"keep": 1}]}],
    }
    assert banned_keys(clean) == set()
    assert "description" in banned_keys(payload)


def test_scrub_leaves_the_facts_alone():
    """Titles, names, slugs, addresses, coordinates and urls are facts or identifiers."""
    raw = load("venues.json")
    assert scrub(raw) == raw  # already scrubbed at capture
    assert parse_terms(scrub(raw)) == parse_terms(raw)


def test_scrub_is_applied_before_validation_so_prose_has_nowhere_to_go():
    """Even unscrubbed, the models drop prose — `scrub()` is the belt, this is the braces. The
    point of both is that no code path can carry `desc_ca` onward."""
    prosy = {
        "id": 1,
        "title": {"rendered": "T"},
        "link": "u",
        "acf": {"desc_ca": "prosa " * 200, "title_en": "T"},
        "description": "more prose",
    }
    event = Event.model_validate(scrub(prosy))
    dumped = event.model_dump_json()
    assert "prosa" not in dumped
    assert not any("desc" in field for field in Event.model_fields)
