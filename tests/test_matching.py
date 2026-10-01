"""The venue term <-> profile matcher. Exact counts are pinned against `identity.json` — all 570
terms and 158 profiles as of 2026-09-28 — so a change to any round shows up as a number moving.

Planted cases use the same parsers as the fixture, so the matcher never sees a shape GRAF cannot
produce.
"""

import json
import re
from pathlib import Path

import pytest

from art_curator.ingest.graf import Profile, Term, parse_profiles, parse_terms
from art_curator.ingest.matching import (
    MIN_CONTAINMENT_LENGTH,
    PILOT_VENUES,
    MatchResult,
    base_slug,
    classify_url,
    match_pilots,
    match_profiles,
    normalize_name,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "graf"


@pytest.fixture(scope="module")
def identity() -> tuple[list[Term], list[Profile]]:
    data = json.loads((FIXTURES / "identity.json").read_text())
    return parse_terms(data["terms"]), parse_profiles(data["profiles"])


@pytest.fixture(scope="module")
def result(identity) -> MatchResult:
    return match_profiles(*identity)


@pytest.fixture(scope="module")
def profiles_by_id(identity) -> dict[int, Profile]:
    return {p.source_profile_id: p for p in identity[1]}


def terms(*rows: tuple[int, str, str]) -> list[Term]:
    return parse_terms({"id": i, "name": name, "slug": slug} for i, name, slug in rows)


def profiles(*rows: tuple[int, str, str]) -> list[Profile]:
    return parse_profiles(
        {"id": i, "name": name, "slug": slug, "url": f"https://p{i}.example"}
        for i, name, slug in rows
    )


# --- normalisation -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("àngels barcelona", "angels barcelona"),
        ("Àngels Barcelona", "angels barcelona"),
        ("Galería Alegría", "galeria alegria"),
        ("#plantauno", "plantauno"),
        ("L&B Gallery", "l and b gallery"),
        # GRAF serves the escaped form as both a term and a profile name.
        ("L&amp;B Gallery", "l and b gallery"),
        ("MACBA, Museu d'Art Contemporani", "macba museu d art contemporani"),
        ("  Espai 13 -  Fundació Joan Miró ", "espai 13 fundacio joan miro"),
        ("A*DESK", "a desk"),
        ("", ""),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_ampersand_does_not_vanish():
    # `L&B` must not collapse onto an unrelated `LB`.
    assert normalize_name("L&B Gallery") != normalize_name("LB Gallery")


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("chiquita-room-2", "chiquita-room"),
        ("ethall-2", "ethall"),
        ("espai-13", "espai"),  # the rule is blind to meaning; round 1 runs first for this reason
        ("macba", "macba"),
        ("Piramidon-2", "piramidon"),
        ("", ""),
    ],
)
def test_base_slug(slug, expected):
    assert base_slug(slug) == expected


# --- classify_url --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (None, (None, None, False)),
        ("", (None, None, False)),
        ("   ", (None, None, False)),
        ("ftp://example.org", (None, None, False)),
        ("mailto:info@example.org", (None, None, False)),
        ("www.example.org", (None, None, False)),  # no scheme: not something to fetch
        ("http://macba.cat", ("http://macba.cat", None, True)),
        ("https://fuga.gallery/", ("https://fuga.gallery/", None, True)),
        (
            "https://www.instagram.com/laskunst",
            (None, "https://www.instagram.com/laskunst", False),
        ),
        ("https://INSTAGRAM.com/x", (None, "https://INSTAGRAM.com/x", False)),
        ("https://m.instagram.com/x", (None, "https://m.instagram.com/x", False)),
        ("https://instagr.am/x", (None, "https://instagr.am/x", False)),
        # A host that merely ends in the same letters is not Instagram.
        ("https://notinstagram.com/x", ("https://notinstagram.com/x", None, True)),
        ("https://instagram.com.example.org", ("https://instagram.com.example.org", None, True)),
    ],
)
def test_classify_url(url, expected):
    assert classify_url(url) == expected


def test_only_two_profiles_are_instagram(identity):
    # PLAN.md § 9 estimated ~12. Measured: 2.
    _, ps = identity
    assert sum(1 for p in ps if classify_url(p.url)[1]) == 2


# --- match_profiles over the real corpus ---------------------------------------------------------


def test_identity_fixture_size(identity):
    ts, ps = identity
    assert (len(ts), len(ps)) == (570, 158)


def test_round_counts(result):
    assert result.counts == {"slug": 92, "base-slug": 2, "name": 34, "contains": 28}


def test_joined_url_floor(result, profiles_by_id):
    # P1 done-when: >=145 terms joined to a profile URL.
    urls = (profiles_by_id[m.source_profile_id].url for m in result.matches.values())
    crawlable = [url for url in urls if classify_url(url)[2]]
    assert len(crawlable) == 149
    assert len(crawlable) >= 145


def test_ambiguities(result):
    # P1 done-when: <=2. The one today is `Prats Nogueras Blanchard - Poblenou`, which contains
    # both `Nogueras Blanchard` and `Prats Nogueras Blanchard`. Decided 2026-10-01: it stays
    # ambiguous. Longest-match-wins would resolve it, but it loosens "exactly one candidate".
    assert result.ambiguous == {706: (92, 177)}
    assert 706 not in result.matches


def test_ambiguous_and_matched_are_disjoint(result):
    assert not result.matches.keys() & result.ambiguous.keys()


@pytest.mark.parametrize(
    ("term_id", "profile_id", "round_"),
    [
        (601, 143, "slug"),  # L&B Gallery — entities on both sides
        (472, 83, "name"),  # Chiquita Room: chiquita-room-2 vs chiquitaroom
        (159, 24, "slug"),  # MACBA
        (487, 24, "contains"),  # MACBA, Museu d'Art Contemporani… — 67 events
        (492, 24, "contains"),  # Convent dels Àngels - MACBA
        (893, 24, "contains"),  # Capella MACBA
        (704, 177, "slug"),  # Prats Nogueras Blanchard
        (684, 178, "contains"),  # FUGA Gallery -> profile `FUGA`
    ],
)
def test_known_joins(result, term_id, profile_id, round_):
    assert result.matches[term_id] == (term_id, profile_id, round_)


def test_one_profile_serves_many_terms(result):
    macba = sorted(t for t, m in result.matches.items() if m.source_profile_id == 24)
    assert macba == [159, 487, 492, 893]


def test_containment_is_token_wise(identity, result):
    # `Sismògraf` contains `GRAF` as a substring, not as a word.
    ts, _ = identity
    sismograf = next(t for t in ts if t.name == "Sismògraf")
    assert sismograf.source_venue_id not in result.matches
    assert sismograf.source_venue_id not in result.ambiguous


# --- match_profiles over planted cases -----------------------------------------------------------


def test_slug_beats_name():
    ts = terms((1, "Alpha", "alpha"))
    ps = profiles((10, "Beta", "alpha"), (11, "Alpha", "something-else"))
    assert match_profiles(ts, ps).matches[1] == (1, 10, "slug")


def test_name_collision_is_ambiguous_and_matches_nothing():
    ts = terms((1, "Espai Nou", "espai-nou-3"))
    ps = profiles((10, "Espai Nou", "espainou"), (11, "Espai  Nóu", "espai_nou"))
    r = match_profiles(ts, ps)
    assert r.matches == {}
    assert r.ambiguous == {1: (10, 11)}


def test_ambiguity_does_not_fall_through_to_a_later_round():
    # Round 3 is ambiguous and decides. Falling through to containment would report (10, 11, 12).
    ts = terms((1, "Sala Gran", "sala-gran-x"))
    ps = profiles((10, "Sala Gran", "a"), (11, "sala gran", "b"), (12, "Gran", "c"))
    r = match_profiles(ts, ps)
    assert r.matches == {}
    assert r.ambiguous == {1: (10, 11)}


def test_containment_with_two_hits_is_ambiguous():
    ts = terms((1, "Galeria Uno Galeria Dos", "galeria-uno-galeria-dos"))
    ps = profiles((10, "Galeria Uno", "uno"), (11, "Galeria Dos", "dos"))
    r = match_profiles(ts, ps)
    assert r.matches == {}
    assert r.ambiguous == {1: (10, 11)}


def test_containment_ignores_short_names():
    short = "Sol"
    assert len(normalize_name(short)) < MIN_CONTAINMENT_LENGTH
    ts = terms((1, "Plaça del Sol", "placa-del-sol"))
    ps = profiles((10, short, "sol"))
    r = match_profiles(ts, ps)
    assert r.matches == {} and r.ambiguous == {}


def test_containment_needs_whole_tokens():
    ts = terms((1, "Ciutadella Park", "ciutadella-park"))
    ps = profiles((10, "Ciutadel", "ciutadel"))
    assert match_profiles(ts, ps).matches == {}


def test_profile_id_lookup(result):
    assert result.profile_id(159) == 24
    assert result.profile_id(706) is None  # ambiguous
    assert result.profile_id(-1) is None


# --- pilots --------------------------------------------------------------------------------------


def _plan_pilots() -> tuple[str, ...]:
    plan = (ROOT / "PLAN.md").read_text()
    match = re.search(r"^\*\*Pilot venues:\*\* (.+)\.$", plan, re.MULTILINE)
    assert match, "no pilot list in PLAN.md § 7"
    return tuple(name.strip() for name in match.group(1).split(","))


def test_pilot_list_matches_plan():
    assert _plan_pilots() == PILOT_VENUES
    assert len(PILOT_VENUES) == 20


def test_all_pilots_resolve(identity):
    ts, _ = identity
    matched, unmatched = match_pilots(ts)
    assert unmatched == ()
    assert set(matched) == set(PILOT_VENUES)
    assert len(set(matched.values())) == 20
    assert matched["Chiquita Room"] == 472  # slug `chiquita-room-2`
    assert matched["#plantauno"] == 868


def test_pilots_without_a_crawlable_url(identity, result, profiles_by_id):
    # P2 needs to know which pilots it cannot crawl: `#plantauno` has no profile, ProjecteSD's
    # profile has `url: ""`. 18 of 20 are crawlable.
    ts, _ = identity
    matched, _ = match_pilots(ts)
    uncrawlable = set()
    for name, term_id in matched.items():
        profile_id = result.profile_id(term_id)
        if profile_id is None or not classify_url(profiles_by_id[profile_id].url)[2]:
            uncrawlable.add(name)
    assert uncrawlable == {"#plantauno", "ProjecteSD"}


def test_pilot_with_two_terms_is_unmatched():
    ts = terms((1, "Sala Parés", "sala-pares"), (2, "Sala Pares", "sala-pares-2"))
    matched, unmatched = match_pilots(ts, names=("Sala Parés",))
    assert matched == {}
    assert unmatched == ("Sala Parés",)
