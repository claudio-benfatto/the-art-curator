"""`eval-curation` (PLAN.md § 8, PR 31): the scenarios, what a turn's messages say the tools
returned, which items a reply names, the report, and the run over a stubbed model and the test
corpus.

No model is called: `StubLlm` runs the real `LlmClient`, and its rows go to the test database so
the runner reads them back from `llm_calls` as it does live. The judge's own tests are in
`tests/test_curation_judge.py`.
"""

import asyncio
import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pytest
from alembic import command
from pydantic import ValidationError
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from typer.testing import CliRunner

from art_curator import cli
from art_curator.config import get_settings
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.eval import curation, curation_runner
from art_curator.eval.curation import (
    Area,
    CallUsage,
    Constraints,
    Exchange,
    Expect,
    Named,
    RunInfo,
    ScenarioRun,
    SeenItem,
)
from art_curator.llm import curation_judge
from art_curator.llm.client import db_recorder
from art_curator.llm.curation_judge import ToolUse, Verdict
from tests.corpus import CORPUS
from tests.db import alembic_config, run_sql
from tests.llm_stub import StubLlm

CHAT = "global.anthropic.claude-opus-4-6-v1"
WEDNESDAY = date(2026, 11, 4)  # in the corpus's week: Saturday is 2026-11-07


def scenarios(text: str) -> curation.Scenarios:
    return curation.load_scenarios(text)


ONE = """
scenarios:
  - id: one
    persona: Someone.
    turns:
      - user: Saturday?
        expect: {dates: saturday}
"""


def search(*rows: dict[str, Any], total: int | None = None, near: str | None = None) -> ToolUse:
    result = {
        "total": len(rows) if total is None else total,
        "events": list(rows),
        "near": {"name": near} if near else None,
        "truncated": total is not None and total > len(rows),
    }
    return ToolUse("search_events", {"date_from": "2026-11-07"}, json.dumps(result), False)


def near_search(near: dict[str, Any]) -> ToolUse:
    """A search with nothing found, measured from `near` (a PlaceRow as the tool sends it)."""
    result = {"total": 0, "events": [], "near": near, "truncated": False}
    return ToolUse("search_events", {"near": near["name"]}, json.dumps(result), False)


def hydrate(*rows: dict[str, Any]) -> ToolUse:
    result = json.dumps({"events": list(rows), "not_found": []})
    return ToolUse("get_events", {"ids": [r["id"] for r in rows]}, result, False)


def row(id_: str, title: str, distance_km: float | None = None, **extra: Any) -> dict[str, Any]:
    return {"id": id_, "title": title, "distance_km": distance_km, **extra}


def seen(*items: tuple[str, str, float | None]) -> dict[str, SeenItem]:
    return {i: SeenItem(i, (title,), distance) for i, title, distance in items}


# --- the scenarios -------------------------------------------------------------------------------


def test_the_committed_scenarios_are_valid():
    committed = curation.load_scenarios()
    assert len(committed.scenarios) >= 10
    for scenario in committed.scenarios:  # every one resolves, whatever day the eval runs
        for offset in range(7):
            scenario.constraints(WEDNESDAY + timedelta(days=offset))


@pytest.mark.parametrize(
    ("spec", "day"),
    [
        ("today", date(2026, 11, 4)),
        ("tomorrow", date(2026, 11, 5)),
        ("wednesday", date(2026, 11, 4)),  # today is the first Wednesday the calendar lists
        ("saturday", date(2026, 11, 7)),
        ("tuesday", date(2026, 11, 10)),
        ("today+13", date(2026, 11, 17)),
        ("monday+6", date(2026, 11, 15)),
    ],
)
def test_a_day_is_read_off_the_calendar(spec, day):
    assert curation.resolve_day(spec, WEDNESDAY) == day


def test_a_range_ends_on_or_after_its_first_day_on_any_day_of_the_week():
    weekend = Expect(dates=("saturday", "sunday"))
    for offset in range(7):
        today = WEDNESDAY + timedelta(days=offset)
        [constraints] = curation.Scenario(
            id="w", persona="p", turns=[{"user": "u", "expect": weekend}]
        ).constraints(today)
        assert constraints.date_from.weekday() == 5
        assert constraints.date_to == constraints.date_from + timedelta(days=1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("expect: {dates: next saturday}", "String should match pattern"),
        ("expect: {dates: [saturday, today]}", "runs backwards"),
        ("expect: {kinds: [party]}", "Input should be"),
        ("expect: {remembers: [hobbies]}", "Input should be"),
        ("expect: {area: {near: Poblenou}}", "Field required"),
        ("expect: {ends: [sunday, today]}", "runs backwards"),
        ("expect: {searches_near: ''}", "at least 1 character"),
        ("notes: anything", "Extra inputs are not permitted"),
    ],
)
def test_a_bad_scenario_is_refused(change, message):
    text = f"""
scenarios:
  - id: bad
    persona: Someone.
    turns:
      - user: Saturday?
        {change}
"""
    with pytest.raises(ValidationError, match=message):
        scenarios(text)


def test_scenario_ids_are_unique_and_selected_in_file_order():
    with pytest.raises(ValidationError, match="more than one scenario called"):
        scenarios(ONE + ONE.replace("scenarios:\n", ""))
    two = scenarios(ONE + ONE.replace("scenarios:\n", "").replace("id: one", "id: two"))
    assert [s.id for s in two.select(["two", "one"])] == ["one", "two"]
    assert [s.id for s in two.select([])] == ["one", "two"]
    with pytest.raises(ValueError, match="no scenario called"):
        two.select(["three"])


def test_constraints_carry_over_until_restated_and_null_drops_one():
    scenario = curation.Scenario(
        id="carry",
        persona="p",
        turns=[
            {"user": "1", "expect": {"dates": "saturday", "free": True, "searches": True}},
            {"user": "2", "expect": {"area": {"near": "Poblenou", "km": 2}}},
            {"user": "3", "expect": {"dates": "sunday", "free": None}},
        ],
    )

    first, second, third = scenario.constraints(WEDNESDAY)

    saturday, sunday = date(2026, 11, 7), date(2026, 11, 8)
    assert first == Constraints(saturday, saturday, free=True)
    assert second == Constraints(saturday, saturday, Area(near="Poblenou", km=2), free=True)
    assert third == Constraints(sunday, sunday, Area(near="Poblenou", km=2), free=False)
    assert scenario.turns[1].expect.searches is None  # checks belong to their own turn


def test_each_constraint_is_checked_by_its_own_wide_search():
    saturday, sunday = date(2026, 11, 7), date(2026, 11, 8)
    constraints = Constraints(
        saturday, sunday, Area(near="Poblenou", km=1), True, ("talk",), sunday, sunday
    )

    checks = {check.constraint: check for check in constraints.checks()}

    assert list(checks) == ["dates", "area", "price", "kinds", "ends"]
    assert checks["dates"].arguments["date_from"] == saturday
    for name in ("area", "price", "kinds"):
        assert checks[name].arguments["date_from"] == curation.ANY_DAY_FROM
        assert checks[name].meets(None)  # only `ends` reads a row's end
    assert checks["area"].arguments["near"] == "Poblenou"
    assert checks["area"].arguments["radius_km"] == 1
    assert checks["price"].arguments["free_only"]
    assert checks["kinds"].arguments["kinds"] == ["talk"]
    assert checks["ends"].arguments["date_from"] == checks["ends"].arguments["date_to"] == sunday
    assert checks["ends"].meets(sunday)
    assert not checks["ends"].meets(saturday) and not checks["ends"].meets(None)
    assert str(constraints) == (
        "2026-11-07 – 2026-11-08, within 1 km of Poblenou, free, talk, ending 2026-11-08"
    )
    assert Constraints().checks() == []


def test_ends_resolves_and_carries_over_like_dates():
    scenario = curation.Scenario(
        id="ends",
        persona="p",
        turns=[
            {"user": "1", "expect": {"ends": ["saturday", "sunday"]}},
            {"user": "2", "expect": {}},
            {"user": "3", "expect": {"ends": None}},
        ],
    )

    first, second, third = scenario.constraints(WEDNESDAY)

    assert (first.ends_from, first.ends_to) == (date(2026, 11, 7), date(2026, 11, 8))
    assert second == first
    assert (third.ends_from, third.ends_to) == (None, None)


def test_scenario_places_are_listed_once():
    found = curation.scenario_places(
        scenarios(
            ONE.replace("expect: {dates: saturday}", "expect: {area: {near: Vic, km: 5}}")
            + "      - user: again\n        expect: {area: {near: Vic, km: 9}}\n"
            + "      - user: and\n        expect: {searches_near: Gràcia}\n"
        ).scenarios
    )
    assert found == ["Vic", "Gràcia"]


# --- what a turn did -----------------------------------------------------------------------------


def test_tool_calls_are_paired_with_their_results_in_order():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "<context>"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "…", "signature": "s"},
                {"type": "tool_use", "id": "a", "name": "search_events", "input": {"x": 1}},
                {"type": "tool_use", "id": "b", "name": "get_events", "input": {"ids": []}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": '{"events":[]}'},
                {"type": "tool_result", "tool_use_id": "b", "content": "No.", "is_error": True},
            ],
        },
    ]

    assert curation.tool_uses(messages) == [
        ToolUse("search_events", {"x": 1}, '{"events":[]}', False),
        ToolUse("get_events", {"ids": []}, "No.", True),
    ]


def test_seen_items_keep_every_title_and_the_last_distance_measured():
    found = curation.seen_items(
        [
            search(row("g:3", "Assaig U", 0.2), row("x:1", "Prova U", 3.5)),
            search(row("x:1", "Prova U")),  # a later search without `near`
            hydrate(row("g:3", "Assaig U", title_en="Rehearsal U")),
            ToolUse("search_events", {}, "Unknown place.", True),
            ToolUse("remember", {"topic": "area"}, '{"id":1}', False),
        ]
    )

    assert found == {
        "g:3": SeenItem("g:3", ("Assaig U", "Rehearsal U"), 0.2),
        "x:1": SeenItem("x:1", ("Prova U",), 3.5),
    }


def test_a_title_is_named_by_whole_words_ignoring_case_accents_and_punctuation():
    items = seen(("x:1", "Prova U", 3.5), ("x:2", "Col·lecció", None), ("x:3", "Arxiu", 1.0))

    named = curation.named_items(
        "Start with **COL·LECCIO**, then prova u. Arxius are closed.", items
    )

    assert [n.title for n in named] == ["Col·lecció", "Prova U"]


def test_a_title_only_inside_a_longer_named_title_is_not_named_on_its_own():
    items = seen(
        ("x:1", "Aurèlia Muñoz. Ens", 1.0),
        ("a:1", "Visites comentades a Aurèlia Muñoz. Ens", 1.0),
        ("x:2", "Stan Douglas", 2.0),
    )

    inside = curation.named_items("Join the Visites comentades a Aurèlia Muñoz. Ens.", items)
    both = curation.named_items(
        "See Aurèlia Muñoz. Ens; the Visites comentades a Aurèlia Muñoz. Ens on Sunday.", items
    )

    assert [n.ids for n in inside] == [("a:1",)]
    assert [n.ids for n in both] == [("x:1",), ("a:1",)]


def test_a_title_is_named_by_a_piece_of_it_in_either_order():
    items = seen(
        ("g:45", "Misteri d'una forma. Jordi Alcaraz", 1.5),
        ("x:108", "Basel Abbas i Ruanne Abou-Rahme. Presoners de l'amor. Fins al sol", 3.5),
    )

    named = curation.named_items(
        "Jordi Alcaraz – Misteri d'una forma, then Basel Abbas i Ruanne Abou-Rahme at MACBA.",
        items,
    )

    assert [n.ids for n in named] == [("g:45",), ("x:108",)]  # once each, in reply order


def test_a_piece_names_a_title_only_if_it_is_two_words_and_its_own():
    items = seen(
        ("g:56", "Foudre. Nicolas Daubanes", 2.2),
        ("g:57", "Encara. Nicolas Daubanes", 2.2),  # his second show: his name names neither
        ("g:107", "Submón. Patricia Dauder", 2.0),
    )

    assert curation.named_items("Nicolas Daubanes opens on Saturday.", items) == []
    assert curation.named_items("Submón is on.", items) == []  # one word
    assert [n.ids for n in curation.named_items("Patricia Dauder is on.", items)] == [("g:107",)]


def test_a_piece_inside_another_title_names_neither():
    # The 2026-11-23 run: "Aurèlia Muñoz – Els Ocells-estels" was counted as "Aurèlia Muñoz. Ens".
    items = seen(
        ("x:368", "Aurèlia Muñoz. Ens", 0.1),
        ("x:109", "Els Ocells-estels d'Aurèlia Muñoz", 0.1),
    )

    assert curation.named_items("Aurèlia Muñoz – Els Ocells-estels, in the atrium.", items) == []
    assert [n.ids for n in curation.named_items("Aurèlia Muñoz. Ens is on.", items)] == [("x:368",)]


def test_an_item_named_by_either_of_its_titles_is_named_once_under_its_first():
    # GRAF gives a Catalan title and an English one; the reply wrote a bit of each.
    items = {
        "g:52": SeenItem(
            "g:52",
            (
                "Present permanent. A cura de Vincenç Altaió",
                "Permanent present. Curated by Vicenç Altaió",
            ),
            1.2,
        ),
        "g:47": SeenItem(
            "g:47", ("El rapte d’Europa. Santi Moix", "The abduction of Europe. Santi Moix"), 1.4
        ),
    }

    named = curation.named_items(
        "Present permanent, curated by Vicenç Altaió; then Santi Moix next door.", items
    )

    assert named == [
        Named("Present permanent. A cura de Vincenç Altaió", ("g:52",), 1.2),
        Named("El rapte d’Europa. Santi Moix", ("g:47",), 1.4),  # his name, in both its titles
    ]


def test_items_under_one_title_are_named_together_at_the_nearest_distance():
    items = seen(("x:2", "Arxiu", None), ("g:7", "Arxiu", 3.4), ("g:9", "U", 0.1))

    [named] = curation.named_items("Arxiu, then U.", items)  # "U" is too short to look for

    assert named == Named("Arxiu", ("x:2", "g:7"), 3.4)


def test_order_is_nearest_first_among_items_with_a_distance():
    def named(*distances: float | None) -> list[Named]:
        return [Named(str(i), (str(i),), d) for i, d in enumerate(distances)]

    assert curation.in_distance_order(named(0.2, None, 3.5)) is True
    assert curation.in_distance_order(named(3.5, 0.2)) is False
    assert curation.in_distance_order(named(1.0, None)) is None


def remembered(topic: str) -> ToolUse:
    return ToolUse("remember", {"topic": topic, "fact": "Something."}, '{"id":3}', False)


REFUSED_SEARCH = ToolUse("search_events", {"near": "Canyelles"}, "Could be any of…", True)


def test_checks_say_what_the_turn_did_wrong():
    did_not = Expect(searches=True, remembers=("area", "media"))
    must_not = Expect(searches=False, remembers=())

    assert curation.check_failures(did_not, [REFUSED_SEARCH, remembered("area")]) == [
        "did not search",
        "did not remember media",
    ]
    assert curation.check_failures(must_not, [search(), remembered("area")]) == [
        "searched, which it should not have",
        "remembered area, which it should not have",
    ]
    assert curation.check_failures(Expect(), [search()]) == []
    with_near = Expect(searches_near="Poblenou")
    assert [curation.checks_made(e) for e in (did_not, must_not, Expect(), with_near)] == [
        4,
        2,
        0,
        1,
    ]


def test_a_refused_search_is_not_searching():
    asks_first = Expect(searches=False)

    assert curation.check_failures(asks_first, [REFUSED_SEARCH]) == []


def test_remembers_is_exactly_those_topics():
    language_only = Expect(remembers=("language",))

    assert curation.check_failures(language_only, [remembered("language")]) == []
    assert curation.check_failures(
        language_only, [remembered("language"), remembered("area"), remembered("budget")]
    ) == ["remembered area, budget, which it should not have"]


def test_search_origins_are_where_successful_searches_measured_from():
    poblenou = {"name": "el Poblenou", "latitude": 41.4, "longitude": 2.2024}

    origins = curation.search_origins(
        [search(), REFUSED_SEARCH, near_search(poblenou), remembered("area")]
    )

    assert origins == [poblenou]


# --- the report ----------------------------------------------------------------------------------


def call(read: int = 0, written: int = 0, fresh: int = 100, cost: str = "0.05") -> CallUsage:
    return CallUsage(fresh, read, written, 50, Decimal(cost), 2000, "end_turn")


def exchange(**changes: Any) -> Exchange:
    fields: dict[str, Any] = {
        "user": "Saturday?",
        "constraints": Constraints(date(2026, 11, 7), date(2026, 11, 7)),
        "expect": Expect(searches=True),
        "reply": "Nit Oberta, then Prova U.",
        "stop_reason": "end_turn",
        "iteration_count": 1,
        "uses": (search(row("g:4", "Nit Oberta", 0.2), row("x:1", "Prova U", 3.5), total=9),),
        "calls": (call(), call()),
        "seconds": 12.0,
        "named": (Named("Nit Oberta", ("g:4",), 0.2), Named("Prova U", ("x:1",), 3.5)),
        "off": (),
        "failures": (),
    }
    return Exchange(**(fields | changes))


def run(*exchanges: Exchange, **changes: Any) -> ScenarioRun:
    scenario = scenarios(ONE).scenarios[0]
    return ScenarioRun(scenario, "<context>", exchanges, **changes)


def test_only_calls_after_a_cacheable_prefix_count_towards_the_cache_share():
    first = exchange(calls=(call(fresh=3000), call(fresh=500, written=4000)))
    second = exchange(calls=(call(read=4000, written=300, fresh=10),))

    assert curation.cache_share([run(first, second)]) == (1, 4000, 4310)
    assert curation.cache_share([run(exchange(calls=(call(fresh=3000),)))]) == (0, 0, 0)


def test_percentile_is_nearest_rank():
    assert curation.percentile([3.0, 1.0, 2.0, 4.0], 0.5) == 2.0
    assert curation.percentile([3.0, 1.0, 2.0, 4.0], 0.95) == 4.0
    assert curation.percentile([7.0], 0.5) == 7.0


INFO = RunInfo(date(2026, 11, 4), CHAT, "medium", "baseline", CHAT)


def test_the_report_gives_the_totals_and_what_each_exchange_got_wrong():
    bad = exchange(
        reply="Prova U, then Nit Oberta.",
        named=(Named("Prova U", ("x:1",), 3.5), Named("Nit Oberta", ("g:4",), 0.2)),
        off=(curation.OffConstraint("Prova U", "area"),),
        failures=("did not search",),
        iteration_count=2,
        stop_reason="max_tokens",
    )
    verdicts = (
        Verdict("fits", "Video, near."),
        Verdict("partly", "One is far.", ("a five-minute walk",)),
    )

    lines = curation.report(INFO, [run(exchange(), bad, verdicts=verdicts)])
    text = "\n".join(lines)

    assert "2 exchanges" in text and "4 items" in text
    assert "1 claims in 1 of 2 judged exchanges" in text
    assert "1 named items off constraint   1 area" in text
    assert "nearest first in 50% (1/2)" in text
    assert "checks       50% (1/2) met" in text
    assert "1 fits, 1 partly, 0 misses, 0 none" in text
    assert "1: 1, 2: 1" in text
    assert "100% (2/2) of searches" in text
    assert "$0.2000   $0.1000 an exchange" in text
    assert "1 max_tokens" in text
    assert "'Prova U'" in text and "'a five-minute walk'" in text and "did not search" in text


def test_a_paced_run_says_so_and_how_long_it_waited_for_the_quota():
    paced = replace(INFO, per_minute=5)
    waited = replace(run(exchange()), waited_seconds=95.4)

    text = "\n".join(curation.report(paced, [waited]))

    assert "paced to 5 calls a minute a model" in text
    assert "(waits for the quota left out)" in text
    assert "waited 95 s in all for the requests-per-minute quota" in text
    assert "waited 95 s for the quota" in curation.progress_line(waited)
    unpaced = "\n".join(curation.report(INFO, [run(exchange())]))
    assert "paced" not in unpaced and "quota" not in unpaced


def test_a_semantic_arm_says_so_and_counts_the_searches_that_passed_a_query():
    queried = ToolUse(
        "search_events", {"date_from": "2026-11-07", "query": "video art"}, search().result, False
    )
    semantic = replace(INFO, arm="semantic", embed_model="cohere.embed-v4:0")

    text = "\n".join(curation.report(semantic, [run(exchange(uses=(search(), queried)))]))

    assert "semantic ranking cohere.embed-v4:0" in text
    assert "queries      50% (1/2) of searches passed a query" in text
    assert "filters only" in "\n".join(curation.report(INFO, [run(exchange())]))
    assert "passed a query" not in "\n".join(curation.report(INFO, [run(exchange())]))


def test_the_transcript_holds_each_conversation_for_a_person_to_judge():
    remembered = ("media: Likes video art.",)
    text = curation.transcript(
        INFO, [run(exchange(), verdicts=(Verdict("fits", "Video."),), remembered=remembered)]
    )

    assert "## one" in text and "**Persona:** Someone." in text and "<context>" in text
    assert "### 1. You: Saturday?" in text and "Nit Oberta, then Prova U." in text
    assert "→ 2 of 9, truncated" in text
    assert "judge: taste fits: Video." in text and "- media: Likes video art." in text
    assert "**Sensible?** yes / no" in text


# --- the run over the corpus ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_db(fresh_database) -> Iterator[URL]:
    """The test corpus in a database of its own: the runs below add users and `llm_calls`."""
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        run_sql(url, *CORPUS)
        yield url


def _in_db(url: URL, work: Callable[[async_sessionmaker], Any]) -> Any:
    async def main() -> Any:
        engine = create_async_engine(url)
        try:
            return await work(async_sessionmaker(engine, expire_on_commit=False))
        finally:
            await engine.dispose()

    return asyncio.run(main())


def test_named_items_are_checked_against_each_constraint_in_sql(eval_db):
    named = [
        Named("Nit Oberta", ("g:4",), 0.2),  # an opening at the annex, free, on the Saturday
        Named("Prova U", ("x:1",), 3.5),  # on, price unknown, 3.5 km away
        Named("Arxiu", ("x:2", "g:7"), 3.4),  # the venue's record is over; GRAF's is on
        Named("Fira", ("g:2",), None),  # one day in October, paid, nowhere
    ]
    saturday = date(2026, 11, 7)
    constraints = Constraints(
        saturday, saturday, Area(near="Poblenou", km=1), True, ("exhibition",)
    )

    async def check(sessions: async_sessionmaker) -> list[curation.OffConstraint]:
        async with sessions() as session:
            return await curation_runner.off_constraint(session, named, constraints)

    off = [(o.constraint, o.title) for o in _in_db(eval_db, check)]

    assert off == [
        ("dates", "Fira"),
        ("area", "Prova U"),
        ("area", "Arxiu"),
        ("area", "Fira"),
        ("price", "Fira"),
        ("kinds", "Nit Oberta"),
        ("kinds", "Fira"),
    ]


def test_ends_keeps_only_what_ends_on_the_days_asked(eval_db):
    named = [
        Named("Stand a la fira", ("x:4",), None),  # 6 to 8 November: ends on the Sunday
        Named("Prova U", ("x:1",), 3.5),  # on until 30 November
        Named("Col·lecció", ("x:3",), 3.5),  # no end date
    ]
    weekend = (date(2026, 11, 7), date(2026, 11, 8))
    constraints = Constraints(ends_from=weekend[0], ends_to=weekend[1])

    async def check(sessions: async_sessionmaker) -> list[curation.OffConstraint]:
        async with sessions() as session:
            return await curation_runner.off_constraint(session, named, constraints)

    off = [(o.constraint, o.title) for o in _in_db(eval_db, check)]

    assert off == [("ends", "Prova U"), ("ends", "Col·lecció")]


# The corpus's one place, and Galeria Exemple ~3.5 km from it.
POBLENOU = {"name": "el Poblenou", "kind": "barri", "latitude": 41.4, "longitude": 2.2024}
GALERIA = {"name": "Galeria Exemple", "kind": "venue", "latitude": 41.38, "longitude": 2.17}


def test_searches_near_passes_on_a_search_from_the_same_spot(eval_db):
    expect = Expect(searches_near="Poblenou")
    annex = {"name": "Annex Exemple", "kind": "venue", "latitude": 41.4, "longitude": 2.2}

    async def check(sessions: async_sessionmaker) -> list[list[str]]:
        async with sessions() as session:
            return [
                await curation_runner.near_failures(session, expect, uses)
                for uses in (
                    [near_search(GALERIA), near_search(POBLENOU)],
                    [near_search(annex)],  # ~0.2 km away: the same spot
                    [near_search(GALERIA), REFUSED_SEARCH],
                    [search()],
                    [],
                )
            ]

    poblenou, annex_ok, galeria, nowhere, none = _in_db(eval_db, check)

    assert poblenou == annex_ok == []
    assert galeria == ["did not search near Poblenou (searched near: Galeria Exemple)"]
    assert nowhere == none == ["did not search near Poblenou (searched near: no place)"]
    assert _in_db(eval_db, lambda s: _near(s, Expect(), [])) == []


async def _near(sessions: async_sessionmaker, expect: Expect, uses: list[ToolUse]) -> list[str]:
    async with sessions() as session:
        return await curation_runner.near_failures(session, expect, uses)


def test_a_place_the_scenarios_cannot_pin_down_is_reported(eval_db):
    async def check(sessions: async_sessionmaker) -> list[str]:
        async with sessions() as session:
            return await curation_runner.unresolved_places(session, ["Poblenou", "Atlantis"])

    [problem] = _in_db(eval_db, check)
    assert "'Atlantis'" in problem


SCENARIO = """
scenarios:
  - id: corpus-saturday
    persona: Starts from Poblenou; likes openings.
    profile:
      - {topic: area, fact: Starts from Poblenou.}
    turns:
      - user: Saturday, within a kilometre?
        expect:
          dates: saturday
          area: {near: Poblenou, km: 1}
          searches: true
          remembers: []
"""

SEARCH_SATURDAY = {"date_from": "2026-11-07", "date_to": "2026-11-07", "near": "Poblenou"}
REPLY = "Nit Oberta at Annex Exemple first, then Prova U at Galeria Exemple."


def queue_turn(stub: StubLlm) -> None:
    stub.reply(
        content=[
            {"type": "tool_use", "id": "s", "name": "search_events", "input": SEARCH_SATURDAY}
        ],
        stop_reason="tool_use",
        input_tokens=3000,
        output_tokens=80,
    )
    stub.reply(
        content=[{"type": "text", "text": REPLY}],
        input_tokens=20,
        output_tokens=60,
        cache_creation_input_tokens=4500,
    )


def queue_verdicts(stub: StubLlm, claims: tuple[str, ...] = ()) -> None:
    verdict = {
        "exchange": 1,
        "taste": "partly",
        "reason": "Prova U is too far.",
        "unsupported_claims": list(claims),
    }
    stub.tool_reply(curation_judge.TOOL, {"verdicts": [verdict]})


def run_corpus(url: URL, stub_for: Callable[[async_sessionmaker], StubLlm], judge: bool):
    scenario = scenarios(SCENARIO).scenarios[0]

    async def work(sessions: async_sessionmaker) -> ScenarioRun:
        stub = stub_for(sessions)
        return await curation_runner.run_scenario(
            stub.client,
            sessions,
            scenario,
            today=WEDNESDAY,
            model=CHAT,
            effort="medium",
            embed_model=None,
            judge_model=CHAT if judge else None,
        )

    return _in_db(url, work)


def eval_users(url: URL) -> list[tuple]:
    [rows] = run_sql(url, "SELECT id FROM users WHERE client = 'eval'")
    return rows


def test_a_scenario_runs_through_the_curator_and_leaves_nothing_but_its_calls(eval_db):
    stubs: list[StubLlm] = []

    def stub_for(sessions: async_sessionmaker) -> StubLlm:
        stub = StubLlm(record=db_recorder(sessions))
        queue_turn(stub)
        queue_verdicts(stub, ("at Annex Exemple first",))
        stubs.append(stub)
        return stub

    done = run_corpus(eval_db, stub_for, judge=True)

    [stub] = stubs
    assert "] area: Starts from Poblenou." in done.context  # with its id, whatever it is
    assert "Wednesday 2026-11-04 (today)" in stub.requests[0]["messages"][0]["content"][0]["text"]
    [only] = done.exchanges
    assert only.reply == REPLY and only.iteration_count == 1
    assert [n.title for n in only.named] == ["Nit Oberta", "Prova U"] and only.in_order
    assert [(o.constraint, o.title) for o in only.off] == [("area", "Prova U")]
    assert only.failures == ()
    assert [c.prompt_tokens for c in only.calls] == [3000, 4520]
    assert only.cost_usd > 0
    assert done.remembered == ("area: Starts from Poblenou.",)
    assert done.verdicts == (Verdict("partly", "Prova U is too far.", ("at Annex Exemple first",)),)
    assert len(done.judge_calls) == 1 and not done.stopped
    assert eval_users(eval_db) == []
    judged = stub.converse_requests[0]["messages"][0]["content"][0]["text"]
    assert "Tool call: search_events" in judged and REPLY in judged


def test_a_refused_call_stops_the_run_and_still_deletes_the_user(eval_db):
    def stub_for(sessions: async_sessionmaker) -> StubLlm:
        stub = StubLlm(record=db_recorder(sessions))
        stub.fail(500)
        return stub

    done = run_corpus(eval_db, stub_for, judge=True)

    assert done.exchanges == () and "InternalServerError" in done.stopped
    assert done.verdicts is None
    assert eval_users(eval_db) == []


def test_without_the_judge_nothing_is_judged(eval_db):
    def stub_for(sessions: async_sessionmaker) -> StubLlm:
        stub = StubLlm(record=db_recorder(sessions))
        queue_turn(stub)
        return stub

    done = run_corpus(eval_db, stub_for, judge=False)

    assert len(done.exchanges) == 1 and done.verdicts is None and done.judge_calls == ()


# --- the command ---------------------------------------------------------------------------------


@pytest.fixture
def eval_curation(eval_db, monkeypatch, tmp_path) -> Iterator[Callable[..., Any]]:
    """`eval_curation(*args)` runs the command on the corpus database with SCENARIO as the
    scenarios and the model stubbed, recording to that database; `.stub` holds what was sent."""
    monkeypatch.setenv("DATABASE_URL", eval_db.render_as_string(hide_password=False))
    parsed = scenarios(SCENARIO)  # before the loader is replaced
    monkeypatch.setattr(curation, "load_scenarios", lambda: parsed)
    _clear_caches()
    stub = StubLlm(record=db_recorder(get_sessionmaker()))
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)

    def _run(*args: str) -> Any:
        return CliRunner().invoke(cli.app, ["eval-curation", "--today", "2026-11-04", *args])

    _run.stub = stub
    _run.transcript = tmp_path / "transcript.md"
    yield _run
    _clear_caches()


def _clear_caches() -> None:
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


def test_cli_eval_curation_reports_and_writes_the_transcript(eval_curation):
    queue_turn(eval_curation.stub)
    queue_verdicts(eval_curation.stub)

    result = eval_curation("--transcript", str(eval_curation.transcript))

    assert result.exit_code == 0, result.output
    assert "corpus-saturday" in result.output and "1 exchanges" in result.output
    assert "1 named items off constraint   1 area" in result.output
    assert str(eval_curation.transcript) in result.output
    assert "**Sensible?**" in eval_curation.transcript.read_text(encoding="utf-8")


def test_cli_eval_curation_refuses_what_it_cannot_run_before_calling_a_model(monkeypatch):
    parsed = scenarios(SCENARIO)  # before the loader is replaced
    monkeypatch.setattr(curation, "load_scenarios", lambda: parsed)
    stub = StubLlm()
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)

    def run(*args: str) -> Any:
        return CliRunner().invoke(cli.app, ["eval-curation", *args])

    unknown = run("--scenario", "nope")
    unpriced = run("--model", "global.anthropic.claude-unpriced-v1")

    assert unknown.exit_code == 1 and "no scenario called ['nope']" in unknown.output
    assert unpriced.exit_code == 1 and "no price in pricing.yaml" in unpriced.output
    assert stub.requests == []
