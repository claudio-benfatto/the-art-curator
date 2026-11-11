"""The curator's loop, tools and prompt (PLAN.md § 8, PR 28), over a stubbed model.

What PR 28 must show without calling a model: the cap holds, a bad tool input comes back as
`is_error`, every call carries one cache breakpoint on its last block, and thinking blocks are
replayed exactly as they came. The tools' database half runs against `tests/corpus.py`.
"""

import asyncio
import json
from datetime import date
from typing import Any

import pytest
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from typer.testing import CliRunner

from art_curator import cli
from art_curator.agent.loop import CAPPED, MAX_TOOL_ROUNDS, Conversation, Curator, Turn
from art_curator.agent.prompt import (
    CALENDAR_DAYS,
    SYSTEM_PROMPT,
    conversation_context,
)
from art_curator.agent.tools import TOOL_DEFINITIONS, CuratorTools, ToolResult
from art_curator.queries import events as q
from tests.llm_stub import StubLlm

CHAT = "global.anthropic.claude-opus-4-6-v1"
TODAY = date(2026, 11, 11)  # a Wednesday
SEARCH = {"date_from": "2026-11-14", "date_to": "2026-11-15", "near": "Poblenou"}
NO_ROWS = ToolResult('{"total":0,"events":[],"near":null,"truncated":false}')


def thinking(text: str = "Weighing the weekend.", signature: str = "sig-1") -> dict[str, Any]:
    return {"type": "thinking", "thinking": text, "signature": signature}


def tool_use(call_id: str, name: str = "search_events", **tool_input: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": tool_input or SEARCH}


def text(reply: str) -> dict[str, Any]:
    return {"type": "text", "text": reply}


class FakeTools:
    """Answers every call with `result` and remembers what it was asked."""

    def __init__(self, result: ToolResult = NO_ROWS) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        self.calls.append((name, tool_input))
        return self.result


class NoDatabase:
    """A sessionmaker that fails the test if a tool reaches the database."""

    def __call__(self) -> Any:
        raise AssertionError("an invalid input reached the database")


def turns(stub: StubLlm, tools: Any, *texts: str) -> tuple[Conversation, list[Turn]]:
    """Run `texts` as consecutive user turns of one conversation, in one event loop."""
    curator = Curator(stub.client, tools, model=CHAT, effort="medium")
    conversation = Conversation(conversation_context(TODAY))

    async def main() -> list[Turn]:
        return [await curator.turn(conversation, t) for t in texts]

    return conversation, asyncio.run(main())


def cache_marks(request: dict[str, Any]) -> list[tuple[int, int]]:
    """(message, block) of every cache breakpoint in a request's messages."""
    return [
        (m, b)
        for m, message in enumerate(request["messages"])
        for b, block in enumerate(message["content"])
        if "cache_control" in block
    ]


def without_marks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **message,
            "content": [
                {k: v for k, v in block.items() if k != "cache_control"}
                for block in message["content"]
            ],
        }
        for message in messages
    ]


# --- the cap (CLAUDE.md § 7) ---------------------------------------------------------------------


def test_cap_holds_and_the_last_call_may_not_use_tools():
    stub = StubLlm()
    for n in range(MAX_TOOL_ROUNDS):
        stub.reply(content=[tool_use(f"call_{n}")], stop_reason="tool_use")
    stub.reply(content=[text("Here is what I found.")])
    tools = FakeTools()

    _, [turn] = turns(stub, tools, "Saturday near Poblenou?")

    assert len(tools.calls) == MAX_TOOL_ROUNDS == turn.iteration_count
    assert len(stub.requests) == MAX_TOOL_ROUNDS + 1
    assert all("tool_choice" not in r for r in stub.requests[:-1])
    assert stub.requests[-1]["tool_choice"] == {"type": "none"}
    assert turn.reply == "Here is what I found."


def test_a_call_made_after_the_cap_is_answered_but_not_run():
    stub = StubLlm()
    for n in range(MAX_TOOL_ROUNDS + 1):  # the last one despite tool_choice: none
        stub.reply(content=[tool_use(f"call_{n}")], stop_reason="tool_use")
    tools = FakeTools()

    conversation, [turn] = turns(stub, tools, "Saturday?")

    assert len(tools.calls) == MAX_TOOL_ROUNDS == turn.iteration_count
    [result] = conversation.messages[-1]["content"]
    assert result == {
        "type": "tool_result",
        "tool_use_id": f"call_{MAX_TOOL_ROUNDS}",
        "content": CAPPED,
        "is_error": True,
    }


def test_a_reply_without_tools_is_one_call_and_no_rounds():
    stub = StubLlm()
    stub.reply(content=[thinking(), text("Where will you be on Saturday?")])

    _, [turn] = turns(stub, FakeTools(), "Anything good on?")

    assert (turn.iteration_count, len(stub.requests)) == (0, 1)
    assert turn.reply == "Where will you be on Saturday?"


def test_parallel_calls_are_one_round_answered_in_one_message():
    stub = StubLlm()
    stub.reply(
        content=[tool_use("a"), tool_use("b", "get_events", ids=["x:1"])], stop_reason="tool_use"
    )
    stub.reply(content=[text("Two picks.")])

    conversation, [turn] = turns(stub, FakeTools(), "Saturday?")

    assert turn.iteration_count == 1
    results = conversation.messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert [c.name for c in turn.tool_calls] == ["search_events", "get_events"]


def test_a_reply_cut_off_after_its_calls_ends_the_turn_and_the_next_text_joins_it():
    stub = StubLlm()
    stub.reply(content=[tool_use("a")], stop_reason="max_tokens")
    stub.reply(content=[text("Sorry, again:")])

    conversation, [first, _] = turns(stub, FakeTools(), "Saturday?", "Go on")

    assert (first.iteration_count, first.stop_reason) == (1, "max_tokens")
    # The tool result and the next text are one user turn: roles still alternate.
    user = stub.requests[1]["messages"][2]
    assert [b["type"] for b in user["content"]] == ["tool_result", "text"]
    assert [m["role"] for m in conversation.messages] == ["user", "assistant", "user", "assistant"]


# --- bad input → is_error ------------------------------------------------------------------------


def test_bad_input_goes_back_to_the_model_as_an_error():
    stub = StubLlm()
    stub.reply(
        content=[tool_use("a", date_from="Saturday", date_to="2026-11-15")],
        stop_reason="tool_use",
    )
    stub.reply(content=[text("Which Saturday?")])

    _, [turn] = turns(stub, CuratorTools(NoDatabase()), "Saturday?")

    [result] = stub.requests[1]["messages"][2]["content"]
    assert result["is_error"] is True and result["tool_use_id"] == "a"
    assert result["content"].startswith("Invalid input: date_from")
    assert turn.tool_calls[0].is_error
    assert turn.reply == "Which Saturday?"


@pytest.mark.parametrize(
    ("name", "tool_input", "expected"),
    [
        ("search_events", {"date_from": "2026-11-14"}, "date_to: Field required"),
        ("search_events", {**SEARCH, "lat": 41.4}, "lat: Extra inputs are not permitted"),
        ("search_events", {**SEARCH, "kinds": ["party"]}, "kinds.0"),
        ("search_events", {**SEARCH, "radius_km": -1}, "radius_km"),
        ("search_events", {**SEARCH, "limit": 500}, "limit"),
        ("get_events", {"ids": []}, "ids"),
        ("get_events", {"ids": ["12"]}, "ids.0"),
        ("get_events", {"ids": [f"x:{n}" for n in range(q.MAX_HYDRATE + 1)]}, "ids"),
        ("remember", {"fact": "likes video"}, "No tool is called 'remember'"),
    ],
)
def test_invalid_calls_never_reach_the_database(name, tool_input, expected):
    result = asyncio.run(CuratorTools(NoDatabase())(name, tool_input))

    assert result.is_error
    assert expected in result.content


# --- cache layout (CLAUDE.md § 6) ----------------------------------------------------------------


def test_every_call_has_one_breakpoint_on_its_last_block():
    stub = StubLlm()
    stub.reply(content=[thinking(), tool_use("a")], stop_reason="tool_use")
    stub.reply(content=[text("Two picks.")])
    stub.reply(content=[text("The second one is free.")])

    conversation, _ = turns(stub, FakeTools(), "Saturday?", "Which is free?")

    assert len(stub.requests) == 3
    for request in stub.requests:
        last = len(request["messages"]) - 1
        assert cache_marks(request) == [(last, len(request["messages"][last]["content"]) - 1)]
        assert request["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # The breakpoint moves; it is never left behind in the conversation itself.
    assert cache_marks({"messages": conversation.messages}) == []


def test_each_request_starts_with_the_previous_one():
    """Caching is a prefix match: call n+1 must repeat call n byte for byte, breakpoint aside."""
    stub = StubLlm()
    stub.reply(content=[thinking(), tool_use("a")], stop_reason="tool_use")
    stub.reply(content=[thinking("Two fit.", "sig-2"), text("Two picks.")])
    stub.reply(content=[text("The second one.")])

    turns(stub, FakeTools(), "Saturday?", "Which is closer?")

    for before, after in zip(stub.requests, stub.requests[1:], strict=False):
        sent = without_marks(before["messages"])
        assert without_marks(after["messages"])[: len(sent)] == sent
        assert (after["system"], after["tools"]) == (before["system"], before["tools"])


def test_the_static_prefix_holds_no_date():
    assert TODAY.isoformat() not in SYSTEM_PROMPT
    assert "2026" not in SYSTEM_PROMPT and "2026" not in json.dumps(TOOL_DEFINITIONS)


# --- thinking, effort and history ----------------------------------------------------------------


def test_thinking_blocks_are_replayed_unchanged():
    first = [thinking("Poblenou first, then the Born.", "sig-abc=="), tool_use("a")]
    stub = StubLlm()
    stub.reply(content=first, stop_reason="tool_use")
    stub.reply(content=[text("Two picks.")])

    turns(stub, FakeTools(), "Saturday near Poblenou?")

    replayed = stub.requests[1]["messages"][1]
    assert replayed == {"role": "assistant", "content": first}


def test_every_call_asks_for_adaptive_thinking_and_the_configured_effort():
    stub = StubLlm()
    stub.reply(content=[tool_use("a")], stop_reason="tool_use")
    stub.reply(content=[text("Done.")])

    turns(stub, FakeTools(), "Saturday?")

    for request in stub.requests:
        assert request["thinking"] == {"type": "adaptive"}
        assert request["output_config"] == {"effort": "medium"}
        assert "budget_tokens" not in json.dumps(request)


def test_the_conversation_opens_with_its_context_then_the_user_text():
    stub = StubLlm()
    stub.reply(content=[text("Hello.")])

    turns(stub, FakeTools(), "Hola")

    [first] = stub.requests[0]["messages"]
    assert [b["text"] for b in first["content"]] == [conversation_context(TODAY), "Hola"]


def test_a_failed_call_leaves_the_conversation_as_it_was():
    stub = StubLlm()
    stub.reply(content=[text("Hello.")])
    stub.fail(500)
    stub.reply(content=[text("Back.")])
    curator = Curator(stub.client, FakeTools(), model=CHAT, effort="medium")
    conversation = Conversation(conversation_context(TODAY))

    async def main() -> Turn:
        await curator.turn(conversation, "Hola")
        before = list(conversation.messages)
        with pytest.raises(Exception, match="500|stub"):
            await curator.turn(conversation, "Saturday?")
        assert conversation.messages == before
        return await curator.turn(conversation, "Saturday?")

    assert asyncio.run(main()).reply == "Back."
    assert [row.error_type is not None for row in stub.calls] == [False, True, False]


def test_a_turn_is_one_trace_with_its_iteration_count():
    stub = StubLlm()
    stub.reply(content=[tool_use("a")], stop_reason="tool_use")
    stub.reply(content=[tool_use("b", "get_events", ids=["x:1"])], stop_reason="tool_use")
    stub.reply(content=[text("Done.")])

    _, [turn] = turns(stub, FakeTools(), "Saturday?")

    assert turn.iteration_count == 2
    assert {row.trace_id for row in stub.calls} == {turn.trace_id}
    [span] = [s for s in stub.spans if s.name == "chat.turn"]
    assert span.attributes["art_curator.iteration_count"] == 2
    assert [s.name for s in stub.spans if s.name.startswith("tool.")] == [
        "tool.search_events",
        "tool.get_events",
    ]


# --- prompt and tool definitions -----------------------------------------------------------------


def test_the_calendar_names_each_day_from_today():
    context = conversation_context(TODAY)

    assert "Today is Wednesday 2026-11-11" in context
    assert "Wednesday 2026-11-11 (today)" in context
    assert "Saturday 2026-11-14" in context and "Sunday 2026-11-15" in context
    assert "Tuesday 2026-11-24" in context and "2026-11-25" not in context
    assert context.count("day 2026-") == CALENDAR_DAYS + 1  # the calendar, and "Today is"


def test_tool_definitions_are_self_contained_and_match_the_queries():
    by_name = {tool["name"]: tool for tool in TOOL_DEFINITIONS}

    assert list(by_name) == ["search_events", "get_events"]
    assert "$ref" not in json.dumps(TOOL_DEFINITIONS)
    search = by_name["search_events"]["input_schema"]
    assert search["required"] == ["date_from", "date_to"]
    assert search["properties"]["kinds"]["anyOf"][0]["items"]["enum"] == list(q.KINDS)
    assert by_name["get_events"]["input_schema"]["properties"]["ids"]["maxItems"] == q.MAX_HYDRATE


# --- cli chat -----------------------------------------------------------------------------------


def test_cli_chat_runs_turns_until_end_of_input(monkeypatch):
    stub = StubLlm()
    stub.reply(content=[tool_use("a")], stop_reason="tool_use")
    stub.reply(content=[text("Prova U, at Galeria Exemple.")])
    stub.reply(content=[text("It closes on the 30th.")])
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)
    monkeypatch.setattr(cli, "CuratorTools", lambda _: FakeTools())

    async def summary(turn: Turn) -> str:
        return f"[{turn.iteration_count} tool rounds]"

    monkeypatch.setattr(cli, "_turn_summary", summary)

    result = CliRunner().invoke(
        cli.app, ["chat", "--today", "2026-11-11"], input="Saturday?\n\nUntil when?\n"
    )

    assert result.exit_code == 0, result.output
    assert "today 2026-11-11" in result.output
    assert "Prova U, at Galeria Exemple." in result.output and "[1 tool rounds]" in result.output
    assert "It closes on the 30th." in result.output and "[0 tool rounds]" in result.output
    assert len(stub.requests) == 3  # the blank line was not sent
    assert "Wednesday 2026-11-11 (today)" in stub.requests[0]["messages"][0]["content"][0]["text"]


# --- the tools over the corpus -------------------------------------------------------------------


def _run_tools(url: URL, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
    async def main() -> list[ToolResult]:
        engine = create_async_engine(url)
        tools = CuratorTools(async_sessionmaker(engine, expire_on_commit=False))
        try:
            return [await tools(name, tool_input) for name, tool_input in calls]
        finally:
            await engine.dispose()

    return asyncio.run(main())


def test_tools_answer_with_compact_json_from_the_queries(corpus_db):
    weekend = {"date_from": "2026-11-05", "date_to": "2026-11-08"}
    cut, near, hydrated = _run_tools(
        corpus_db,
        [
            ("search_events", {**weekend, "limit": 1}),
            ("search_events", {**weekend, "near": "poblenou", "radius_km": 5}),
            ("get_events", {"ids": ["x:1", "x:999"]}),
        ],
    )

    assert not (cut.is_error or near.is_error or hydrated.is_error)
    cut_rows = json.loads(cut.content)
    assert cut_rows["truncated"] and cut_rows["total"] > len(cut_rows["events"]) == 1
    near_rows = json.loads(near.content)
    assert near_rows["near"]["name"] == "el Poblenou"
    # Galeria Exemple is ~3.5 km away, so its shows are in and each carries its distance.
    assert near_rows["events"] and all(row["distance_km"] <= 5 for row in near_rows["events"])
    details = json.loads(hydrated.content)
    assert [d["id"] for d in details["events"]] == ["x:1"] and details["not_found"] == ["x:999"]
    assert ", " not in cut.content  # compact separators


def test_a_place_that_names_nothing_is_an_error_the_model_can_read(corpus_db):
    weekend = {"date_from": "2026-11-05", "date_to": "2026-11-08"}
    [result] = _run_tools(corpus_db, [("search_events", {**weekend, "near": "Atlantis"})])

    assert result.is_error and "'Atlantis'" in result.content


def test_dates_the_wrong_way_round_are_an_error(corpus_db):
    [result] = _run_tools(
        corpus_db, [("search_events", {"date_from": "2026-11-08", "date_to": "2026-11-05"})]
    )

    assert result.is_error and "after" in result.content
