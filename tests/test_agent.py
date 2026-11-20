"""The curator's loop, tools and prompt (PLAN.md § 8, PR 28), over a stubbed model, and the
conversations it keeps (PR 29).

What PR 28 must show without calling a model: the cap holds, a bad tool input comes back as
`is_error`, every call carries one cache breakpoint on its last block, and thinking blocks are
replayed exactly as they came. The tools' database half runs against `tests/corpus.py`.

PR 29: a turn replayed from the database starts with the turn before it, byte for byte, and sends
what an in-memory conversation would; a conversation ends when idle or when its day is over; and
`cli chat` resumes, starts over and rates replies.

PR 30: a fact remembered in one conversation is in the next one's context block, a correction
supersedes the fact it names and nothing else, and forget leaves no rows of the user.
"""

import asyncio
import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import pytest
from alembic import command
from anthropic import APIStatusError
from sqlalchemy import text as sql_text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from typer.testing import CliRunner

from art_curator import cli
from art_curator.agent.loop import (
    CAPPED,
    MAX_TOOL_ROUNDS,
    Conversation,
    Curator,
    TextDelta,
    ToolRunning,
    Turn,
    TurnDone,
    TurnEvent,
)
from art_curator.agent.profile import Fact
from art_curator.agent.prompt import (
    CALENDAR_DAYS,
    SYSTEM_PROMPT,
    conversation_context,
)
from art_curator.agent.store import IDLE, ConversationStore
from art_curator.agent.tools import TOOL_DEFINITIONS, CuratorTools, ToolResult
from art_curator.config import get_settings
from art_curator.db.models import PROFILE_TOPICS
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.queries import events as q
from tests.db import alembic_config, run_sql
from tests.llm_stub import StubLlm

CHAT = "global.anthropic.claude-opus-4-6-v1"
TODAY = date(2026, 11, 11)  # a Wednesday
SEARCH = {"date_from": "2026-11-14", "date_to": "2026-11-15", "near": "Poblenou"}
NOBODY = 0  # the user a read-only tool runs for: it reads nothing of theirs
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

    _, [turn] = turns(stub, CuratorTools(NoDatabase(), NOBODY), "Saturday?")

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
        ("remember", {"fact": "Likes video art."}, "topic: Field required"),
        ("remember", {"topic": "mood", "fact": "Likes video art."}, "topic"),
        ("remember", {"topic": "media", "fact": "x" * 201}, "fact"),
        ("remember", {"topic": "media", "fact": "Likes video.", "replaces": "x"}, "replaces"),
        ("forget", {}, "No tool is called 'forget'"),
    ],
)
def test_invalid_calls_never_reach_the_database(name, tool_input, expected):
    result = asyncio.run(CuratorTools(NoDatabase(), NOBODY)(name, tool_input))

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


# --- streamed turns (PR 33) ---------------------------------------------------------------------

LOOKING = [thinking(), text("Let me look."), tool_use("a")]
ANSWER = "Two shows near Poblenou on Saturday."


def streamed_turn(stub: StubLlm, *texts: str) -> tuple[Conversation, list[list[TurnEvent]]]:
    """`turns`, streamed: each turn's events, as `/chat` reads them."""
    curator = Curator(stub.client, FakeTools(), model=CHAT, effort="medium")
    conversation = Conversation(conversation_context(TODAY))

    async def main() -> list[list[TurnEvent]]:
        return [[e async for e in curator.events(conversation, t)] for t in texts]

    return conversation, asyncio.run(main())


def test_a_streamed_turn_yields_tools_as_they_run_and_text_as_it_comes():
    streamed, unstreamed = StubLlm(), StubLlm()
    for queue in (streamed.stream_reply, unstreamed.reply):
        queue(content=LOOKING, stop_reason="tool_use", model=CHAT)
        queue(ANSWER, model=CHAT)

    conversation, [events] = streamed_turn(streamed, "Saturday near Poblenou?")
    same, [turn] = turns(unstreamed, FakeTools(), "Saturday near Poblenou?")

    running = events.index(ToolRunning("search_events"))
    before, after = events[:running], events[running + 1 : -1]
    # The words before the tool call arrive too; the reply is the last response's text.
    assert "".join(e.text for e in before) == "Let me look."
    assert "".join(e.text for e in after) == ANSWER
    assert all(isinstance(e, TextDelta) for e in before + after) and len(after) > 1
    done = events[-1]
    assert isinstance(done, TurnDone)
    assert replace(done.turn, trace_id=None) == replace(turn, trace_id=None)
    # The same requests, and the same history kept, as the turn unstreamed.
    assert streamed.requests == unstreamed.requests
    assert conversation.messages == same.messages
    assert all(p.endswith("/invoke-with-response-stream") for p in streamed.paths)
    assert [row.error_type for row in streamed.calls] == [None, None]


def test_a_streamed_turn_the_reader_abandons_keeps_nothing_and_records_its_calls():
    # A client that disconnects mid-reply: /chat stops reading and closes the generator.
    stub = StubLlm()
    stub.stream_reply(content=LOOKING, stop_reason="tool_use", model=CHAT, input_tokens=50)
    stub.stream_reply(ANSWER, model=CHAT, input_tokens=70)
    curator = Curator(stub.client, FakeTools(), model=CHAT, effort="medium")
    conversation = Conversation(conversation_context(TODAY))

    async def main() -> None:
        events = curator.events(conversation, "Saturday near Poblenou?")
        async for event in events:
            if event == ToolRunning("search_events"):
                break
        assert isinstance(await anext(events), TextDelta)  # the answer has started
        await events.aclose()

    asyncio.run(main())
    assert conversation.messages == []
    assert [(row.error_type, row.input_tokens) for row in stub.calls] == [
        (None, 50),
        ("GeneratorExit", 70),
    ]


def test_a_stream_that_breaks_fails_the_turn_and_keeps_nothing():
    stub = StubLlm()
    stub.stream_reply(ANSWER, model=CHAT, break_after=3, input_tokens=70)

    with pytest.raises(APIStatusError):
        streamed_turn(stub, "Saturday?")
    [row] = stub.calls
    assert (row.input_tokens, row.error_type is not None) == (70, True)


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

    assert list(by_name) == ["search_events", "get_events", "remember"]
    assert "$ref" not in json.dumps(TOOL_DEFINITIONS)
    search = by_name["search_events"]["input_schema"]
    assert search["required"] == ["date_from", "date_to"]
    assert search["properties"]["kinds"]["anyOf"][0]["items"]["enum"] == list(q.KINDS)
    assert by_name["get_events"]["input_schema"]["properties"]["ids"]["maxItems"] == q.MAX_HYDRATE
    remember = by_name["remember"]["input_schema"]
    assert remember["required"] == ["topic", "fact"]
    assert remember["properties"]["topic"]["enum"] == list(PROFILE_TOPICS)


# --- the tools over the corpus -------------------------------------------------------------------


def _run_tools(url: URL, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
    async def main() -> list[ToolResult]:
        engine = create_async_engine(url)
        tools = CuratorTools(async_sessionmaker(engine, expire_on_commit=False), NOBODY)
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


# --- conversations kept in the database (PR 29) -------------------------------------------------


@pytest.fixture(scope="module")
def conversations_db(fresh_database) -> Iterator[URL]:
    """A migrated database the tests below share, each as its own user."""
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        yield url


def kept(url: URL, stub: StubLlm, tools: Any, user: str, *texts: str) -> list[Turn]:
    """Run `texts` as consecutive turns of `user`'s conversation, each one replayed from the
    database and saved back, as a server would."""

    async def main() -> list[Turn]:
        engine = create_async_engine(url)
        store = ConversationStore(async_sessionmaker(engine, expire_on_commit=False))
        curator = Curator(stub.client, tools, model=CHAT, effort="medium")
        try:
            user_id = await store.user("test", user)
            done = []
            for t in texts:
                stored = await store.current(user_id, TODAY)
                done.append(await curator.turn(stored.conversation, t))
                await store.save(stored, done[-1])
            return done
        finally:
            await engine.dispose()

    return asyncio.run(main())


def serialized(request: dict[str, Any]) -> str:
    """A request's messages as JSON, without the breakpoint. `json.loads` keeps an object's keys
    in the order they came, so equal strings mean equal bytes on the wire up to whitespace, which
    dict equality would not check."""
    return json.dumps(without_marks(request["messages"]))


def two_turns(stub: StubLlm) -> None:
    """Queue a search, its answer, and a reply to a second question. The search input's keys are
    in an order `jsonb` would not keep: it sorts them by length."""
    stub.reply(content=[thinking(), tool_use("a")], stop_reason="tool_use")
    stub.reply(content=[thinking("Two fit.", "sig-2"), text("Two picks.")])
    stub.reply(content=[text("The second one.")])


def test_a_resumed_turn_starts_with_the_last_request_byte_for_byte(conversations_db):
    stub = StubLlm()
    two_turns(stub)

    kept(conversations_db, stub, FakeTools(), "bytes", "Saturday?", "Which is closer?")

    assert list(SEARCH) == ["date_from", "date_to", "near"]
    for before, after in zip(stub.requests, stub.requests[1:], strict=False):
        assert serialized(after).startswith(serialized(before)[:-1])  # up to the closing ]
    # Replayed from the database, every request is the one an in-memory conversation sends.
    memory = StubLlm()
    two_turns(memory)
    turns(memory, FakeTools(), "Saturday?", "Which is closer?")
    assert [serialized(r) for r in stub.requests] == [serialized(r) for r in memory.requests]


def test_each_message_is_a_row_with_its_turn_and_the_reply_its_rounds(conversations_db):
    stub = StubLlm()
    two_turns(stub)

    first, second = kept(conversations_db, stub, FakeTools(), "rows", "Saturday?", "Closer?")

    [rows] = run_sql(
        conversations_db,
        "SELECT m.position, m.role, m.trace_id, m.iteration_count FROM messages m "
        "JOIN conversations c ON c.id = m.conversation_id JOIN users u ON u.id = c.user_id "
        "WHERE u.external_id = 'rows' ORDER BY m.position",
    )
    assert rows == [
        (0, "user", first.trace_id, None),
        (1, "assistant", first.trace_id, None),
        (2, "user", first.trace_id, None),  # the tool results
        (3, "assistant", first.trace_id, 1),
        (4, "user", second.trace_id, None),
        (5, "assistant", second.trace_id, 0),
    ]


def test_a_cut_off_turn_is_completed_by_the_next_text(conversations_db):
    stub = StubLlm()
    stub.reply(content=[tool_use("a")], stop_reason="max_tokens")
    stub.reply(content=[text("Sorry, again:")])

    kept(conversations_db, stub, FakeTools(), "cut", "Saturday?", "Go on")

    # The stored tool results grew by the next text, which went out as one user turn.
    [[(blocks,)]] = run_sql(
        conversations_db,
        "SELECT m.content::text FROM messages m JOIN conversations c ON c.id = m.conversation_id "
        "JOIN users u ON u.id = c.user_id WHERE u.external_id = 'cut' AND m.position = 2",
    )
    assert [b["type"] for b in json.loads(blocks)] == ["tool_result", "text"]
    assert without_marks(stub.requests[1]["messages"])[2]["content"] == json.loads(blocks)


def test_a_conversation_ends_after_idle_hours_or_when_its_day_is_over(conversations_db):
    async def main() -> list[int]:
        engine = create_async_engine(conversations_db)
        store = ConversationStore(async_sessionmaker(engine, expire_on_commit=False))
        try:
            user_id = await store.user("test", "expiry")
            same = [(await store.current(user_id, TODAY)).id for _ in range(2)]
            async with engine.begin() as conn:
                await conn.execute(
                    sql_text(
                        "UPDATE conversations SET last_turn_at = now() - CAST(:idle AS interval) "
                        "WHERE user_id = :user AND ended_at IS NULL"
                    ),
                    {"idle": IDLE + timedelta(minutes=1), "user": user_id},
                )
            idle = (await store.current(user_id, TODAY)).id
            tomorrow = await store.current(user_id, TODAY + timedelta(days=1))
            assert tomorrow.conversation.context == conversation_context(TODAY + timedelta(days=1))
            return [*same, idle, tomorrow.id]
        finally:
            await engine.dispose()

    first, again, after_idle, next_day = asyncio.run(main())

    assert first == again != after_idle != next_day
    [rows] = run_sql(
        conversations_db,
        "SELECT c.id, c.ended_at IS NOT NULL FROM conversations c JOIN users u ON u.id = c.user_id"
        " WHERE u.external_id = 'expiry' ORDER BY c.id",
    )
    assert rows == [(first, True), (after_idle, True), (next_day, False)]


@pytest.fixture
def chat(conversations_db, monkeypatch) -> Iterator[Callable[..., Any]]:
    """`chat(input, user=...)` runs `cli chat` on the conversations database, the model stubbed;
    `chat.stub` holds what it sent, `chat.url` is the database."""
    monkeypatch.setenv("DATABASE_URL", conversations_db.render_as_string(hide_password=False))
    stub = StubLlm()
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)
    monkeypatch.setattr(cli, "CuratorTools", lambda *_: FakeTools())

    async def summary(turn: Turn) -> str:  # its rows went to the stub, not the database
        return f"[{turn.iteration_count} tool rounds]"

    monkeypatch.setattr(cli, "_turn_summary", summary)

    def _chat(input: str, *, user: str) -> Any:
        _clear_caches()  # each run is a new process, with its own engine
        return CliRunner().invoke(
            cli.app, ["chat", "--today", "2026-11-11", "--user", user], input=input
        )

    _chat.stub = stub
    _chat.url = conversations_db
    yield _chat
    _clear_caches()


def _clear_caches() -> None:
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


def test_cli_chat_runs_turns_until_end_of_input(chat):
    chat.stub.reply(content=[tool_use("a")], stop_reason="tool_use")
    chat.stub.reply(content=[text("Prova U, at Galeria Exemple.")])
    chat.stub.reply(content=[text("It closes on the 30th.")])

    result = chat("Saturday?\n\nUntil when?\n", user="turns")

    assert result.exit_code == 0, result.output
    assert "today 2026-11-11" in result.output and "Resuming" not in result.output
    assert "Prova U, at Galeria Exemple." in result.output and "[1 tool rounds]" in result.output
    assert "It closes on the 30th." in result.output and "[0 tool rounds]" in result.output
    assert len(chat.stub.requests) == 3  # the blank line was not sent
    first = chat.stub.requests[0]["messages"][0]["content"][0]["text"]
    assert "Wednesday 2026-11-11 (today)" in first


def test_cli_chat_resumes_the_conversation_and_new_starts_another(chat):
    for reply in ("Two picks.", "The second one.", "Hola!"):
        chat.stub.reply(content=[text(reply)])

    first = chat("Saturday?\n", user="resume")
    resumed = chat("Which is closer?\n/new\nHola\n", user="resume")

    assert first.exit_code == resumed.exit_code == 0, resumed.output
    assert "Resuming conversation" in resumed.output and "(2 messages)" in resumed.output
    assert "New conversation" in resumed.output
    assert [len(r["messages"]) for r in chat.stub.requests] == [1, 3, 1]
    assert serialized(chat.stub.requests[1]).startswith(serialized(chat.stub.requests[0])[:-1])


def test_cli_chat_rates_the_last_reply_once(chat):
    chat.stub.reply(content=[text("Two picks.")])

    result = chat("/good\nSaturday?\n/good\n/bad\n/what\n", user="rate")

    assert result.exit_code == 0, result.output
    assert "No reply to rate yet." in result.output and result.output.count("Noted.") == 2
    assert result.output.count(cli.CHAT_HELP) == 2  # on start, and for an unknown command
    [rows] = run_sql(
        chat.url,
        "SELECT f.rating, m.iteration_count FROM feedback f JOIN messages m ON m.id = f.message_id"
        " JOIN conversations c ON c.id = m.conversation_id JOIN users u ON u.id = c.user_id"
        " WHERE u.external_id = 'rate'",
    )
    assert rows == [("bad", 0)]


# --- the profile (PR 30) -------------------------------------------------------------------------


def test_the_context_lists_the_profile_with_the_ids_remember_corrects_by():
    facts = [Fact(3, "media", "Likes video art."), Fact(7, "area", "Starts from Poblenou.")]

    assert "Remembered about this person: nothing yet." in conversation_context(TODAY)
    context = conversation_context(TODAY, facts)
    assert "[3] media: Likes video art.\n[7] area: Starts from Poblenou.\n</context>" in context


def remembering(url: URL, stub: StubLlm, user: str, text: str) -> tuple[Turn, str]:
    """Run `text` as the first turn of a new conversation of `user`, with the real tools bound to
    them; returns the turn and the context block its conversation opened with."""

    async def main() -> tuple[Turn, str]:
        engine = create_async_engine(url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        store = ConversationStore(sessions)
        try:
            user_id = await store.user("test", user)
            tools = CuratorTools(sessions, user_id)
            curator = Curator(stub.client, tools, model=CHAT, effort="medium")
            await store.close(user_id)
            stored = await store.current(user_id, TODAY)
            turn = await curator.turn(stored.conversation, text)
            await store.save(stored, turn)
            return turn, stored.conversation.context
        finally:
            await engine.dispose()

    return asyncio.run(main())


def tool_result(request: dict[str, Any]) -> dict[str, Any]:
    """The first tool result of the last message of `request`, parsed."""
    result = request["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and not result.get("is_error"), result
    return json.loads(result["content"])


def facts_of(url: URL, user: str) -> list[tuple]:
    [rows] = run_sql(
        url,
        "SELECT f.id, f.topic, f.fact, f.superseded_by FROM profile_facts f "
        f"JOIN users u ON u.id = f.user_id WHERE u.external_id = '{user}' ORDER BY f.id",
    )
    return rows


def test_a_fact_remembered_in_one_conversation_opens_the_next(conversations_db):
    stub = StubLlm()
    likes = {"topic": "media", "fact": "Likes video art."}
    stub.reply(content=[tool_use("a", "remember", **likes)], stop_reason="tool_use")
    stub.reply(content=[text("Video it is.")])
    _, first = remembering(conversations_db, stub, "taste", "I love video art")
    remembered = tool_result(stub.requests[1])

    stub.reply(content=[text("Hola!")])
    _, second = remembering(conversations_db, stub, "taste", "Hola")

    assert "Remembered about this person: nothing yet." in first
    assert f"[{remembered['id']}] media: Likes video art." in second
    assert remembered == {"id": remembered["id"], "replaced": None}
    # The new conversation's first request carries it: the model sees it from the first turn.
    assert stub.requests[-1]["messages"][0]["content"][0]["text"] == second


def test_a_correction_supersedes_the_fact_it_names(conversations_db):
    stub = StubLlm()
    stub.reply(
        content=[tool_use("a", "remember", topic="area", fact="Starts from Gràcia.")],
        stop_reason="tool_use",
    )
    stub.reply(content=[text("Gràcia, noted.")])
    remembering(conversations_db, stub, "moved", "I live in Gràcia")
    old = tool_result(stub.requests[1])["id"]
    moved = {"topic": "area", "fact": "Starts from Poblenou.", "replaces": old}
    stub.reply(content=[tool_use("b", "remember", **moved)], stop_reason="tool_use")
    stub.reply(content=[text("Poblenou, then.")])
    remembering(conversations_db, stub, "moved", "I moved to Poblenou")
    new = tool_result(stub.requests[3])["id"]

    stub.reply(content=[text("Hola!")])
    _, context = remembering(conversations_db, stub, "moved", "Hola")

    assert f"[{new}] area: Starts from Poblenou." in context and "Gràcia" not in context
    # Append-only: the old fact is kept, pointing at the one that replaced it.
    assert facts_of(conversations_db, "moved") == [
        (old, "area", "Starts from Gràcia.", new),
        (new, "area", "Starts from Poblenou.", None),
    ]


def test_a_correction_of_no_live_fact_is_an_error_and_writes_nothing(conversations_db):
    async def main() -> list[ToolResult]:
        engine = create_async_engine(conversations_db)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        store = ConversationStore(sessions)
        try:
            mine = CuratorTools(sessions, await store.user("test", "corrects"))
            theirs = CuratorTools(sessions, await store.user("test", "someone else"))
            other = json.loads((await theirs("remember", {**fact, "fact": "Theirs."})).content)
            first = json.loads((await mine("remember", fact)).content)["id"]
            await mine("remember", {**fact, "replaces": first})
            return [
                await mine("remember", {**fact, "replaces": first}),  # already replaced
                await mine("remember", {**fact, "replaces": other["id"]}),  # another user's
                await mine("remember", {**fact, "replaces": 10**9}),  # no such fact
            ]
        finally:
            await engine.dispose()

    fact = {"topic": "budget", "fact": "Prefers free entry."}
    replaced, theirs, missing = asyncio.run(main())

    assert all(result.is_error for result in (replaced, theirs, missing))
    successor = facts_of(conversations_db, "corrects")[1][0]
    assert f"already replaced by fact {successor}" in replaced.content
    assert "No remembered fact has id" in theirs.content and "No remembered" in missing.content
    assert len(facts_of(conversations_db, "corrects")) == 2  # the refused facts were rolled back
    assert [row[3] for row in facts_of(conversations_db, "someone else")] == [None]


USER_ROWS = {
    "profile_facts": "SELECT count(*) FROM profile_facts WHERE user_id = {user}",
    "conversations": "SELECT count(*) FROM conversations WHERE user_id = {user}",
    "messages": "SELECT count(*) FROM messages m JOIN conversations c "
    "ON c.id = m.conversation_id WHERE c.user_id = {user}",
    "feedback": "SELECT count(*) FROM feedback f JOIN messages m ON m.id = f.message_id "
    "JOIN conversations c ON c.id = m.conversation_id WHERE c.user_id = {user}",
}


def rows_of(url: URL, user_id: int) -> dict[str, int]:
    counts = run_sql(url, *(sql.format(user=user_id) for sql in USER_ROWS.values()))
    return {table: rows[0][0] for table, rows in zip(USER_ROWS, counts, strict=True)}


def test_forget_leaves_no_rows_of_the_user(conversations_db):
    stub = StubLlm()
    for user in ("forgotten", "kept"):
        stub.reply(
            content=[tool_use("a", "remember", topic="themes", fact="Likes archives.")],
            stop_reason="tool_use",
        )
        stub.reply(content=[text("Archives, noted.")])
        remembering(conversations_db, stub, user, "I like archives")

    async def main() -> tuple[int, int]:
        engine = create_async_engine(conversations_db)
        store = ConversationStore(async_sessionmaker(engine, expire_on_commit=False))
        try:
            forgotten, kept = [await store.user("test", u) for u in ("forgotten", "kept")]
            for user_id in (forgotten, kept):
                assert await store.rate((await store.current(user_id, TODAY)).id, "good")
            await store.forget(forgotten)
            assert await store.remembered(forgotten) == []
            return forgotten, kept
        finally:
            await engine.dispose()

    forgotten, kept = asyncio.run(main())

    assert rows_of(conversations_db, forgotten) == dict.fromkeys(USER_ROWS, 0)
    assert rows_of(conversations_db, kept) == {
        "profile_facts": 1,
        "conversations": 1,
        "messages": 4,
        "feedback": 1,
    }


def test_cli_chat_shows_and_forgets_the_profile(chat):
    chat.stub.reply(content=[text("Video, then.")])
    chat("/profile\n", user="forget")  # creates the user
    run_sql(
        chat.url,
        "INSERT INTO profile_facts (user_id, topic, fact) SELECT id, 'media', 'Likes video art.' "
        "FROM users WHERE external_id = 'forget'",
    )

    result = chat("/profile\n/new\nSaturday?\n/forget\n/profile\n", user="forget")

    assert result.exit_code == 0, result.output
    assert "] media: Likes video art." in result.output and "Forgotten" in result.output
    assert result.output.endswith("you> Nothing yet.\n\nyou> ")
    assert "media: Likes video art." in chat.stub.requests[0]["messages"][0]["content"][0]["text"]
    [[(user_id,)]] = run_sql(chat.url, "SELECT id FROM users WHERE external_id = 'forget'")
    # Only the conversation /forget opened to carry on with, empty and knowing nothing.
    assert rows_of(chat.url, user_id) == {
        "profile_facts": 0,
        "conversations": 1,
        "messages": 0,
        "feedback": 0,
    }
