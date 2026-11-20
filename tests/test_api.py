"""The HTTP API (PLAN.md § 8, PR 34) and `cli chat --api`, over a stubbed model.

PR 34 must show the same conversation in-process and over HTTP: a turn begun by `cli chat` and
carried on by `cli chat --api` sends what two in-process turns would, byte for byte. Requests go
to the app in-process (`httpx2.ASGITransport`), so nothing listens on a port.

The tests without a database swap the conversation store for one in memory: the token, request
validation, the event stream and a failed turn. The rest run on a migrated database (CI).
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from datetime import date
from typing import Any

import httpx2
import pytest
from alembic import command
from sqlalchemy.engine import URL
from typer.testing import CliRunner

from art_curator import cli
from art_curator.agent.loop import Conversation, Turn
from art_curator.agent.prompt import conversation_context
from art_curator.agent.store import StoredConversation
from art_curator.api import main as api_main
from art_curator.api.main import app, get_store, lifespan
from art_curator.clients.api import (
    ChatFailed,
    CuratorApi,
    ReplyDone,
    TextChunk,
    ToolStatus,
    _server_sent_events,
)
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.llm.client import get_llm_client
from tests.db import alembic_config, run_sql
from tests.llm_stub import StubLlm
from tests.test_agent import (
    ANSWER,
    CHAT,
    LOOKING,
    TODAY,
    USER_ROWS,
    FakeTools,
    NoDatabase,
    _clear_caches,
    rows_of,
    serialized,
    text,
    thinking,
    tool_use,
)

TOKEN = "test-token"
BASE = "http://curator.test"


class MemoryStore:
    """The part of `ConversationStore` that `/chat` uses, in memory: one conversation a user."""

    def __init__(self) -> None:
        self.users: dict[tuple[str, str], int] = {}
        self.conversations: dict[int, StoredConversation] = {}
        self.saved: list[Turn] = []

    async def user(self, client: str, external_id: str) -> int:
        return self.users.setdefault((client, external_id), len(self.users) + 1)

    async def current(self, user_id: int, today: date) -> StoredConversation:
        if user_id not in self.conversations:
            conversation = Conversation(conversation_context(today))
            self.conversations[user_id] = StoredConversation(user_id, conversation, saved=0)
        return self.conversations[user_id]

    async def save(self, stored: StoredConversation, turn: Turn) -> int:
        stored.saved = len(stored.conversation.messages)
        self.saved.append(turn)
        return 100 + len(self.saved)


@pytest.fixture
def api(monkeypatch) -> Iterator[StubLlm]:
    """The app with API_TOKEN set, the model stubbed and the tools faked; yields the stub."""
    monkeypatch.setenv("API_TOKEN", TOKEN)
    _clear_caches()
    stub = StubLlm()
    app.dependency_overrides[get_llm_client] = lambda: stub.client
    monkeypatch.setattr(api_main, "CuratorTools", lambda *_: FakeTools())
    yield stub
    app.dependency_overrides.clear()
    _clear_caches()


@pytest.fixture
def memory(api) -> MemoryStore:
    """No database: conversations in memory, and a test fails if anything reaches Postgres."""
    store = MemoryStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_sessionmaker] = NoDatabase
    return store


def over_http[T](main: Callable[[CuratorApi], Awaitable[T]]) -> T:
    """Run `main` with a client of the app, in one event loop, then drop the loop's engine."""

    async def run() -> T:
        transport = httpx2.ASGITransport(app=app)
        try:
            async with CuratorApi(BASE, TOKEN, client="test", transport=transport) as client:
                return await main(client)
        finally:
            if get_sessionmaker not in app.dependency_overrides:
                await get_engine().dispose()
            _clear_caches()

    return asyncio.run(run())


def chat_events(user: str, *texts: str) -> list[list[Any]]:
    """Each text as `user`'s next message over HTTP: the events of each reply."""

    async def main(client: CuratorApi) -> list[list[Any]]:
        return [[e async for e in client.chat(user, t, today=TODAY)] for t in texts]

    return over_http(main)


def raw(method: str, path: str, **kwargs: Any) -> httpx2.Response:
    """One request to the app as it comes back, for what `CuratorApi` would raise on."""

    async def main() -> httpx2.Response:
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE) as c:
            return await c.request(method, path, **kwargs)

    return asyncio.run(main())


class InProcessApi(CuratorApi):
    """What `cli chat --api` talks to in these tests: the app in-process, whose engine belongs to
    the command's event loop and is dropped with it, as a server's would be at shutdown."""

    def __init__(self, base_url: str, token: str, *, client: str) -> None:
        super().__init__(base_url, token, client=client, transport=httpx2.ASGITransport(app=app))

    async def aclose(self) -> None:
        await super().aclose()
        await get_engine().dispose()


BEARER = {"Authorization": f"Bearer {TOKEN}"}
ROUTES = [
    ("POST", "/chat"),
    ("POST", "/feedback"),
    ("GET", "/users/test/someone/profile"),
    ("DELETE", "/users/test/someone/profile"),
    ("DELETE", "/users/test/someone/conversation"),
]


# --- the token -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "path"), ROUTES)
def test_every_route_refuses_a_request_without_the_token(memory, method, path):
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}):
        response = raw(method, path, headers=headers)
        assert response.status_code == 401, (headers, response.text)
        assert response.headers["www-authenticate"] == "Bearer"


def test_there_are_no_unguarded_docs(memory):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert raw("GET", path).status_code in (401, 404), path


def test_without_api_token_the_api_will_not_start_or_answer(memory, monkeypatch):
    monkeypatch.delenv("API_TOKEN")
    _clear_caches()

    async def start() -> None:
        async with lifespan(app):
            pass

    with pytest.raises(RuntimeError, match="API_TOKEN"):
        asyncio.run(start())
    assert raw("GET", "/users/test/someone/profile", headers=BEARER).status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {"client": "test", "user": "u"},  # no text
        {"client": "test", "user": "u", "text": ""},
        {"client": "", "user": "u", "text": "Hola"},
        {"client": "test", "user": "u", "text": "Hola", "model": "other"},  # unknown field
        {"client": "test", "user": "u", "text": "Hola", "today": "Saturday"},
    ],
)
def test_a_malformed_message_is_refused_before_any_model_call(api, memory, body):
    response = raw("POST", "/chat", headers=BEARER, json=body)

    assert response.status_code == 422
    assert api.requests == [] and memory.users == {}


# --- the event stream ----------------------------------------------------------------------------


def test_a_turn_streams_its_tools_and_text_then_done(api, memory):
    api.stream_reply(content=LOOKING, stop_reason="tool_use", model=CHAT)
    api.stream_reply(ANSWER, model=CHAT)

    [events] = chat_events("streamer", "Saturday near Poblenou?")

    running = events.index(ToolStatus("search_events"))
    before, after, done = events[:running], events[running + 1 : -1], events[-1]
    assert "".join(e.text for e in before) == "Let me look."
    assert "".join(e.text for e in after) == ANSWER
    assert all(isinstance(e, TextChunk) for e in before + after) and len(after) > 1
    assert done == ReplyDone(message_id=101, iteration_count=1, stop_reason="end_turn")
    # Streamed from the model, never thinking, and saved just before `done`.
    assert all(p.endswith("/invoke-with-response-stream") for p in api.paths)
    assert "Weighing" not in repr(events)
    [turn] = memory.saved
    assert turn.reply == ANSWER and turn.iteration_count == 1
    [stored] = memory.conversations.values()
    assert stored.saved == len(stored.conversation.messages) == 4


def test_a_failed_turn_ends_in_an_error_event_and_saves_nothing(api, memory):
    api.stream_reply(ANSWER, model=CHAT, break_after=3)

    with pytest.raises(ChatFailed, match="APIStatusError"):
        chat_events("broken", "Saturday?")
    assert memory.saved == []
    [stored] = memory.conversations.values()
    assert stored.conversation.messages == []
    [row] = api.calls  # recorded all the same
    assert row.error_type is not None


def test_the_reader_skips_keep_alives_and_joins_split_data():
    body = (
        b": ping\n\n"
        b'event: text\ndata: {"text":\ndata:  "Hola"}\n\n'
        b"event: status\r\n"
        b'data: {"tool": "get_events"}\r\n\r\n'
        b": ping\n\n"
    )

    async def main() -> list[tuple[str, Any]]:
        response = httpx2.Response(200, content=body)
        return [event async for event in _server_sent_events(response)]

    assert asyncio.run(main()) == [("text", {"text": "Hola"}), ("status", {"tool": "get_events"})]


def test_cli_chat_over_the_api_prints_the_reply_as_it_streams(api, memory, monkeypatch):
    api.stream_reply(content=LOOKING, stop_reason="tool_use", model=CHAT)
    api.stream_reply(ANSWER, model=CHAT)
    api.stream_reply(ANSWER, model=CHAT, break_after=3)
    monkeypatch.setattr(cli, "CuratorApi", InProcessApi)

    result = CliRunner().invoke(
        cli.app, ["chat", "--api", BASE, "--user", "u"], input="Saturday?\nAgain?\n"
    )

    assert result.exit_code == 0, result.output
    assert f"Let me look.\n  · search_events\n{ANSWER}\n\n  [1 tool rounds · message 101]" in (
        result.output
    )
    assert "error: ChatFailed: APIStatusError" in result.output
    assert len(memory.saved) == 1


def test_cli_chat_over_the_api_wants_the_token(memory, monkeypatch):
    monkeypatch.delenv("API_TOKEN")
    _clear_caches()

    result = CliRunner().invoke(cli.app, ["chat", "--api", BASE], input="Hola\n")

    assert result.exit_code == 2 and "API_TOKEN is not set" in result.output


# --- on the database (CI) ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conversations_db(fresh_database) -> Iterator[URL]:
    """A migrated database the tests below share, each as its own users."""
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@pytest.fixture
def db_api(api, conversations_db, monkeypatch) -> Iterator[StubLlm]:
    """The app on the conversations database; yields the model stub, `.url` the database."""
    monkeypatch.setenv("DATABASE_URL", conversations_db.render_as_string(hide_password=False))
    _clear_caches()
    api.url = conversations_db
    yield api


def queue_two_turns(stub: StubLlm, *, second_streamed: bool) -> None:
    """A search, its answer, and a reply to a second question (`tests.test_agent.two_turns`),
    the first turn unstreamed as `cli chat` calls the model, the second as `/chat` does if
    `second_streamed`."""
    stub.reply(content=[thinking(), tool_use("a")], stop_reason="tool_use")
    stub.reply(content=[thinking("Two fit.", "sig-2"), text("Two picks.")])
    second = stub.stream_reply if second_streamed else stub.reply
    second(content=[text("The second one.")])


def test_the_same_conversation_in_process_and_over_http(db_api, monkeypatch):
    monkeypatch.setattr(cli, "get_llm_client", lambda: db_api.client)
    monkeypatch.setattr(cli, "CuratorTools", lambda *_: FakeTools())
    monkeypatch.setattr(cli, "_turn_summary", _summary)
    monkeypatch.setattr(cli, "CuratorApi", InProcessApi)

    def run_cli(user: str, input: str, *api: str) -> Any:
        _clear_caches()  # each run is a new process, with its own engine
        args = ["chat", "--today", TODAY.isoformat(), "--user", user, *api]
        return CliRunner().invoke(cli.app, args, input=input)

    queue_two_turns(db_api, second_streamed=True)
    local = run_cli("both", "Saturday?\n")
    remote = run_cli("both", "Which is closer?\n", "--api", BASE)
    queue_two_turns(db_api, second_streamed=False)
    run_cli("memory", "Saturday?\nWhich is closer?\n")

    assert local.exit_code == remote.exit_code == 0, remote.output
    assert "Two picks." in local.output
    assert "The second one." in remote.output and "[0 tool rounds · message " in remote.output
    over_api, in_process = db_api.requests[:3], db_api.requests[3:]
    # The HTTP turn carried on the conversation `cli chat` began, byte for byte...
    assert serialized(over_api[2]).startswith(serialized(over_api[1])[:-1])
    # ...and sent what a second in-process turn sends.
    assert [serialized(r) for r in over_api] == [serialized(r) for r in in_process]
    assert db_api.paths[2].endswith("/invoke-with-response-stream")


async def _summary(turn: Turn) -> str:  # its rows went to the stub, not the database
    return f"[{turn.iteration_count} tool rounds]"


def test_feedback_rates_the_users_own_replies_only(db_api):
    for _ in range(2):
        db_api.stream_reply(content=[text("Two picks.")])
    mine = chat_events("rater", "Saturday?")[0][-1]
    theirs = chat_events("other", "Hi")[0][-1]

    async def main(client: CuratorApi) -> list[int]:
        await client.feedback("rater", mine.message_id, "good")
        await client.feedback("rater", mine.message_id, "bad")  # replaces the first
        refused = []
        for user, message_id in [
            ("other", mine.message_id),  # someone else's reply
            ("nobody", mine.message_id),  # a user never seen
            ("rater", theirs.message_id),
            ("rater", mine.message_id - 1),  # the user's own message, not a reply
        ]:
            with pytest.raises(httpx2.HTTPStatusError) as refusal:
                await client.feedback(user, message_id, "good")
            refused.append(refusal.value.response.status_code)
        return refused

    assert over_http(main) == [404] * 4
    [rows] = run_sql(
        db_api.url,
        "SELECT u.external_id, f.rating FROM feedback f JOIN messages m ON m.id = f.message_id "
        "JOIN conversations c ON c.id = m.conversation_id JOIN users u ON u.id = c.user_id",
    )
    assert rows == [("rater", "bad")]


def test_the_profile_is_read_and_forgotten_over_http(db_api):
    db_api.stream_reply(content=[text("Video, then.")])
    chat_events("forget", "Saturday?")
    run_sql(
        db_api.url,
        "INSERT INTO profile_facts (user_id, topic, fact) SELECT id, 'media', 'Likes video art.' "
        "FROM users WHERE external_id = 'forget'",
    )

    async def main(client: CuratorApi) -> tuple[list[Any], list[Any]]:
        before = await client.profile("forget")
        await client.forget("forget")
        await client.forget("stranger")  # unknown: nothing to do, and no user made
        return before, await client.profile("forget")

    before, after = over_http(main)

    assert [(f.topic, f.fact) for f in before] == [("media", "Likes video art.")]
    assert after == []
    [[(user_id,)], strangers] = run_sql(
        db_api.url,
        "SELECT id FROM users WHERE external_id = 'forget'",
        "SELECT id FROM users WHERE external_id = 'stranger'",
    )
    assert rows_of(db_api.url, user_id) == dict.fromkeys(USER_ROWS, 0) and strangers == []


def test_a_new_conversation_over_http_starts_from_its_context_alone(db_api):
    for reply in ("Two picks.", "Hola!"):
        db_api.stream_reply(content=[text(reply)])
    chat_events("renew", "Saturday?")

    async def main(client: CuratorApi) -> None:
        await client.new_conversation("renew")

    over_http(main)
    chat_events("renew", "Hola")

    assert [len(r["messages"]) for r in db_api.requests] == [1, 1]
    [rows] = run_sql(
        db_api.url,
        "SELECT c.ended_at IS NOT NULL FROM conversations c JOIN users u ON u.id = c.user_id "
        "WHERE u.external_id = 'renew' ORDER BY c.id",
    )
    assert rows == [(True,), (False,)]
