"""The curator over HTTP: what every client talks to (PLAN.md § 8, PR 34).

    uv run uvicorn art_curator.api.main:app    # 127.0.0.1:8000

The backend never learns what a client is (CLAUDE.md § 3): a user is `(client, user)`, two opaque
strings the client picks, and the API holds no client types. One bearer token per deployment,
`API_TOKEN`, guards every route; the API is bound to localhost until P7.

`POST /chat` answers one message as Server-Sent Events, each a JSON object:

    status  {"tool": "search_events"}      a tool is about to run
    text    {"text": "Two shows"}          the next piece of the reply, as the model writes it
    done    {"message_id": 41, "iteration_count": 1, "stop_reason": "end_turn"}
    error   {"error": "APIStatusError", "message": "..."}

`done` or `error` is always the last event. The turn is saved just before `done`, and
`message_id` is the reply `POST /feedback` rates. A turn that fails, or whose client disconnects
before `done`, saves nothing and leaves the conversation as it was; its model calls are recorded
all the same (`Curator.events`). Thinking is never sent. The stream sends a keep-alive comment
every 15 s of silence, so a long think does not look like a dead connection.
"""

import logging
import secrets
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from art_curator.agent.loop import Curator, TextDelta, ToolRunning, TurnDone
from art_curator.agent.prompt import local_today
from art_curator.agent.store import ConversationStore
from art_curator.agent.tools import CuratorTools
from art_curator.config import Settings, get_settings
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.llm.client import LlmClient, get_llm_client
from art_curator.obs import telemetry

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if get_settings().api_token is None:
        raise RuntimeError("API_TOKEN is not set: the API has no unauthenticated mode")
    try:
        yield
    finally:
        telemetry.setup_telemetry().force_flush()
        await get_engine().dispose()


_bearer = HTTPBearer(auto_error=False)


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Every route's guard: the request carries `Authorization: Bearer <API_TOKEN>`."""
    if settings.api_token is None:  # lifespan refuses to start; this covers an app run without it
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API_TOKEN is not set")
    expected = settings.api_token.get_secret_value().encode()
    sent = credentials.credentials.encode() if credentials else b""
    if not secrets.compare_digest(sent, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Bad or missing token", {"WWW-Authenticate": "Bearer"}
        )


# No /docs or /openapi.json: FastAPI serves them outside the app's dependencies, so they would be
# the only routes the token does not guard.
app = FastAPI(
    title="The Art Curator",
    lifespan=lifespan,
    dependencies=[Depends(require_token)],
    openapi_url=None,
)

Sessions = Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)]


def get_store(sessions: Sessions) -> ConversationStore:
    return ConversationStore(sessions)


Store = Annotated[ConversationStore, Depends(get_store)]
Name = Annotated[str, Field(min_length=1, max_length=200)]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client: Name  # opaque: "cli", say
    user: Name  # the client's id for the person
    text: str = Field(min_length=1, max_length=4000)
    # Talk as if today were this day, as `cli chat --today`: for tests and replays. A conversation
    # opened on another day ends, and the new one's calendar starts here.
    today: date | None = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client: Name
    user: Name
    message_id: int  # from the reply's `done` event
    rating: Literal["good", "bad"]


class ProfileFact(BaseModel):
    id: int
    topic: str
    fact: str


@app.post("/chat", response_class=EventSourceResponse)
async def chat(
    request: ChatRequest,
    store: Store,
    sessions: Sessions,
    llm: Annotated[LlmClient, Depends(get_llm_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterable[ServerSentEvent]:
    """Answer `text` in the user's open conversation (a new one if it expired), as events."""
    try:
        user_id = await store.user(request.client, request.user)
        stored = await store.current(user_id, request.today or local_today())
        curator = Curator(
            llm,
            CuratorTools(sessions, user_id),
            model=settings.chat_model,
            effort=settings.chat_effort,
        )
        async for event in curator.events(stored.conversation, request.text):
            match event:
                case ToolRunning(name):
                    yield ServerSentEvent(event="status", data={"tool": name})
                case TextDelta(text):
                    yield ServerSentEvent(event="text", data={"text": text})
                case TurnDone(turn):
                    message_id = await store.save(stored, turn)
                    done = {
                        "message_id": message_id,
                        "iteration_count": turn.iteration_count,
                        "stop_reason": turn.stop_reason,
                    }
                    yield ServerSentEvent(event="done", data=done)
    except Exception as exc:  # the stream has begun: the status code is already 200
        log.exception("chat turn failed")
        error = {"error": type(exc).__name__, "message": str(exc)}
        yield ServerSentEvent(event="error", data=error)


@app.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(request: FeedbackRequest, store: Store) -> Response:
    """Rate one reply; a second verdict on it replaces the first."""
    user_id = await store.find_user(request.client, request.user)
    if user_id is None or not await store.rate_reply(user_id, request.message_id, request.rating):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such reply in this user's conversations")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/users/{client}/{user}/profile")
async def profile(client: str, user: str, store: Store) -> list[ProfileFact]:
    """What is remembered about the user, as their next conversation will open with it."""
    user_id = await store.find_user(client, user)
    facts = await store.remembered(user_id) if user_id is not None else []
    return [ProfileFact(id=f.id, topic=f.topic, fact=f.fact) for f in facts]


@app.delete("/users/{client}/{user}/profile", status_code=status.HTTP_204_NO_CONTENT)
async def forget(client: str, user: str, store: Store) -> Response:
    """Delete everything kept about the user: profile, conversations, messages and feedback."""
    user_id = await store.find_user(client, user)
    if user_id is not None:
        await store.forget(user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.delete("/users/{client}/{user}/conversation", status_code=status.HTTP_204_NO_CONTENT)
async def close_conversation(client: str, user: str, store: Store) -> Response:
    """End the user's open conversation: their next message starts a new one."""
    user_id = await store.find_user(client, user)
    if user_id is not None:
        await store.close(user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
