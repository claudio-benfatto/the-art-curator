"""A client of the curator's HTTP API (`api/main.py`), for one client name and one deployment.

`chat` reads `POST /chat`'s Server-Sent Events as they arrive and yields them as `ToolStatus`,
`TextChunk` and a last `ReplyDone`; an `error` event raises `ChatFailed`. The rest are one request
each. Nothing here imports the backend: this is what any client over HTTP sees.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date
from types import TracebackType
from typing import Any, Literal, Self
from urllib.parse import quote

import httpx2

# A turn can think for a while before its first word; the server sends a keep-alive comment every
# 15 s of silence, so a read that waits longer than this means the connection is gone.
READ_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class ToolStatus:
    """A tool is about to run."""

    tool: str


@dataclass(frozen=True)
class TextChunk:
    """The next piece of the reply."""

    text: str


@dataclass(frozen=True)
class ReplyDone:
    """The turn is over and saved; `message_id` is what `feedback` rates."""

    message_id: int
    iteration_count: int
    stop_reason: str | None


ChatEvent = ToolStatus | TextChunk | ReplyDone


class ChatFailed(Exception):
    """The server could not finish the turn; nothing of it was saved."""


@dataclass(frozen=True)
class ProfileFact:
    id: int
    topic: str
    fact: str


class CuratorApi:
    """`async with CuratorApi(url, token, client="cli") as api:` and then one call per action."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        client: str,
        transport: httpx2.AsyncBaseTransport | None = None,  # tests: the app in-process
    ) -> None:
        self._client = client
        self._http = httpx2.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx2.Timeout(30.0, read=READ_TIMEOUT_S),
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(
        self, user: str, text: str, *, today: date | None = None
    ) -> AsyncIterator[ChatEvent]:
        """Send `text` as `user`'s next message and yield the reply as it comes."""
        body: dict[str, Any] = {"client": self._client, "user": user, "text": text}
        if today is not None:
            body["today"] = today.isoformat()
        async with self._http.stream("POST", "/chat", json=body) as response:
            if response.is_error:
                await response.aread()
                response.raise_for_status()
            async for name, data in _server_sent_events(response):
                match name:
                    case "status":
                        yield ToolStatus(data["tool"])
                    case "text":
                        yield TextChunk(data["text"])
                    case "done":
                        yield ReplyDone(
                            data["message_id"], data["iteration_count"], data["stop_reason"]
                        )
                    case "error":
                        raise ChatFailed(f"{data['error']}: {data['message']}")

    async def feedback(self, user: str, message_id: int, rating: Literal["good", "bad"]) -> None:
        response = await self._http.post(
            "/feedback",
            json={"client": self._client, "user": user, "message_id": message_id, "rating": rating},
        )
        response.raise_for_status()

    async def profile(self, user: str) -> list[ProfileFact]:
        response = await self._http.get(f"{self._user_path(user)}/profile")
        response.raise_for_status()
        return [ProfileFact(**fact) for fact in response.json()]

    async def forget(self, user: str) -> None:
        """Delete the user's profile and every conversation."""
        response = await self._http.delete(f"{self._user_path(user)}/profile")
        response.raise_for_status()

    async def new_conversation(self, user: str) -> None:
        """End the user's open conversation: the next message starts a new one."""
        response = await self._http.delete(f"{self._user_path(user)}/conversation")
        response.raise_for_status()

    def _user_path(self, user: str) -> str:
        return f"/users/{quote(self._client, safe='')}/{quote(user, safe='')}"


async def _server_sent_events(response: httpx2.Response) -> AsyncIterator[tuple[str, Any]]:
    """(event name, JSON data) for each event of a `text/event-stream` body. Comment lines (the
    server's keep-alives) are skipped; a `data` field split over several lines is joined."""
    name, data = "message", []
    async for line in response.aiter_lines():
        if not line:  # a blank line ends the event
            if data:
                yield name, json.loads("\n".join(data))
            name, data = "message", []
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
