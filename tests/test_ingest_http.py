"""`ingest/http.py`: WordPress pagination, retries and politeness.

No test here spends wall-clock time: `sleep` is injected and records what it was asked to wait.
"""

import asyncio
import json

import httpx2
import pytest

from art_curator.ingest.http import (
    FetchError,
    FetchPolicy,
    PoliteClient,
    open_client,
)

POLICY = FetchPolicy(user_agent="test-agent/1.0", delay_s=0.5, timeout_s=5.0, max_attempts=3)
URL = "https://graf.cat/wp-json/wp/v2/events"


class Recorder:
    """An injected `sleep` that records instead of waiting."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _json(body, *, status: int = 200, **headers) -> httpx2.Response:
    return httpx2.Response(status, content=json.dumps(body), headers=headers)


def _run(responses, fn, *, policy: FetchPolicy = POLICY):
    """Drive `fn(client)` against a queue of canned responses. Returns (result, requests, waits)."""
    queue = list(responses)
    requests: list[httpx2.Request] = []
    sleep = Recorder()

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return queue.pop(0)

    async def main():
        async with open_client(
            policy, transport=httpx2.MockTransport(handle), sleep=sleep
        ) as client:
            return await fn(client)

    return asyncio.run(main()), requests, sleep.waits


# --- pagination ---------------------------------------------------------------------------------


async def _collect(client: PoliteClient) -> list[dict]:
    return [item async for item in client.paginate(URL)]


def test_short_page_still_fetches_every_page():
    """The live failure mode: 100 requested, 36 returned, and two pages exist anyway."""
    page1 = [{"id": i} for i in range(36)]
    page2 = [{"id": i} for i in range(36, 57)]
    items, requests, _ = _run(
        [
            _json(page1, **{"X-WP-Total": "57", "X-WP-TotalPages": "2"}),
            _json(page2, **{"X-WP-Total": "57", "X-WP-TotalPages": "2"}),
        ],
        _collect,
    )

    assert len(items) == 57
    assert [r.url.params.get("page") for r in requests] == ["1", "2"]


def test_page_count_is_read_case_insensitively():
    """Served as `X-WP-TotalPages`; read as anything else the sync silently gets one page."""
    items, requests, _ = _run(
        [
            _json([{"id": 1}], **{"X-WP-TotalPages": "3"}),
            _json([{"id": 2}], **{"X-WP-TotalPages": "3"}),
            _json([{"id": 3}], **{"X-WP-TotalPages": "3"}),
        ],
        _collect,
    )
    assert len(items) == 3
    assert len(requests) == 3


def test_missing_page_header_means_one_page():
    items, requests, _ = _run([_json([{"id": 1}])], _collect)
    assert len(items) == 1
    assert len(requests) == 1


def test_pagination_stops_when_the_collection_shrinks():
    """Header says 3 pages, but page 2 is empty because events dropped out of the live window."""
    items, requests, _ = _run(
        [
            _json([{"id": 1}], **{"X-WP-TotalPages": "3"}),
            _json([], **{"X-WP-TotalPages": "3"}),
        ],
        _collect,
    )
    assert len(items) == 1
    assert len(requests) == 2


def test_non_array_body_is_an_error():
    with pytest.raises(FetchError, match="expected a JSON array"):
        _run([_json({"code": "rest_no_route"})], _collect)


# --- retries ------------------------------------------------------------------------------------


async def _get(client: PoliteClient):
    body, _ = await client.get_json(URL)
    return body


def test_rate_limit_then_success_honours_retry_after():
    body, requests, waits = _run(
        [
            _json({"code": "too_many"}, status=429, **{"Retry-After": "2"}),
            _json({"ok": True}),
        ],
        _get,
    )

    assert body == {"ok": True}
    assert len(requests) == 2
    # One backoff of exactly Retry-After, plus one per-host delay before the second request.
    assert 2.0 in waits
    assert waits.count(POLICY.delay_s) == 1


def test_server_errors_exhaust_attempts_and_raise():
    with pytest.raises(FetchError, match="3 attempts failed, last was HTTP 500"):
        _run([_json({}, status=500) for _ in range(3)], _get)


def test_client_errors_are_not_retried():
    """A 404 is our bug, not the server's mood — retrying just spends politeness budget."""
    with pytest.raises(httpx2.HTTPStatusError):
        _run([_json({"code": "rest_no_route"}, status=404)], _get)


def test_transport_errors_are_retried():
    calls = {"n": 0}

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx2.ConnectError("boom", request=request)
        return _json({"ok": True})

    async def main():
        async with open_client(
            POLICY, transport=httpx2.MockTransport(handle), sleep=Recorder()
        ) as client:
            return await client.get_json(URL)

    body, _ = asyncio.run(main())
    assert body == {"ok": True}
    assert calls["n"] == 3


# --- politeness ---------------------------------------------------------------------------------


def test_first_request_to_a_host_is_not_delayed():
    _, _, waits = _run([_json([{"id": 1}])], _collect)
    assert waits == []


def test_every_later_request_to_the_same_host_is_delayed():
    _, _, waits = _run(
        [_json([{"id": i}], **{"X-WP-TotalPages": "3"}) for i in range(3)],
        _collect,
    )
    assert waits == [POLICY.delay_s, POLICY.delay_s]


def test_descriptive_user_agent_is_sent():
    _, requests, _ = _run([_json([{"id": 1}])], _collect)
    assert requests[0].headers["user-agent"] == "test-agent/1.0"


def test_policy_comes_from_settings(monkeypatch):
    monkeypatch.setenv("HTTP_DELAY_S", "1.5")
    monkeypatch.setenv("HTTP_MAX_ATTEMPTS", "7")
    from art_curator.config import Settings

    policy = FetchPolicy.from_settings(Settings(_env_file=None))
    assert policy.delay_s == 1.5
    assert policy.max_attempts == 7
    assert "art-curator" in policy.user_agent
