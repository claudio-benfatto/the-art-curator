"""`ingest/http.py`: WordPress pagination, retries, politeness, venue pages and robots.txt.

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
    RobotsPolicy,
    RobotsStatus,
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


# --- documents: get_text ------------------------------------------------------------------------

PAGE = "https://venue.example/exposicions/"


def _html(body: str = "<html><body>ok</body></html>", *, status: int = 200, **headers):
    return httpx2.Response(
        status,
        content=body.encode(headers.pop("encoding", "utf-8")),
        headers={"Content-Type": "text/html; charset=utf-8", **headers},
    )


def _drive(handle, fn, *, policy: FetchPolicy = POLICY):
    """Like `_run`, with a routing handler instead of a queue. Returns (result, waits)."""
    sleep = Recorder()

    async def main():
        async with open_client(
            policy, transport=httpx2.MockTransport(handle), sleep=sleep
        ) as client:
            return await fn(client)

    return asyncio.run(main()), sleep.waits


async def _page(client: PoliteClient):
    return await client.get_text(PAGE)


def test_a_403_is_returned_as_a_page_not_raised():
    """A Cloudflare challenge is a fact about the site, and its body is what identifies it."""
    page, requests, _ = _run([_html("<title>Just a moment...</title>", status=403)], _page)
    assert page.status == 403
    assert b"Just a moment" in page.content
    assert len(requests) == 1  # 4xx is not retried


def test_documents_ask_for_html_not_json():
    """The client's default Accept is JSON (for GRAF); a venue page must not inherit it."""
    _, requests, _ = _run([_html()], _page)
    assert requests[0].headers["accept"].startswith("text/html")


def test_final_url_is_reported_after_redirects():
    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "venue.example":
            return httpx2.Response(301, headers={"Location": "https://www.venue.example/ca/"})
        return _html()

    page, _ = _drive(handle, _page)
    assert page.url == "https://www.venue.example/ca/"
    assert page.status == 200


def test_unwanted_media_type_is_not_downloaded():
    pdf = httpx2.Response(200, content=b"%PDF-1.7", headers={"Content-Type": "application/pdf"})
    page, _, _ = _run([pdf], _page)
    assert page.status == 200
    assert page.content == b""
    assert not page.is_html


def test_body_is_cut_at_the_size_cap():
    async def fetch(client: PoliteClient):
        return await client.get_text(PAGE, max_bytes=10)

    page, _, _ = _run([_html("x" * 25)], fetch)
    assert page.content == b"x" * 10
    assert page.truncated


def test_header_charset_decodes_text():
    page, _, _ = _run(
        [
            httpx2.Response(
                200,
                content="Exposició".encode("latin-1"),
                headers={"Content-Type": "text/html; charset=ISO-8859-1"},
            )
        ],
        _page,
    )
    assert page.text == "Exposició"


def test_persistent_server_error_returns_the_last_page():
    page, requests, waits = _run([_html(status=503) for _ in range(3)], _page)
    assert page.status == 503
    assert len(requests) == 3
    assert len(waits) == 4  # two backoffs + two per-host delays, none of them real


def test_no_response_at_all_raises():
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(FetchError, match="ConnectError"):
        _drive(handle, _page)


def _redirect_loop(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(302, headers={"Location": str(request.url)})


def _bad_gzip(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200,
        content=b"not gzip",
        headers={"Content-Type": "text/html", "Content-Encoding": "gzip"},
    )


@pytest.mark.parametrize(
    ("handle", "error"),
    [(_redirect_loop, "TooManyRedirects"), (_bad_gzip, "DecodingError")],
)
def test_an_unusable_response_is_a_fetch_error_and_not_retried(handle, error):
    """One broken site must surface as `FetchError`, which the crawl skips, not as a raw
    `httpx2` exception, which would end the run."""
    sleep = Recorder()

    async def main():
        async with open_client(
            POLICY, transport=httpx2.MockTransport(handle), sleep=sleep
        ) as client:
            return await client.get_text(PAGE)

    with pytest.raises(FetchError, match=error):
        asyncio.run(main())
    assert sleep.waits == []  # no backoff: it was not tried again


def test_a_redirect_loop_on_robots_skips_the_site():
    [(status, ok)], _ = _check(_redirect_loop, PAGE)
    assert (status, ok) == (RobotsStatus.UNAVAILABLE, False)


# --- robots.txt ---------------------------------------------------------------------------------

ROBOTS = "https://venue.example/robots.txt"


def _robots(response: httpx2.Response | None = None, *, fail: bool = False):
    """A handler serving `response` at /robots.txt and a plain page everywhere else."""
    seen: list[str] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        if request.url.path == "/robots.txt":
            if fail:
                raise httpx2.ConnectTimeout("slow", request=request)
            return response
        return _html()

    return handle, seen


def _text(body: str, *, status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, content=body, headers={"Content-Type": "text/plain"})


def _check(handle, *urls: str):
    """`[(status, allowed)]` for each URL, through one `RobotsPolicy`."""

    async def fn(client: PoliteClient):
        policy = RobotsPolicy(client)
        out = []
        for url in urls:
            rules = await policy.for_url(url)
            out.append((rules.status, await policy.allows(url)))
        return out

    return _drive(handle, fn)


@pytest.mark.parametrize(
    ("status", "expected", "allowed"),
    [
        (404, RobotsStatus.ALLOW_ALL, True),
        (410, RobotsStatus.ALLOW_ALL, True),
        (401, RobotsStatus.DISALLOW_ALL, False),
        (403, RobotsStatus.DISALLOW_ALL, False),
        (500, RobotsStatus.UNAVAILABLE, False),
        (503, RobotsStatus.UNAVAILABLE, False),
        (429, RobotsStatus.UNAVAILABLE, False),
    ],
)
def test_robots_status_mapping(status, expected, allowed):
    handle, _ = _robots(_text("", status=status))
    [(got, ok)], _ = _check(handle, PAGE)
    assert (got, ok) == (expected, allowed)


def test_unreachable_robots_skips_the_site():
    handle, seen = _robots(fail=True)
    [(status, ok)], waits = _check(handle, PAGE)
    assert (status, ok) == (RobotsStatus.UNAVAILABLE, False)
    assert len(seen) == POLICY.max_attempts  # retried, with no wall-clock sleep
    assert waits


def test_rules_for_other_bots_do_not_apply_to_us():
    """The ADN Galeria / ethall shape: `Disallow: /` lines, all aimed at named crawlers."""
    handle, _ = _robots(
        _text("User-agent: MJ12bot\nDisallow: /\n\nUser-agent: *\nDisallow: /admin/\n")
    )
    results, _ = _check(handle, PAGE, "https://venue.example/admin/login")
    assert results == [(RobotsStatus.RULES, True), (RobotsStatus.RULES, False)]


def test_a_group_naming_us_is_obeyed():
    handle, _ = _robots(_text("User-agent: art-curator\nDisallow: /\n\nUser-agent: *\nAllow: /\n"))

    async def fn(client: PoliteClient):
        return await RobotsPolicy(client).allows(PAGE)

    agent = FetchPolicy(
        user_agent="art-curator/0.1 (+https://example.org)",
        delay_s=0.5,
        timeout_s=5.0,
        max_attempts=3,
    )
    allowed, _ = _drive(handle, fn, policy=agent)
    assert allowed is False


def test_html_served_as_robots_means_no_rules():
    """An SPA answering every path with its shell (Dilalica, Galería Alegría in the probe)."""
    handle, _ = _robots(_html("<html><body><div id='app'></div></body></html>"))
    [(status, ok)], _ = _check(handle, PAGE)
    assert (status, ok) == (RobotsStatus.RULES, True)


def test_robots_is_fetched_once_per_origin():
    handle, seen = _robots(_text("User-agent: *\nDisallow: /private/\n"))
    _check(handle, PAGE, "https://venue.example/agenda/", "https://venue.example/private/x")
    assert seen.count(ROBOTS) == 1


def test_crawl_delay_raises_the_host_delay():
    handle, _ = _robots(_text("User-agent: *\nCrawl-delay: 5\n"))

    async def fn(client: PoliteClient):
        await RobotsPolicy(client).for_url(PAGE)
        await client.get_text(PAGE)
        await client.get_text(PAGE)

    _, waits = _drive(handle, fn)
    assert waits == [5.0, 5.0]  # robots.txt was the first request; both pages wait 5 s


def test_crawl_delay_never_lowers_the_host_delay():
    handle, _ = _robots(_text("User-agent: *\nCrawl-delay: 0.1\n"))

    async def fn(client: PoliteClient):
        await RobotsPolicy(client).for_url(PAGE)
        await client.get_text(PAGE)

    _, waits = _drive(handle, fn)
    assert waits == [POLICY.delay_s]


def test_crawl_delay_follows_a_robots_redirect():
    """`http://macba.cat/robots.txt` lands on `www.macba.cat`, where the pages are."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "venue.example":
            return httpx2.Response(
                301, headers={"Location": f"https://www.venue.example{request.url.path}"}
            )
        if request.url.path == "/robots.txt":
            return _text("User-agent: *\nCrawl-delay: 3\n")
        return _html()

    async def fn(client: PoliteClient):
        await RobotsPolicy(client).for_url(PAGE)
        await client.get_text("https://www.venue.example/a")

    _, waits = _drive(handle, fn)
    assert waits == [3.0]
