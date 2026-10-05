"""Polite HTTP for ingestion — the only place in the package that fetches a third-party URL.

Two things here are easy to get wrong and expensive to notice later.

**Pagination.** WordPress caps `per_page` at 100, but a page can come back short: GRAF returned
36 items for `per_page=100` with `x-wp-total: 57`. Paginating by "the page was not full" therefore
stops after one page and silently loses most of the corpus. The page count comes from
`x-wp-totalpages` and nothing else (CLAUDE.md § The GRAF API).

**Header case.** `x-wp-totalpages` is served as `X-WP-TotalPages`. `httpx2.Headers` looks up
case-insensitively, but `dict(response.headers)` lowercases every key, so a lookup written with
the documented casing misses and the sync quietly fetches page 1 only. `get_json` hands back the
`Headers` object for that reason — do not turn it into a dict on the way.

Politeness (CLAUDE.md § Stack & conventions): a descriptive User-Agent, a per-host delay, a
timeout, and bounded retries with backoff. `sleep` is injected so tests exercise the delay and the
backoff without spending wall-clock time.

**Venue sites (P2)** add two things. `get_text` fetches a page as data: a 403 is an answer about
the site, not an error, so it comes back as a `Page` rather than raising. And `RobotsPolicy`
fetches `robots.txt` *through* this client — `RobotFileParser.read()` would use `urllib` with its
own User-Agent, skipping the delay, the retries and the identification all at once.
"""

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx2

from art_curator.config import Settings, get_settings

# Worth another attempt: rate limiting, and the transient 5xx family. Everything else (401, 403,
# 404, and the rest of 4xx) means the request itself is wrong, so retrying just wastes politeness
# budget.
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

BACKOFF_BASE_S = 0.5
BACKOFF_JITTER_S = 0.25

HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
HTML_ACCEPT = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"

# The largest pilot page in the 2026-10-03 probe was 576 KB (MACBA's homepage). Past the cap the
# body is truncated, not refused: the text extracted from it is capped far lower anyway.
MAX_PAGE_BYTES = 5_000_000
# RFC 9309 § 2.5: a crawler must parse at least the first 500 KiB of robots.txt.
ROBOTS_MAX_BYTES = 500 * 1024


class FetchError(Exception):
    """A request that stayed broken after `max_attempts`."""


@dataclass(frozen=True)
class FetchPolicy:
    user_agent: str
    delay_s: float
    timeout_s: float
    max_attempts: int

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "FetchPolicy":
        s = settings or get_settings()
        return cls(
            user_agent=s.http_user_agent,
            delay_s=s.http_delay_s,
            timeout_s=s.http_timeout_s,
            max_attempts=s.http_max_attempts,
        )


@dataclass(frozen=True)
class Page:
    """One fetched URL, whatever its status. `url` is the final URL after redirects."""

    url: str
    status: int
    headers: httpx2.Headers
    content: bytes  # empty when the media type was not one the caller accepts
    truncated: bool = False

    @property
    def media_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    @property
    def is_html(self) -> bool:
        return self.media_type in HTML_TYPES

    @property
    def text(self) -> str:
        """Decoded with the header charset, else UTF-8. For HTML, prefer handing `content` to the
        parser, which also reads `<meta charset>`."""
        charset = _charset(self.headers) or "utf-8"
        try:
            return self.content.decode(charset, errors="replace")
        except LookupError:  # a charset name Python does not know
            return self.content.decode("utf-8", errors="replace")


class PoliteClient:
    """Rate-limited, retrying GET over an `httpx2.AsyncClient`.

    The delay is applied before every request to a host after the first, rather than being
    measured against a clock. That is marginally more conservative than "at least `delay_s`
    between requests" — time already spent in-flight does not count against it — which is the
    right side to err on for someone else's server, and it keeps the behaviour deterministic
    under an injected `sleep`.
    """

    def __init__(
        self,
        client: httpx2.AsyncClient,
        policy: FetchPolicy,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._policy = policy
        self._sleep = sleep
        self._seen_hosts: set[str] = set()
        self._host_delay: dict[str, float] = {}

    @property
    def user_agent(self) -> str:
        return self._policy.user_agent

    def set_crawl_delay(self, host: str, seconds: float) -> None:
        """Honour a site's `Crawl-delay`. It can raise this host's delay, never lower it."""
        current = self._host_delay.get(host, self._policy.delay_s)
        self._host_delay[host] = max(current, seconds)

    async def get_json(
        self, url: str, params: Mapping[str, Any] | None = None
    ) -> tuple[Any, httpx2.Headers]:
        """GET `url` and parse JSON. Returns the body and the response headers (not a dict)."""

        async def send() -> tuple[httpx2.Response, None]:
            return await self._client.get(url, params=params), None

        response, _ = await self._retrying(url, send)
        response.raise_for_status()
        return response.json(), response.headers

    async def get_text(
        self,
        url: str,
        *,
        accept: str = HTML_ACCEPT,
        media_types: frozenset[str] | None = HTML_TYPES,
        max_bytes: int = MAX_PAGE_BYTES,
    ) -> Page:
        """GET `url` as a document. Any final status comes back as a `Page` — a 403 or 404 is
        something to record about the site, not an exception. Retryable statuses are still
        retried; if they persist, the last response is returned.

        The body is streamed and cut at `max_bytes`. If `media_types` is given and the response's
        type is not in it (a PDF behind a link that looked like a page), the body is not read at
        all. Raises `FetchError` only when no usable response arrives (DNS, connection, timeout,
        a redirect loop, a body that does not decode).
        """
        headers = {"Accept": accept}

        async def send() -> tuple[httpx2.Response, tuple[bytes, bool]]:
            request = self._client.build_request("GET", url, headers=headers)
            response = await self._client.send(request, stream=True)
            try:
                media_type = response.headers.get("content-type", "").split(";")[0].strip()
                if media_types is not None and media_type.lower() not in media_types:
                    return response, (b"", False)
                return response, await _read_capped(response, max_bytes)
            finally:
                await response.aclose()

        response, (content, truncated) = await self._retrying(url, send, return_last=True)
        return Page(
            url=str(response.url),
            status=response.status_code,
            headers=response.headers,
            content=content,
            truncated=truncated,
        )

    async def paginate(
        self,
        url: str,
        *,
        per_page: int = 100,
        params: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield every item of a paginated WordPress collection, page count from the header."""
        page = 1
        total_pages = 1
        while page <= total_pages:
            body, headers = await self.get_json(
                url, {**(params or {}), "per_page": per_page, "page": page}
            )
            if not isinstance(body, list):
                raise FetchError(f"{url}: expected a JSON array, got {type(body).__name__}")
            if page == 1:
                total_pages = _total_pages(headers)
            if not body:
                # The collection shrank mid-run; the remaining pages no longer exist.
                return
            for item in body:
                yield item
            page += 1

    async def _retrying[T](
        self,
        url: str,
        send: Callable[[], Awaitable[tuple[httpx2.Response, T]]],
        *,
        return_last: bool = False,
    ) -> tuple[httpx2.Response, T]:
        """Run `send` until it yields a non-retryable status. With `return_last`, a retryable
        status that outlasts every attempt is returned instead of raised."""
        attempts = self._policy.max_attempts
        last: str = "no attempt made"
        for attempt in range(1, attempts + 1):
            await self._throttle(url)
            try:
                response, body = await send()
            except httpx2.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt < attempts:
                    await self._sleep(_retry_delay(attempt, None))
                continue
            except httpx2.RequestError as exc:
                # A redirect loop or an undecodable body: the server answered, and would answer
                # the same way again.
                raise FetchError(f"{url}: {type(exc).__name__}: {exc}") from exc
            # A redirect reached another host (`macba.cat` → `www.macba.cat`): that server has now
            # been hit too, so the next request to it waits like any other.
            self._seen_hosts.add(urlsplit(str(response.url)).netloc)
            if response.status_code not in RETRY_STATUSES:
                return response, body
            if attempt == attempts and return_last:
                return response, body
            last = f"HTTP {response.status_code}"
            if attempt < attempts:
                await self._sleep(_retry_delay(attempt, response.headers))
        raise FetchError(f"{url}: {attempts} attempts failed, last was {last}")

    async def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        if host in self._seen_hosts:
            await self._sleep(self._host_delay.get(host, self._policy.delay_s))
        else:
            self._seen_hosts.add(host)


@asynccontextmanager
async def open_client(
    policy: FetchPolicy | None = None,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[PoliteClient]:
    """A `PoliteClient` over a configured `AsyncClient`. `transport` is the seam tests fake."""
    policy = policy or FetchPolicy.from_settings()
    async with httpx2.AsyncClient(
        headers={"User-Agent": policy.user_agent, "Accept": "application/json"},
        timeout=policy.timeout_s,
        follow_redirects=True,
        transport=transport,
    ) as client:
        yield PoliteClient(client, policy, sleep=sleep)


# --- robots.txt ----------------------------------------------------------------------------------


class RobotsStatus(StrEnum):
    RULES = "rules"  # a robots.txt was served and parsed
    ALLOW_ALL = "allow_all"  # 404 and the rest of 4xx: no robots.txt, so no restrictions
    DISALLOW_ALL = "disallow_all"  # 401 / 403: the site refuses to say, so we stay out
    UNAVAILABLE = "unavailable"  # 5xx, persistent 429, or no response: skip the site this run


class HostRobots:
    """What one origin's robots.txt lets us fetch.

    `UNAVAILABLE` is not `DISALLOW_ALL`: it refuses everything *this run* and says nothing about
    the next, so the crawl reports it as a transient failure rather than a site decision.

    Matching is `urllib.robotparser`'s (CLAUDE.md § Stack & conventions), which compares path
    prefixes literally: a wildcard rule such as `Disallow: /*?add-to-cart=` matches nothing. None
    of the pilot sites' wildcard rules cover a page we would fetch (2026-10-03 probe).
    """

    def __init__(
        self,
        origin: str,
        status: RobotsStatus,
        http_status: int | None = None,
        parser: RobotFileParser | None = None,
        crawl_delay: float | None = None,
    ) -> None:
        self.origin = origin
        self.status = status
        self.http_status = http_status
        self.crawl_delay = crawl_delay
        self._parser = parser

    def allows(self, url: str, user_agent: str) -> bool:
        if self.status is RobotsStatus.ALLOW_ALL:
            return True
        if self.status is RobotsStatus.RULES and self._parser is not None:
            return self._parser.can_fetch(user_agent, url)
        return False


class RobotsPolicy:
    """Per-origin robots.txt, fetched once per run through the `PoliteClient`.

    The status mapping is PLAN.md § 8 (P2): 404 → allow all; 401/403 → disallow all; 5xx, a
    persistent 429, or no response → unavailable. A `Crawl-delay` for our agent is passed to the
    client, which may raise the host's delay but never lower it.
    """

    def __init__(self, client: PoliteClient) -> None:
        self._client = client
        self._cache: dict[str, HostRobots] = {}

    async def for_url(self, url: str) -> HostRobots:
        origin = _origin(url)
        if origin not in self._cache:
            self._cache[origin] = await self._fetch(origin)
        return self._cache[origin]

    async def allows(self, url: str) -> bool:
        return (await self.for_url(url)).allows(url, self._client.user_agent)

    async def _fetch(self, origin: str) -> HostRobots:
        try:
            page = await self._client.get_text(
                f"{origin}/robots.txt",
                accept="text/plain,*/*;q=0.1",
                media_types=None,  # served as text/html often enough; the parser copes
                max_bytes=ROBOTS_MAX_BYTES,
            )
        except FetchError:
            return HostRobots(origin, RobotsStatus.UNAVAILABLE)

        status = page.status
        if status in (401, 403):
            return HostRobots(origin, RobotsStatus.DISALLOW_ALL, status)
        if status in RETRY_STATUSES or status >= 500:
            return HostRobots(origin, RobotsStatus.UNAVAILABLE, status)
        if 400 <= status < 500:
            return HostRobots(origin, RobotsStatus.ALLOW_ALL, status)
        if not 200 <= status < 300:  # a 3xx with no `Location` to follow
            return HostRobots(origin, RobotsStatus.UNAVAILABLE, status)

        parser = RobotFileParser()
        parser.parse(page.text.splitlines())  # also stamps last_checked, which can_fetch needs
        delay = parser.crawl_delay(self._client.user_agent)
        crawl_delay = float(delay) if delay is not None else None
        if crawl_delay:
            # Both hosts: `http://macba.cat/robots.txt` redirects to `www.macba.cat`, which is
            # where the pages are actually fetched from.
            for host in {urlsplit(origin).netloc, urlsplit(page.url).netloc}:
                self._client.set_crawl_delay(host, crawl_delay)
        return HostRobots(origin, RobotsStatus.RULES, status, parser, crawl_delay)


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _charset(headers: httpx2.Headers) -> str | None:
    for param in headers.get("content-type", "").split(";")[1:]:
        key, _, value = param.strip().partition("=")
        if key.lower() == "charset" and value:
            return value.strip('"').strip()
    return None


async def _read_capped(response: httpx2.Response, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        room = max_bytes - size
        if len(chunk) > room:
            chunks.append(chunk[:room])
            return b"".join(chunks), True
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks), False


def _total_pages(headers: httpx2.Headers) -> int:
    """`X-WP-TotalPages`, the only trustworthy end-of-collection signal."""
    try:
        return max(int(headers.get("x-wp-totalpages", "1") or 1), 1)
    except ValueError:
        return 1


def _retry_delay(attempt: int, headers: httpx2.Headers | None) -> float:
    """Exponential backoff with jitter, unless the server named a wait (`Retry-After`)."""
    if headers is not None:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass  # http-date form; fall through to backoff
    return BACKOFF_BASE_S * 2 ** (attempt - 1) + random.uniform(0, BACKOFF_JITTER_S)
