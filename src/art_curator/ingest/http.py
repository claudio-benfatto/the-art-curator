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
"""

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx2

from art_curator.config import Settings, get_settings

# Worth another attempt: rate limiting, and the transient 5xx family. Everything else (401, 403,
# 404, and the rest of 4xx) means the request itself is wrong, so retrying just wastes politeness
# budget.
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

BACKOFF_BASE_S = 0.5
BACKOFF_JITTER_S = 0.25


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

    async def get_json(
        self, url: str, params: Mapping[str, Any] | None = None
    ) -> tuple[Any, httpx2.Headers]:
        """GET `url` and parse JSON. Returns the body and the response headers (not a dict)."""
        response = await self._get(url, params)
        return response.json(), response.headers

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

    async def _get(self, url: str, params: Mapping[str, Any] | None) -> httpx2.Response:
        last: str = "no attempt made"
        for attempt in range(1, self._policy.max_attempts + 1):
            await self._throttle(url)
            try:
                response = await self._client.get(url, params=params)
            except httpx2.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code not in RETRY_STATUSES:
                    response.raise_for_status()
                    return response
                last = f"HTTP {response.status_code}"
                if attempt < self._policy.max_attempts:
                    await self._sleep(_retry_delay(attempt, response.headers))
                continue
            if attempt < self._policy.max_attempts:
                await self._sleep(_retry_delay(attempt, None))
        raise FetchError(f"{url}: {self._policy.max_attempts} attempts failed, last was {last}")

    async def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        if host in self._seen_hosts:
            await self._sleep(self._policy.delay_s)
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
