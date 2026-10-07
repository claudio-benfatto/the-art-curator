"""A fake venue website for tests: canned pages over an httpx2 `MockTransport`.

`PoliteClient`, `RobotsPolicy` and `fetch_page` run unchanged; only the wire is fake. The pages
are the hand-written fixtures in `tests/fixtures/crawl/` or strings built in the test, never a
recorded page (CLAUDE.md § 1).
"""

from pathlib import Path

import httpx2

from art_curator.ingest.http import FetchPolicy

FIXTURES = Path(__file__).parent / "fixtures" / "crawl"
POLICY = FetchPolicy(user_agent="test-agent/1.0", delay_s=0.5, timeout_s=5.0, max_attempts=1)

HOSTNAME = "exemple.cat"
HOST = f"https://www.{HOSTNAME}"
HOMEPAGE = (FIXTURES / "homepage.html").read_bytes()
LISTING = (FIXTURES / "listing.html").read_bytes()


def html(body: bytes | str, status: int = 200, **headers: str) -> httpx2.Response:
    return httpx2.Response(
        status, content=body, headers={"content-type": "text/html; charset=utf-8", **headers}
    )


async def no_sleep(_: float) -> None:
    pass


class FakeSite:
    """`pages` maps a path (with its query, if any) to a response. Anything else, and every other
    host, is a 404, which for `/robots.txt` means "no rules"."""

    def __init__(self, pages: dict[str, httpx2.Response] | None = None) -> None:
        self.pages = {"/": html(HOMEPAGE), "/ca/exposicions/": html(LISTING), **(pages or {})}
        self.requests: list[httpx2.Request] = []
        self.transport = httpx2.MockTransport(self._handle)

    def paths(self) -> list[str]:
        return [r.url.raw_path.decode() for r in self.requests]

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        page = None
        if request.url.host.removeprefix("www.") == HOSTNAME:
            page = self.pages.get(request.url.raw_path.decode())
        page = page or httpx2.Response(404)
        # A fresh response each time: one object cannot be streamed twice.
        return httpx2.Response(page.status_code, content=page.content, headers=page.headers)
