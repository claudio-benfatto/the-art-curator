"""A fake GRAF for tests: the recorded fixtures, served over an httpx2 `MockTransport`.

`PoliteClient` and `graf.fetch` run unchanged; only the wire is fake. Pages are served **short**,
as GRAF does (36 items for `per_page=100`), so a fetch that paginates by "the page was not full"
fails here too. `/events/{id}/occurrences` is derived from the event rows unless overridden.
"""

import json
import re
from pathlib import Path
from typing import Any

import httpx2

from art_curator.ingest.http import FetchPolicy

FIXTURES = Path(__file__).parent / "fixtures" / "graf"
BASE_URL = "https://graf.test/wp-json/wp/v2"
POLICY = FetchPolicy(user_agent="test-agent/1.0", delay_s=0.5, timeout_s=5.0, max_attempts=1)

_OCCURRENCES = re.compile(r"/events/(\d+)/occurrences$")


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def full_corpus() -> dict[str, list[dict[str, Any]]]:
    """Every term and profile GRAF served on 2026-09-28 (identifiers only), and the event rows.

    Enough for the 20 pilots to match and for `--min-venues 500` to pass.
    """
    identity = fixture("identity.json")
    return {
        "venues": identity["terms"],
        "profiles": identity["profiles"],
        "events": fixture("events.json"),
    }


async def no_sleep(_: float) -> None:
    pass


class FakeGraf:
    def __init__(
        self,
        venues: list[dict[str, Any]] | None = None,
        profiles: list[dict[str, Any]] | None = None,
        events: list[dict[str, Any]] | None = None,
        *,
        occurrences: dict[int, Any] | None = None,
        page_size: int = 3,
    ) -> None:
        self.collections = {
            "event-venues": fixture("venues.json") if venues is None else venues,
            "users": fixture("profiles.json") if profiles is None else profiles,
            "events": fixture("events.json") if events is None else events,
        }
        self.occurrences = occurrences or {}  # post id -> body, overriding the derived one
        self.page_size = page_size
        self.requests: list[httpx2.Request] = []
        self.transport = httpx2.MockTransport(self._handle)

    def paths(self) -> list[str]:
        return [r.url.path.removeprefix("/wp-json/wp/v2/") for r in self.requests]

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/wp-json/wp/v2/")
        if match := _OCCURRENCES.search(request.url.path):
            return httpx2.Response(200, json=self._occurrences(int(match.group(1))))
        if path not in self.collections:
            return httpx2.Response(404, json={"code": "rest_no_route"})
        rows = self.collections[path]
        page = int(request.url.params.get("page", 1))
        pages = max(-(-len(rows) // self.page_size), 1)
        start = (page - 1) * self.page_size
        return httpx2.Response(
            200,
            json=rows[start : start + self.page_size],
            headers={"X-WP-Total": str(len(rows)), "X-WP-TotalPages": str(pages)},
        )

    def _occurrences(self, post_id: int) -> Any:
        if post_id in self.occurrences:
            return self.occurrences[post_id]
        return [
            {
                "occurrence_id": int(row["occurrence_id"]),
                "event_id": post_id,
                "start": row["start"],
                "end": row["end"],
            }
            for row in self.collections["events"]
            if row["id"] == post_id
        ]
