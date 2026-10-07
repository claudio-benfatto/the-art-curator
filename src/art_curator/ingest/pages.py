"""Fetch one venue page and say whether it is usable (PLAN.md § 8, P2: Text, Unusable sites).

**Text** is `trafilatura.html2txt`: the page's visible strings, markup gone, navigation and
headers kept. Not the main-body `extract()`, which drops the dates (probe, 2026-10-03).

It is not quite all of the page. `html2txt` cleans by default, which removes `<footer>`,
`<aside>`, anything whose class or id says "footer", and cookie banners, along with scripts and
styles. Turning cleaning off brings those back together with the script and style source, so the
default stays. A site that keeps its dates in an `<aside>` would lose them here.

**A page we cannot use is an answer, not an error.** `blocked` and `thin` are recorded and
reported, and the venue stays facts-only. Nothing here retries with another User-Agent, solves a
challenge or renders JavaScript.

The text this module returns is third-party prose. It may go to a model and into
`venue_pages.raw_text` (7-day TTL), and nowhere else (CLAUDE.md § 1).
"""

from dataclasses import dataclass
from enum import StrEnum

from trafilatura import html2txt

from art_curator.ingest.http import FetchError, Page, PoliteClient, RobotsPolicy, RobotsStatus

MAX_TEXT_CHARS = 24_000
# Under this, the page was rendered by JavaScript we did not run, or there is nothing on it.
THIN_WORDS = 50

BLOCKED_STATUSES = frozenset({401, 403})
# Cloudflare marks a challenge response with this header, whatever status it carries.
CHALLENGE_HEADER = "cf-mitigated"


class PageStatus(StrEnum):
    OK = "ok"
    THIN = "thin"  # fetched, but fewer than THIN_WORDS words of text
    BLOCKED = "blocked"  # 401 / 403 / bot challenge
    ROBOTS = "robots"  # robots.txt disallows it
    UNAVAILABLE = "unavailable"  # no answer from the page or from robots.txt: try again next run
    ERROR = "error"  # any other status, or not HTML


@dataclass(frozen=True)
class Fetched:
    url: str  # as requested
    status: PageStatus
    http_status: int | None = None
    page: Page | None = None  # present for OK and THIN
    text: str = ""  # present for OK and THIN

    @property
    def detail(self) -> str:
        """The status as a report prints it: `blocked (HTTP 403)`."""
        return f"{self.status} (HTTP {self.http_status})" if self.http_status else str(self.status)


def page_text(html: bytes | str) -> str:
    """All visible text of `html`, whitespace collapsed, capped at `MAX_TEXT_CHARS`."""
    return " ".join(html2txt(html).split())[:MAX_TEXT_CHARS]


async def fetch_page(client: PoliteClient, robots: RobotsPolicy, url: str) -> Fetched:
    """Fetch `url` if robots.txt allows it, and classify what came back. Never raises for
    something the site did."""
    host = await robots.for_url(url)
    if host.status is RobotsStatus.UNAVAILABLE:
        return Fetched(url, PageStatus.UNAVAILABLE)
    if not host.allows(url, client.user_agent):
        return Fetched(url, PageStatus.ROBOTS)
    try:
        page = await client.get_text(url)
    except FetchError:
        return Fetched(url, PageStatus.UNAVAILABLE)

    if page.status in BLOCKED_STATUSES or CHALLENGE_HEADER in page.headers:
        return Fetched(url, PageStatus.BLOCKED, page.status)
    if not 200 <= page.status < 300 or not page.is_html:
        return Fetched(url, PageStatus.ERROR, page.status)
    text = page_text(page.content)
    status = PageStatus.OK if len(text.split()) >= THIN_WORDS else PageStatus.THIN
    return Fetched(url, status, page.status, page, text)
