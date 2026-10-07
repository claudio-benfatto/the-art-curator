"""Fetch one venue page and say whether it is usable (PLAN.md § 8, P2: Text, Unusable sites).

**Text** is every visible string on the page: markup gone, navigation, asides and footers kept.
Not the main-body `extract()`, which drops the dates (probe, 2026-10-03).

Nor `html2txt`'s own cleaning, which is on by default and removes any `div` whose class or id
merely *contains* "footer". On two pilot listings that was the listing itself: FUGA kept 129 of
549 words and Sala Parés 125 of 525, and discovery called both pages "other" (2026-10-07). So
`page_text` removes only what a browser never shows (`INVISIBLE_TAGS`) and turns that cleaning
off. Opening hours and addresses come along; a model reads past those, and cannot read what was
cut.

**A page we cannot use is an answer, not an error.** `blocked` and `thin` are recorded and
reported, and the venue stays facts-only. Nothing here retries with another User-Agent, solves a
challenge or renders JavaScript.

The text this module returns is third-party prose. It may go to a model and into
`venue_pages.raw_text` (7-day TTL), and nowhere else (CLAUDE.md § 1).
"""

from dataclasses import dataclass
from enum import StrEnum

from trafilatura import html2txt, load_html

from art_curator.ingest.http import FetchError, Page, PoliteClient, RobotsPolicy, RobotsStatus

MAX_TEXT_CHARS = 24_000
# Under this, the page was rendered by JavaScript we did not run, or there is nothing on it.
THIN_WORDS = 50

# Never rendered as text by a browser. `<noscript>` is, but only with scripting off.
INVISIBLE_TAGS = ("script", "style", "noscript", "template", "svg")

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
    tree = load_html(html)
    if tree is None:
        return ""
    for element in list(tree.iter(*INVISIBLE_TAGS)):
        if element.getparent() is not None:
            element.drop_tree()  # keeps the text that follows the element
    return " ".join(html2txt(tree, clean=False).split())[:MAX_TEXT_CHARS]


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
