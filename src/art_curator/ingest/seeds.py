"""Run discovery over the pilot sites and keep what it found (PLAN.md § 8, P2).

`ingest/discover.py` is the two model passes and the routing rules, with no I/O. This module is
everything around them, in the order `cli discover` runs it:

1. `load_overrides()` — the human decisions in `seeds.yaml`: URLs pinned or rejected per venue.
2. `load_sites()` — one `Site` per distinct `website_url` among the crawlable pilots, with its GRAF
   hints and the seeds already in use. MACBA is four terms and one site.
3. `discover_site()` — homepage → pass 1 → candidates → pass 2 → routing. Network and model, no
   database. Page text exists only inside this call: it goes to the model and is dropped.
4. `write_seeds()` — upserts the verdicts into `venue_seeds`, in the caller's transaction, and
   **never commits** (`--dry-run` is the real write rolled back, as in `sync-graf`).
5. `report()` — pure. What a person reads to decide what to put in `seeds.yaml`.

Rules the code encodes:

- **A human decision outlives the machine.** A pinned venue is not discovered at all; a rejected
  URL is removed from the homepage's links before the model sees them.
- **The machine may confirm a seed, never swap one** (`route_site`). A proposal that differs from
  the seed in use is held as ambiguous.
- **`venue_seeds` is the latest verdict per (venue, URL).** A URL the model was not shown this run
  keeps its row untouched.
- **A site we cannot read is reported, not worked around**: `blocked`, `robots`, `thin`,
  `unavailable`. No headless browser, no challenge bypass.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from typing import Any, Self
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from sqlalchemy import exists, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.db.models import EventOccurrence, EventSnapshot, Venue, VenueSeed
from art_curator.ingest.discover import (
    Candidate,
    DiscoveryError,
    GrafHints,
    SeedStatus,
    classify_page,
    collect_links,
    pick_candidates,
    route_site,
    url_key,
)
from art_curator.ingest.http import PoliteClient, RobotsPolicy
from art_curator.ingest.matching import normalize_name
from art_curator.ingest.pages import PageStatus, fetch_page
from art_curator.llm.client import LlmClient

VENUES = Venue.__table__
EVENTS = EventSnapshot.__table__
OCCURRENCES = EventOccurrence.__table__
SEEDS = VenueSeed.__table__


# --- seeds.yaml: the human decisions -------------------------------------------------------------


class VenueOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pin: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()

    @field_validator("pin", "reject", mode="before")
    @classmethod
    def _one_or_many(cls, value: Any) -> Any:
        return (value,) if isinstance(value, str) else value

    @field_validator("pin", "reject")
    @classmethod
    def _web_urls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for url in value:
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                raise ValueError(f"{url!r} is not an http(s) URL")
        return value

    @model_validator(mode="after")
    def _not_both(self) -> Self:
        both = {url_key(u) for u in self.pin} & {url_key(u) for u in self.reject}
        if both:
            raise ValueError(f"pinned and rejected at once: {sorted(both)}")
        return self


class SeedOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    venues: dict[str, VenueOverride] = {}

    @field_validator("venues", mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _one_entry_per_venue(self) -> Self:
        repeated = [k for k, n in Counter(map(normalize_name, self.venues)).items() if n > 1]
        if repeated:
            raise ValueError(f"more than one entry for {repeated}")
        return self

    def for_venue(self, name: str) -> VenueOverride:
        key = normalize_name(name)
        return next(
            (o for venue, o in self.venues.items() if normalize_name(venue) == key),
            VenueOverride(),
        )

    def unmatched(self, names: Iterable[str]) -> list[str]:
        """Entries naming no venue in `names`: a typo here would otherwise never apply."""
        known = {normalize_name(name) for name in names}
        return [venue for venue in self.venues if normalize_name(venue) not in known]


def load_overrides(text: str | None = None) -> SeedOverrides:
    """Parse `seeds.yaml` (the committed file unless `text` is given)."""
    if text is None:
        text = files("art_curator.ingest").joinpath("seeds.yaml").read_text(encoding="utf-8")
    return SeedOverrides.model_validate(yaml.safe_load(text) or {})


# --- sites ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """One website to discover. Seeds hang off `venue_id`, the pilot's own venue row."""

    venue_id: int
    name: str
    url: str  # `venues.website_url`: the homepage
    hints: GrafHints = GrafHints()
    seeds_in_use: tuple[str, ...] = ()  # accepted earlier and not rejected by a person since
    pinned: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()


async def load_sites(session: AsyncSession, overrides: SeedOverrides) -> list[Site]:
    """One `Site` per distinct `website_url` among `is_pilot AND crawl_enabled` venues.

    Hints come from every venue sharing that URL, not only the pilot row: MACBA's events hang off
    four terms. "Current" is decided in Postgres (CLAUDE.md § 8): an occurrence that has not ended.
    """
    venues = (
        await session.execute(
            select(VENUES.c.id, VENUES.c.name, VENUES.c.website_url, VENUES.c.is_pilot)
            .where(VENUES.c.crawl_enabled, VENUES.c.website_url.is_not(None))
            .order_by(VENUES.c.id)
        )
    ).all()
    pilots = {}
    for venue in venues:
        if venue.is_pilot:
            pilots.setdefault(venue.website_url, venue)
    if not pilots:
        return []

    current = exists().where(
        OCCURRENCES.c.event_id == EVENTS.c.id,
        func.coalesce(OCCURRENCES.c.ends_at, OCCURRENCES.c.starts_at) >= func.now(),
    )
    siblings = {v.id: v.website_url for v in venues if v.website_url in pilots}
    events: dict[str, list[Any]] = {}
    for event in await session.execute(
        select(
            EVENTS.c.venue_id,
            EVENTS.c.title,
            EVENTS.c.web_url_ca,
            EVENTS.c.web_url_es,
            EVENTS.c.web_url_en,
        )
        .where(EVENTS.c.venue_id.in_(list(siblings)), current)
        .order_by(EVENTS.c.id)
    ):
        events.setdefault(siblings[event.venue_id], []).append(event)

    accepted: dict[int, list[str]] = {}
    for venue_id, url in await session.execute(
        select(SEEDS.c.venue_id, SEEDS.c.url)
        .where(
            SEEDS.c.venue_id.in_([v.id for v in pilots.values()]),
            SEEDS.c.status == SeedStatus.ACCEPTED.value,
        )
        .order_by(SEEDS.c.id)
    ):
        accepted.setdefault(venue_id, []).append(url)

    sites = []
    for url, venue in pilots.items():
        override = overrides.for_venue(venue.name)
        rejected = {url_key(u) for u in override.reject}
        sites.append(
            Site(
                venue_id=venue.id,
                name=venue.name,
                url=url,
                hints=GrafHints.from_events(events.get(url, ()), url),
                seeds_in_use=tuple(
                    u for u in accepted.get(venue.id, ()) if url_key(u) not in rejected
                ),
                pinned=override.pin,
                rejected=override.reject,
            )
        )
    return sites


def select_sites(sites: Sequence[Site], venue: str | None) -> list[Site]:
    """`--venue`: the sites whose name matches, compared as `normalize_name`. None means all."""
    if venue is None:
        return list(sites)
    key = normalize_name(venue)
    return [site for site in sites if normalize_name(site.name) == key]


# --- one site ------------------------------------------------------------------------------------


class SiteOutcome(StrEnum):
    ACCEPTED = "accepted"  # at least one candidate accepted
    AMBIGUOUS = "ambiguous"  # a person decides, in seeds.yaml
    PINNED = "pinned"  # a person already decided; not discovered
    BLOCKED = "blocked"
    ROBOTS = "robots"
    THIN = "thin"  # the homepage has no links we can read: rendered by JavaScript
    UNAVAILABLE = "unavailable"
    ERROR = "error"


# A homepage we could not use, by what `fetch_page` said about it.
_HOMEPAGE_OUTCOMES = {
    PageStatus.BLOCKED: SiteOutcome.BLOCKED,
    PageStatus.ROBOTS: SiteOutcome.ROBOTS,
    PageStatus.UNAVAILABLE: SiteOutcome.UNAVAILABLE,
    PageStatus.ERROR: SiteOutcome.ERROR,
}

INVALID_ANSWER = "invalid model answer"


@dataclass(frozen=True)
class Skipped:
    """A candidate the model chose that could not be classified. `reason` is a status, never
    anything read off the page."""

    url: str
    reason: str


@dataclass(frozen=True)
class SiteResult:
    site: Site
    outcome: SiteOutcome
    candidates: tuple[Candidate, ...] = ()
    skipped: tuple[Skipped, ...] = ()
    detail: str = ""


async def discover_site(
    client: PoliteClient, robots: RobotsPolicy, llm: LlmClient, model: str, site: Site
) -> SiteResult:
    """Discover one site's listing page. Raises only for what is not the site's doing, such as
    the model endpoint refusing the call."""
    if site.pinned:
        return SiteResult(site, SiteOutcome.PINNED)

    home = await fetch_page(client, robots, site.url)
    if home.page is None:
        return SiteResult(site, _HOMEPAGE_OUTCOMES[home.status], detail=home.detail)
    # A homepage can be thin on words and still have a working menu, so only the links decide.
    links = collect_links(home.page.content, home.page.url, exclude=site.rejected)
    if not links:
        return SiteResult(site, SiteOutcome.THIN, detail="no links on the homepage")

    try:
        chosen = await pick_candidates(
            llm, model, venue_name=site.name, page_url=home.page.url, links=links, hints=site.hints
        )
    except DiscoveryError:
        return SiteResult(site, SiteOutcome.ERROR, detail=INVALID_ANSWER)

    verdicts = []
    skipped = []
    for link in chosen:
        fetched = await fetch_page(client, robots, link.url)
        if fetched.status is not PageStatus.OK:
            skipped.append(Skipped(link.url, fetched.detail))
            continue
        try:
            verdict = await classify_page(
                llm, model, venue_name=site.name, url=link.url, text=fetched.text, hints=site.hints
            )
        except DiscoveryError:
            skipped.append(Skipped(link.url, INVALID_ANSWER))
            continue
        verdicts.append((link.url, verdict))

    routing = route_site(verdicts, site.seeds_in_use)
    return SiteResult(site, SiteOutcome(routing.status), routing.candidates, tuple(skipped))


# --- venue_seeds ---------------------------------------------------------------------------------


async def write_seeds(session: AsyncSession, results: Sequence[SiteResult], model: str) -> int:
    """Upsert every classified candidate. Returns the number of rows written. Runs in the
    caller's transaction and never commits."""
    venue_ids = [r.site.venue_id for r in results if r.candidates]
    if not venue_ids:
        return 0
    # `/ca/agenda` and `/ca/agenda/` are one page: reuse the spelling already stored.
    stored: dict[tuple[int, str], str] = {
        (venue_id, url_key(url)): url
        for venue_id, url in await session.execute(
            select(SEEDS.c.venue_id, SEEDS.c.url).where(SEEDS.c.venue_id.in_(venue_ids))
        )
    }
    rows = [
        {
            "venue_id": result.site.venue_id,
            "url": stored.get((result.site.venue_id, url_key(c.url)), c.url),
            "page_type": c.verdict.page_type,
            "confidence": c.verdict.confidence,
            "dated_items": c.verdict.dated_items,
            "language": c.verdict.language,
            "status": c.status.value,
            "model": model,
        }
        for result in results
        for c in result.candidates
    ]
    stmt = insert(SEEDS).values(rows)
    keys = {"venue_id", "url"}
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[SEEDS.c.venue_id, SEEDS.c.url],
            set_={c: stmt.excluded[c] for c in rows[0] if c not in keys}
            | {"discovered_at": func.now()},
        )
    )
    return len(rows)


# --- the report ----------------------------------------------------------------------------------

_MARKS = {SeedStatus.ACCEPTED: "+", SeedStatus.REJECTED: "-", SeedStatus.AMBIGUOUS: "?"}


def report(results: Sequence[SiteResult], unmatched_overrides: Sequence[str] = ()) -> list[str]:
    """What `discover` prints: a count per outcome, then every site under its outcome.

    Candidates are marked `+` accepted, `-` rejected, `?` ambiguous. The `ambiguous` block is the
    one a person acts on, by pinning or rejecting URLs in `seeds.yaml`.
    """
    lines: list[str] = []

    def block(label: str, first: str, rest: Sequence[str] = ()) -> None:
        lines.append(f"{label:<11} {first}")
        lines.extend(f"{'':<11} {line}" for line in rest)

    counts = Counter(r.outcome for r in results)
    block(
        "sites",
        "   ".join([str(len(results))] + [f"{counts[o]} {o}" for o in SiteOutcome if counts[o]]),
    )
    for outcome in SiteOutcome:
        for result in sorted(
            (r for r in results if r.outcome is outcome), key=lambda r: normalize_name(r.site.name)
        ):
            site = result.site
            head = f"{site.name}   {result.detail}" if result.detail else site.name
            block(
                outcome,
                head,
                [f"pin {url}" for url in site.pinned if outcome is SiteOutcome.PINNED]
                + [_candidate_line(c) for c in result.candidates]
                + [f"! {s.url}   {s.reason}" for s in result.skipped]
                + (
                    [f"in use {url}" for url in site.seeds_in_use]
                    if outcome is SiteOutcome.AMBIGUOUS
                    else []
                ),
            )
    for name in unmatched_overrides:
        block("override", f"{name}: no crawlable pilot venue has this name")
    return lines


def _candidate_line(candidate: Candidate) -> str:
    verdict = candidate.verdict
    return (
        f"{_MARKS[candidate.status]} {candidate.url}   {verdict.page_type} {verdict.confidence}   "
        f"{verdict.dated_items} dated   {verdict.language}"
    )
