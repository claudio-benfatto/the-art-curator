"""Score extraction and discovery against hand-checked labels (PLAN.md § 8, P2).

The gold set, `eval/gold/extraction.yaml`, holds labels and nothing else. Per page: its URL, the
`content_hash` it was labelled against, and the items a person confirmed on the live page, as
facts (kind, title, dates, artists, `at_venue`). Per venue: the listing page(s) discovery should
have found. There is no page text and no summary in it. The text cannot outlive its 7-day TTL or
enter git (CLAUDE.md § 1), so a label is pinned to a hash instead, and goes **stale** when the
venue changes the page.

This module is pure. It parses the gold set, matches what extraction found against it, scores
discovery's verdicts against the labelled listings, and writes the report. Finding the pages'
text and calling the model is `eval/runner.py`.

How items are matched, page by page:

- **By title**, as `title_key`, so case, accents and punctuation do not count. Exact matches go
  first. A title that still differs is matched if one is a whole-word run inside the other
  ("Prova U" in "Prova U. Obra recent"): a model that drops a subtitle has still found the show.
- Among equal titles the same kind wins, then the same start date: a guided tour repeats under one
  title.
- One extracted item answers one label. A label left over is **missed**; an extracted item left
  over is **extra**.

What is scored, over every scored page together:

- item **recall** (matched / labelled) and **precision** (matched / extracted);
- a **partial** listing, one labelled only as far as its first items of each kind, counts towards
  recall and the fields but not precision: what extraction found past those items was never
  checked, so it cannot be called extra;
- on matched items: **dates** (start and end both exact), **artists** (the same names as
  `normalize_name`, in any order), **at_venue** and **kind**;
- items extracted from an **empty** page, one where a person found nothing: an invented show, the
  worst answer there is;
- a page whose extraction failed counts as a page that found nothing.

Discovery, per labelled venue, from what `discover` left in `venue_seeds`:

- **correct**: something was accepted, and every accepted URL is a labelled listing;
- **accepted-but-wrong**: an accepted URL is not one, and the crawl would follow it. The
  dangerous case;
- **ambiguous**: nothing was accepted, so a person had to decide. The human load;
- **undiscovered**: no verdict at all, because the venue was pinned first or never run.

Then, per `(page_type, confidence)`, how many candidates were labelled listings. That is what the
routing thresholds in `ingest/discover.py` are tuned against.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from importlib.resources import files
from typing import Annotated, Any, Literal, Self, get_args
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from art_curator.ingest.discover import Confidence, PageType, SeedStatus, url_key
from art_curator.ingest.extraction import title_key
from art_curator.ingest.matching import normalize_name
from art_curator.llm.extract import (
    MAX_LISTED_ITEMS,
    NAME_CHARS,
    TITLE_CHARS,
    Exhibition,
    Extraction,
)

GOLD_FILE = "gold/extraction.yaml"

# A title shorter than this is never matched by containment: "Art" is in too many titles.
MIN_CONTAINED_CHARS = 4

ItemKind = Literal["exhibition", "event"]


# --- the gold set --------------------------------------------------------------------------------

_LABEL = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _web_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"{url!r} is not an http(s) URL")
    return url


class GoldItem(BaseModel):
    """One item a person confirmed on the page. Facts only: no field can hold a sentence."""

    model_config = _LABEL

    kind: ItemKind = "exhibition"
    title: Annotated[str, StringConstraints(min_length=1, max_length=TITLE_CHARS)]
    artists: tuple[Annotated[str, StringConstraints(min_length=1, max_length=NAME_CHARS)], ...] = ()
    start: date | None = None
    end: date | None = None
    at_venue: bool = True

    @model_validator(mode="after")
    def _ends_after_it_starts(self) -> Self:
        if self.start and self.end and self.end < self.start:
            raise ValueError(f"{self.title!r} ends before it starts")
        return self


class GoldPage(BaseModel):
    """One labelled page. `items` is required, so a page that announces nothing says `[]`."""

    model_config = _LABEL

    url: str
    venue: str = Field(min_length=1)  # as in `venues.name`; extraction is told it, as in `extract`
    kind: Literal["listing", "detail"]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")  # what the labels were checked against
    confirmed: bool = False  # true once a person has checked every item on the live page
    partial: bool = False  # a listing labelled only as far as its first items of each kind
    items: tuple[GoldItem, ...]

    @field_validator("url")
    @classmethod
    def _url(cls, url: str) -> str:
        return _web_url(url)

    @model_validator(mode="after")
    def _as_extraction_reads_it(self) -> Self:
        """A listing is labelled the way extraction reads it: its first `MAX_LISTED_ITEMS`
        exhibitions and first `MAX_LISTED_ITEMS` events in page order."""
        if self.kind == "listing":
            counts = Counter(item.kind for item in self.items)
            over = [kind for kind, n in counts.items() if n > MAX_LISTED_ITEMS]
            if over:
                raise ValueError(
                    f"{self.url}: more than {MAX_LISTED_ITEMS} {over[0]}s. Label only the first "
                    f"{MAX_LISTED_ITEMS} in page order, as extraction keeps"
                )
        if self.partial and self.kind != "listing":
            raise ValueError(f"{self.url}: only a listing can be labelled partially")
        if self.partial and not self.items:
            raise ValueError(f"{self.url}: a partial label needs its first items")
        return self

    @property
    def listing(self) -> bool:
        return self.kind == "listing"


class GoldSet(BaseModel):
    model_config = _LABEL

    # Venue name -> its listing pages. An empty list: the venue has no listing we can use.
    listings: dict[str, tuple[str, ...]] = {}
    pages: tuple[GoldPage, ...] = ()

    @field_validator("listings", "pages", mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            return {} if info.field_name == "listings" else ()
        return value

    @field_validator("listings", mode="before")
    @classmethod
    def _one_or_many(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {
            venue: (urls,) if isinstance(urls, str) else (urls or ())
            for venue, urls in value.items()
        }

    @field_validator("listings")
    @classmethod
    def _listing_urls(cls, value: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        for urls in value.values():
            for url in urls:
                _web_url(url)
        repeated = [k for k, n in Counter(map(normalize_name, value)).items() if n > 1]
        if repeated:
            raise ValueError(f"more than one listings entry for {repeated}")
        return value

    @field_validator("pages")
    @classmethod
    def _one_label_per_page(cls, value: tuple[GoldPage, ...]) -> tuple[GoldPage, ...]:
        repeated = [k for k, n in Counter(url_key(p.url) for p in value).items() if n > 1]
        if repeated:
            raise ValueError(f"more than one label for {repeated}")
        return value

    def for_venue(self, venue: str | None) -> "GoldSet":
        """`--venue`: only this venue's pages and listings, compared as `normalize_name`."""
        if venue is None:
            return self
        key = normalize_name(venue)
        return GoldSet(
            listings={v: urls for v, urls in self.listings.items() if normalize_name(v) == key},
            pages=tuple(p for p in self.pages if normalize_name(p.venue) == key),
        )

    @property
    def venues(self) -> list[str]:
        names = {normalize_name(n): n for n in [*self.listings, *(p.venue for p in self.pages)]}
        return sorted(names.values(), key=normalize_name)


def load_gold(text: str | None = None) -> GoldSet:
    """Parse the gold set (the committed file unless `text` is given)."""
    if text is None:
        text = files("art_curator.eval").joinpath(GOLD_FILE).read_text(encoding="utf-8")
    return GoldSet.model_validate(yaml.safe_load(text) or {})


# --- matching ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """An item, labelled or extracted, in the form the scorer compares."""

    kind: ItemKind
    title: str
    artists: tuple[str, ...]
    start: date | None
    end: date | None
    at_venue: bool


def labelled(item: GoldItem) -> Item:
    return Item(item.kind, item.title, item.artists, item.start, item.end, item.at_venue)


def extracted(extraction: Extraction) -> list[Item]:
    """What extraction found, exhibitions then events, each in page order."""

    def item(kind: ItemKind, found: Exhibition) -> Item:
        return Item(
            kind, found.title, found.artists, found.start_date, found.end_date, found.at_venue
        )

    return [item("exhibition", show) for show in extraction.exhibitions] + [
        item("event", event) for event in extraction.events
    ]


def _artist_names(artists: Iterable[str]) -> set[str]:
    return {normalize_name(name) for name in artists} - {""}


@dataclass(frozen=True)
class Match:
    label: Item
    found: Item

    @property
    def dates_ok(self) -> bool:
        return (self.label.start, self.label.end) == (self.found.start, self.found.end)

    @property
    def artists_ok(self) -> bool:
        return _artist_names(self.label.artists) == _artist_names(self.found.artists)

    @property
    def at_venue_ok(self) -> bool:
        return self.label.at_venue == self.found.at_venue

    @property
    def kind_ok(self) -> bool:
        return self.label.kind == self.found.kind


def same_title(a: str, b: str) -> bool:
    return title_key(a) == title_key(b)


def contained_title(a: str, b: str) -> bool:
    """One title is a whole-word run inside the other: `prova u` is in `prova u obra recent`,
    and `art` is not in `martí`. `title_key` separates words with single spaces."""
    shorter, longer = sorted((title_key(a), title_key(b)), key=len)
    return len(shorter) >= MIN_CONTAINED_CHARS and f" {shorter} " in f" {longer} "


def match_items(
    labels: Sequence[Item], found: Sequence[Item]
) -> tuple[list[Match], list[Item], list[Item]]:
    """`(matches, missed, extra)`: every exact title first, then containment for what is left."""
    left = list(found)
    matches: list[Match] = []
    unmatched = list(labels)
    for titles_agree in (same_title, contained_title):
        still_unmatched = []
        for label in unmatched:
            candidates = [item for item in left if titles_agree(label.title, item.title)]
            if not candidates:
                still_unmatched.append(label)
                continue
            # `max` keeps the first of equals, so page order breaks the last tie.
            best = max(
                candidates, key=lambda item: (item.kind == label.kind, item.start == label.start)
            )
            left.remove(best)
            matches.append(Match(label, best))
        unmatched = still_unmatched
    return matches, unmatched, left


# --- one page ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PageScore:
    page: GoldPage
    matches: tuple[Match, ...]
    missed: tuple[Item, ...]
    extra: tuple[Item, ...]
    error: str = ""  # extraction failed, and the page counts as one that found nothing

    @property
    def found(self) -> int:
        return len(self.matches) + len(self.extra)


def score_page(page: GoldPage, extraction: Extraction | None, error: str = "") -> PageScore:
    found = extracted(extraction) if extraction is not None else []
    matches, missed, extra = match_items([labelled(item) for item in page.items], found)
    return PageScore(page, tuple(matches), tuple(missed), tuple(extra), error)


class SkipReason(StrEnum):
    """Why a labelled page was not scored. The value is what the report prints."""

    UNCONFIRMED = "not confirmed by a person yet"
    STALE = "changed since it was labelled: label it again"
    UNREADABLE = "could not be fetched"


@dataclass(frozen=True)
class Skipped:
    page: GoldPage
    reason: SkipReason
    detail: str = ""  # a fetch status, never anything read off the page


# --- every page ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Ratio:
    hits: int
    total: int

    def __str__(self) -> str:
        if not self.total:
            return "n/a (0/0)"
        return f"{self.hits / self.total:.2f} ({self.hits}/{self.total})"


@dataclass(frozen=True)
class Totals:
    recall: Ratio
    precision: Ratio
    dates: Ratio
    artists: Ratio
    at_venue: Ratio
    kind: Ratio
    empty_pages: int
    invented: int  # items extracted from empty pages
    failed: int


def totals(scores: Sequence[PageScore]) -> Totals:
    matches = [match for score in scores for match in score.matches]

    def on_matches(ok: str) -> Ratio:
        return Ratio(sum(getattr(match, ok) for match in matches), len(matches))

    empty = [score for score in scores if not score.page.items]
    complete = [score for score in scores if not score.page.partial]
    return Totals(
        recall=Ratio(len(matches), sum(len(score.page.items) for score in scores)),
        precision=Ratio(
            sum(len(score.matches) for score in complete), sum(score.found for score in complete)
        ),
        dates=on_matches("dates_ok"),
        artists=on_matches("artists_ok"),
        at_venue=on_matches("at_venue_ok"),
        kind=on_matches("kind_ok"),
        empty_pages=len(empty),
        invented=sum(score.found for score in empty),
        failed=sum(bool(score.error) for score in scores),
    )


def on_or_coming(item: GoldItem, day: date) -> bool:
    """Not over on `day`, as `queries/events.py` means it: an exhibition with no end date may
    still be on, and an event with no end date is over after its start date."""
    last = item.end or (item.start if item.kind == "event" else None)
    return item.at_venue and (last is None or last >= day)


# --- discovery -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedVerdict:
    """One `venue_seeds` row: what discovery last concluded about one candidate URL."""

    venue_name: str
    url: str
    page_type: str
    confidence: str
    status: str


class Outcome(StrEnum):
    CORRECT = "correct"
    WRONG = "accepted-but-wrong"
    AMBIGUOUS = "ambiguous"
    UNDISCOVERED = "undiscovered"


@dataclass(frozen=True)
class VenueOutcome:
    venue: str
    outcome: Outcome
    wrong: tuple[str, ...] = ()  # accepted URLs that are not labelled listings


@dataclass(frozen=True)
class Calibration:
    page_type: str
    confidence: str
    candidates: int
    listings: int  # of them, labelled listings


@dataclass(frozen=True)
class DiscoveryScore:
    venues: tuple[VenueOutcome, ...] = ()
    calibration: tuple[Calibration, ...] = ()
    unlabelled: tuple[str, ...] = ()  # venues discovery has verdicts for, with no labelled listing

    def share(self, outcome: Outcome) -> Ratio:
        """Out of the venues discovery has a verdict for."""
        decided = [v for v in self.venues if v.outcome is not Outcome.UNDISCOVERED]
        return Ratio(sum(v.outcome is outcome for v in decided), len(decided))


def score_discovery(
    listings: Mapping[str, Sequence[str]], verdicts: Sequence[SeedVerdict]
) -> DiscoveryScore:
    by_venue: dict[str, list[SeedVerdict]] = {}
    for verdict in verdicts:
        by_venue.setdefault(normalize_name(verdict.venue_name), []).append(verdict)

    venues = []
    seen: Counter[tuple[str, str]] = Counter()
    right: Counter[tuple[str, str]] = Counter()
    for venue, urls in sorted(listings.items(), key=lambda entry: normalize_name(entry[0])):
        rows = by_venue.get(normalize_name(venue), [])
        listed = {url_key(url) for url in urls}
        accepted = [row.url for row in rows if row.status == SeedStatus.ACCEPTED]
        wrong = tuple(url for url in accepted if url_key(url) not in listed)
        if not rows:
            outcome = Outcome.UNDISCOVERED
        elif wrong:
            outcome = Outcome.WRONG
        elif accepted:
            outcome = Outcome.CORRECT
        else:
            outcome = Outcome.AMBIGUOUS
        venues.append(VenueOutcome(venue, outcome, wrong))
        for row in rows:
            seen[row.page_type, row.confidence] += 1
            right[row.page_type, row.confidence] += url_key(row.url) in listed

    calibration = []
    for page_type in get_args(PageType):
        for confidence in get_args(Confidence):
            key = (page_type, confidence)
            if seen[key]:
                calibration.append(Calibration(page_type, confidence, seen[key], right[key]))
    known = {normalize_name(venue) for venue in listings}
    unlabelled = sorted(
        {rows[0].venue_name for key, rows in by_venue.items() if key not in known},
        key=normalize_name,
    )
    return DiscoveryScore(tuple(venues), tuple(calibration), tuple(unlabelled))


# --- the report ----------------------------------------------------------------------------------


def _days(start: date | None, end: date | None) -> str:
    return f"{start or '?'} – {end or '?'}"


def _mismatches(match: Match) -> list[tuple[str, str]]:
    """`(field, what differs)` for each field of a matched item that is wrong."""
    label, found = match.label, match.found
    checks = [
        (match.dates_ok, "dates", f"{_days(label.start, label.end)}, got "
                                  f"{_days(found.start, found.end)}"),
        (match.artists_ok, "artists", f"{', '.join(label.artists) or 'none'}, got "
                                      f"{', '.join(found.artists) or 'none'}"),
        (match.at_venue_ok, "at_venue", f"{label.at_venue}, got {found.at_venue}"),
        (match.kind_ok, "kind", f"{label.kind}, got {found.kind}"),
    ]  # fmt: skip
    return [(name, detail) for ok, name, detail in checks if not ok]


def report(
    gold: GoldSet,
    today: date,
    scores: Sequence[PageScore],
    skipped: Sequence[Skipped],
    discovery: DiscoveryScore,
    stopped: str = "",
) -> list[str]:
    """What `eval-extraction` prints: what the gold set covers, the scores, every scored page
    with what it got wrong, the pages that were not scored, then discovery.

    Titles, artists and dates are printed: they are facts, and in the gold file already. Page
    text never is."""
    lines: list[str] = []

    def block(label: str, *parts: str) -> None:
        lines.append(f"{label:<11} " + "   ".join(part for part in parts if part))

    def detail(label: str, *parts: str) -> None:
        lines.append(f"{'':<11} {label:<9} " + "   ".join(part for part in parts if part))

    confirmed = [page for page in gold.pages if page.confirmed]
    unconfirmed = len(gold.pages) - len(confirmed)
    no_current = [
        page for page in confirmed if not any(on_or_coming(item, today) for item in page.items)
    ]
    partial = sum(page.partial for page in confirmed)
    block(
        "gold",
        f"{len(confirmed)} pages",
        f"{len({normalize_name(page.venue) for page in confirmed})} venues",
        f"{len(no_current)} with no current show",
        f"{sum(not page.items for page in confirmed)} empty",
        f"{partial} partial" if partial else "",
        f"{unconfirmed} unconfirmed" if unconfirmed else "",
    )
    reasons = Counter(skip.reason for skip in skipped)
    block(
        "scored",
        f"{len(scores)} pages",
        *(f"{reasons[r]} {r.name.lower()}" for r in SkipReason if reasons[r]),
    )

    if scores:
        total = totals(scores)
        block("recall", str(total.recall))
        block("precision", str(total.precision))
        block("dates", str(total.dates), "on matched items")
        block("artists", str(total.artists))
        block("at_venue", str(total.at_venue))
        block("kind", str(total.kind))
        block("invented", f"{total.invented} items on {total.empty_pages} empty pages")
        if total.failed:
            block("failed", f"{total.failed} pages, scored as finding nothing")

    by_venue: dict[str, list[PageScore]] = {}
    for score in scores:
        by_venue.setdefault(score.page.venue, []).append(score)
    for venue in sorted(by_venue, key=normalize_name):
        block("site", venue)
        for score in by_venue[venue]:
            if score.error:
                block("failed", score.page.url, score.error)
            else:
                block(
                    "ok",
                    score.page.url,
                    f"{len(score.matches)} of {len(score.page.items)} labelled",
                    f"{score.found} extracted",
                    "partial: extras not judged" if score.page.partial else "",
                )
            for item in score.missed:
                detail("missed", repr(item.title))
            if not score.page.partial:
                for item in score.extra:
                    detail("extra", repr(item.title))
            for match in score.matches:
                for name, what in _mismatches(match):
                    detail(name, repr(match.label.title), what)

    for skip in skipped:
        block(skip.reason.name.lower(), skip.page.url, skip.reason, skip.detail)

    if gold.listings:
        outcomes = Counter(venue.outcome for venue in discovery.venues)
        block(
            "discovery",
            f"{len(discovery.venues)} labelled venues",
            *(f"{outcomes[o]} {o}" for o in Outcome if outcomes[o]),
        )
        block("accuracy", str(discovery.share(Outcome.CORRECT)), "of venues with a verdict")
        block("ambiguous", str(discovery.share(Outcome.AMBIGUOUS)), "left for a person")
        block("wrong", str(discovery.share(Outcome.WRONG)), "accepted a page that is no listing")
        for venue in discovery.venues:
            for url in venue.wrong:
                detail("wrong", venue.venue, url)
        for row in discovery.calibration:
            block(
                "verdict",
                f"{row.page_type} {row.confidence}",
                f"{row.candidates} candidates",
                f"{row.listings} labelled listings",
            )
        for name in discovery.unlabelled:
            block("unlabelled", name, "discovery has verdicts for it, but no listing is labelled")

    if stopped:
        block("stopped", f"the endpoint refused a call: {stopped}")
    return lines
