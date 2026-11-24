"""Score the curator over scripted conversations (PLAN.md § 8, P3).

The scenarios, `eval/gold/curation.yaml`, are conversations a person wrote: who is asking
(`persona`), what the curator remembers of them before it starts (`profile`), and what they say,
turn by turn. With each turn go its **hard constraints**: the dates, area, price and kinds of
thing the user asked for. Dates are relative ("saturday", "today+13") and resolved against the
day the eval runs. A scenario never names a show: expected answers rot as shows close, and
constraints do not.

This module is pure: the scenarios, what a turn's messages say the tools returned, which of those
items the reply names, and the report. Running the conversations, checking the constraints in SQL
and calling the judge is `eval/curation_runner.py`.

Per exchange (one user turn and the curator's reply):

- **Named items**: the items the tools returned in the conversation so far whose title, as the
  tool gave it, appears in the reply, compared as `title_key` over whole words. A title is also
  named by one piece of it at ". " ("Title. Artist", either piece, in either order), if that piece
  has two words or more and no other title seen has it. A title the reply translates is not
  counted. A title found only inside a longer named one ("Aurèlia Muñoz. Ens" inside "Visites
  comentades a Aurèlia Muñoz. Ens") is not named on its own. A named item may still be one the
  reply advises against: the transcript shows which.
- **Off constraint**: a named item outside the turn's dates, area, price or kinds, decided in SQL
  by the runner. A title passes a constraint if any item under it does: a GRAF listing and the
  venue's own record of a show share a title, and so do the dates of a repeated tour.
- **Order**: whether the named items come nearest first, by the distances the search gave.
- **Checks**: whether the turn searched (a search that came back as an error does not count),
  searched from the place it should, and remembered exactly the topics it should, where the
  scenario says.
- **Judge** (`llm/curation_judge.py`): taste fit, and the reply's claims nothing on record
  supports, which is where an invented item shows.
- **Cost**: tool rounds, model calls, tokens and cache, cost and latency, from `llm_calls`.

Constraints carry over from turn to turn, so a turn states only what changed ("and on Sunday?"
restates `dates`); `null` drops one. `searches`, `searches_near` and `remembers` are about one
turn and never carry.

Every reply and transcript holds our own text, facts and our summaries: no venue page reaches a
conversation (CLAUDE.md § 1).
"""

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from importlib.resources import files
from statistics import median
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from art_curator.agent.prompt import WEEKDAYS
from art_curator.db.models import PROFILE_TOPICS
from art_curator.ingest.extraction import title_key
from art_curator.llm.curation_judge import TASTES, ToolUse, Verdict
from art_curator.queries.events import KINDS

SCENARIOS_FILE = "gold/curation.yaml"

# Opus 4.6's minimum cacheable prefix (CLAUDE.md § 6). A call can read the cache only once an
# earlier call of the conversation sent a prefix at least this long.
CACHE_MIN_TOKENS = 4096

# A title shorter than this is never looked for in a reply: "Arxiu" is a word, "U" is a letter.
MIN_TITLE_CHARS = 4

# The date range of a check that is not about dates: wider than any record's.
ANY_DAY_FROM = date(1900, 1, 1)
ANY_DAY_TO = date(2999, 12, 31)
CHECK_LIMIT = 100_000  # rows a check may return: all of them

_DAY = re.compile(
    r"^(?P<day>today|tomorrow|" + "|".join(day.lower() for day in WEEKDAYS) + r")"
    r"(?:\+(?P<plus>[0-9]+))?$"
)


def resolve_day(spec: str, today: date, on_or_after: date | None = None) -> date:
    """The day `spec` names, seen from `today`. A weekday is the first on or after `on_or_after`
    (today, unless given), which is the one the calendar lists first; `+N` adds N days to it:
    `today+13` is the calendar's last day."""
    found = _DAY.match(spec)
    if found is None:
        raise ValueError(f"{spec!r} is not a day: today, tomorrow or a weekday, then +N if needed")
    start = on_or_after or today
    match found["day"]:
        case "today":
            day = today
        case "tomorrow":
            day = today + timedelta(days=1)
        case weekday:
            wanted = [name.lower() for name in WEEKDAYS].index(weekday)
            day = start + timedelta(days=(wanted - start.weekday()) % 7)
    return day + timedelta(days=int(found["plus"] or 0))


# --- the scenarios -------------------------------------------------------------------------------

_SCENARIO = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

DaySpec = Annotated[str, StringConstraints(pattern=_DAY.pattern)]
Kind = Literal[KINDS]
Topic = Literal[PROFILE_TOPICS]


class Area(BaseModel):
    """Within `km` of the place `near` names, as `search_events` resolves it."""

    model_config = _SCENARIO

    near: str = Field(min_length=1)
    km: float = Field(gt=0, le=200)


class Expect(BaseModel):
    """What one turn asks for. The first five are hard constraints on every item the reply names,
    and carry over to the turns after; the last three are checks on this turn alone."""

    model_config = _SCENARIO

    # One day, or [first, last]. `last` is resolved from `first`: [saturday, sunday] is a weekend
    # even on a Sunday.
    dates: tuple[DaySpec, DaySpec] | None = None
    area: Area | None = None
    free: bool | None = None  # true: nothing known to cost money; an unknown price passes
    kinds: tuple[Kind, ...] | None = None
    # One day, or [first, last], as `dates`: every item named must end on one of those days.
    # An item with no end date does not.
    ends: tuple[DaySpec, DaySpec] | None = None
    # This turn calls search_events successfully (true) or does not (false). A search the tool
    # refused, such as a place it could not pin down, does not count.
    searches: bool | None = None
    # Some successful search this turn measures from this place, as `search_events` resolves
    # `near`. For where the user is, without the distance limit `area` would set.
    searches_near: str | None = Field(default=None, min_length=1)
    # Exactly the topics this turn must remember: one missing or one more is a failure. []
    # means nothing.
    remembers: tuple[Topic, ...] | None = None

    @field_validator("dates", "ends", mode="before")
    @classmethod
    def _one_day_or_two(cls, value: Any) -> Any:
        return (value, value) if isinstance(value, str) else value

    @field_validator("dates", "ends")
    @classmethod
    def _forwards(cls, value: tuple[str, str] | None) -> tuple[str, str] | None:
        if value is not None:
            # Whatever day of the week the eval runs on.
            for offset in range(7):
                today = date(2026, 1, 5) + timedelta(days=offset)
                first = resolve_day(value[0], today)
                if resolve_day(value[1], today, on_or_after=first) < first:
                    raise ValueError(f"{list(value)} runs backwards on a {WEEKDAYS[offset]}")
        return value


CARRIED = ("dates", "area", "free", "kinds", "ends")


class Fact(BaseModel):
    """A fact the curator remembers about the person before the scenario starts."""

    model_config = _SCENARIO

    topic: Topic
    fact: str = Field(min_length=1, max_length=200)


class ScenarioTurn(BaseModel):
    model_config = _SCENARIO

    user: str = Field(min_length=1)
    expect: Expect = Expect()


@dataclass(frozen=True)
class ConstraintCheck:
    """One constraint, as a search: an item meets it if the search returns a row of it that
    `meets` accepts."""

    constraint: str  # dates, area, price, kinds or ends
    arguments: dict[str, Any]  # for search_events
    ends_between: tuple[date, date] | None = None  # `ends`: the row's end must be one of these

    def meets(self, end: date | None) -> bool:
        """Whether a row the search returned, ending on `end`, meets the constraint."""
        if self.ends_between is None:
            return True
        first, last = self.ends_between
        return end is not None and first <= end <= last


def _span(first: date, last: date) -> str:
    return f"{first}" if first == last else f"{first} – {last}"


@dataclass(frozen=True)
class Constraints:
    """A turn's hard constraints with its dates resolved: what every item it names must meet."""

    date_from: date | None = None
    date_to: date | None = None
    area: Area | None = None
    free: bool = False
    kinds: tuple[str, ...] | None = None
    ends_from: date | None = None
    ends_to: date | None = None

    def checks(self) -> list[ConstraintCheck]:
        """One check for each constraint there is. One search per constraint, wide on everything
        else, so an item off two constraints is reported off both."""
        wide: dict[str, Any] = {
            "date_from": ANY_DAY_FROM,
            "date_to": ANY_DAY_TO,
            "limit": CHECK_LIMIT,
        }
        checks = []
        if self.date_from is not None:
            dates = {**wide, "date_from": self.date_from, "date_to": self.date_to}
            checks.append(ConstraintCheck("dates", dates))
        if self.area is not None:
            area = {**wide, "near": self.area.near, "radius_km": self.area.km}
            checks.append(ConstraintCheck("area", area))
        if self.free:
            checks.append(ConstraintCheck("price", {**wide, "free_only": True}))
        if self.kinds is not None:
            checks.append(ConstraintCheck("kinds", {**wide, "kinds": list(self.kinds)}))
        if self.ends_from is not None and self.ends_to is not None:
            # What ends on one of those days is on that day, so the search finds it there; its
            # end, which the search reads off the record, says whether it ends then.
            on_those_days = {**wide, "date_from": self.ends_from, "date_to": self.ends_to}
            between = (self.ends_from, self.ends_to)
            checks.append(ConstraintCheck("ends", on_those_days, ends_between=between))
        return checks

    def __str__(self) -> str:
        parts = []
        if self.date_from is not None and self.date_to is not None:
            parts.append(_span(self.date_from, self.date_to))
        if self.area is not None:
            parts.append(f"within {self.area.km:g} km of {self.area.near}")
        if self.free:
            parts.append("free")
        if self.kinds is not None:
            parts.append("/".join(self.kinds))
        if self.ends_from is not None and self.ends_to is not None:
            parts.append(f"ending {_span(self.ends_from, self.ends_to)}")
        return ", ".join(parts) or "none"


class Scenario(BaseModel):
    model_config = _SCENARIO

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    persona: str = Field(min_length=1)  # for the judge and the person reading the transcript
    profile: tuple[Fact, ...] = ()
    turns: tuple[ScenarioTurn, ...] = Field(min_length=1)

    def constraints(self, today: date) -> list[Constraints]:
        """Each turn's constraints, carried over from the turns before and resolved on `today`."""
        carried: dict[str, Any] = {}
        resolved = []
        for turn in self.turns:
            for name in CARRIED:
                if name in turn.expect.model_fields_set:
                    carried[name] = getattr(turn.expect, name)
            date_from, date_to = _resolve_span(carried.get("dates"), today)
            ends_from, ends_to = _resolve_span(carried.get("ends"), today)
            resolved.append(
                Constraints(
                    date_from=date_from,
                    date_to=date_to,
                    area=carried.get("area"),
                    free=bool(carried.get("free")),
                    kinds=carried.get("kinds"),
                    ends_from=ends_from,
                    ends_to=ends_to,
                )
            )
        return resolved


def _resolve_span(span: tuple[str, str] | None, today: date) -> tuple[date | None, date | None]:
    """`[first, last]` as days seen from `today`, `last` counted from `first`; (None, None) for
    no span."""
    if span is None:
        return None, None
    first = resolve_day(span[0], today)
    return first, resolve_day(span[1], today, on_or_after=first)


class Scenarios(BaseModel):
    model_config = _SCENARIO

    scenarios: tuple[Scenario, ...]

    @field_validator("scenarios")
    @classmethod
    def _unique_ids(cls, value: tuple[Scenario, ...]) -> tuple[Scenario, ...]:
        repeated = [i for i, n in Counter(s.id for s in value).items() if n > 1]
        if repeated:
            raise ValueError(f"more than one scenario called {repeated}")
        return value

    def select(self, ids: Sequence[str]) -> list[Scenario]:
        """`--scenario`: those ids, in the file's order; every scenario when `ids` is empty.
        An unknown id raises ValueError."""
        if not ids:
            return list(self.scenarios)
        known = {s.id for s in self.scenarios}
        if unknown := [i for i in ids if i not in known]:
            raise ValueError(f"no scenario called {unknown}. Known: {', '.join(sorted(known))}")
        return [s for s in self.scenarios if s.id in ids]


def scenario_places(scenarios: Sequence[Scenario]) -> list[str]:
    """Every place the scenarios check against (`area` and `searches_near`), once each."""
    names = []
    for scenario in scenarios:
        for turn in scenario.turns:
            if turn.expect.area is not None:
                names.append(turn.expect.area.near)
            if turn.expect.searches_near is not None:
                names.append(turn.expect.searches_near)
    return list(dict.fromkeys(names))


def load_scenarios(text: str | None = None) -> Scenarios:
    """Parse the scenarios (the committed file unless `text` is given)."""
    if text is None:
        text = files("art_curator.eval").joinpath(SCENARIOS_FILE).read_text(encoding="utf-8")
    return Scenarios.model_validate(yaml.safe_load(text) or {})


# --- what a turn did -----------------------------------------------------------------------------


def tool_uses(messages: Iterable[Mapping[str, Any]]) -> list[ToolUse]:
    """Every tool call in `messages`, paired with the result that answered it, in order."""
    calls: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        for block in message["content"]:
            if block.get("type") == "tool_use":
                calls[block["id"]] = block
            elif block.get("type") == "tool_result":
                results[block["tool_use_id"]] = block
    return [
        ToolUse(
            name=call["name"],
            input=call["input"],
            result=results[call_id]["content"],
            is_error=results[call_id].get("is_error", False),
        )
        for call_id, call in calls.items()
        if call_id in results
    ]


@dataclass(frozen=True)
class SeenItem:
    """An item a tool returned in the conversation."""

    id: str
    titles: tuple[str, ...]  # its title, and GRAF's English one when get_events gave it
    distance_km: float | None  # from the last search that measured it


def seen_items(uses: Iterable[ToolUse]) -> dict[str, SeenItem]:
    """Every item `search_events` or `get_events` returned in `uses`, by id."""
    seen: dict[str, SeenItem] = {}
    for use in uses:
        if use.name not in ("search_events", "get_events") or use.is_error:
            continue
        for row in use.parsed()["events"]:
            before = seen.get(row["id"])
            titles = [*(before.titles if before else ()), row["title"]]
            if row.get("title_en"):
                titles.append(row["title_en"])
            distance = row.get("distance_km")
            if distance is None and before is not None:
                distance = before.distance_km
            seen[row["id"]] = SeenItem(row["id"], tuple(dict.fromkeys(titles)), distance)
    return seen


@dataclass(frozen=True)
class Named:
    """A title the reply names, and every item seen under it."""

    title: str  # as the tool gave it
    ids: tuple[str, ...]
    distance_km: float | None  # the nearest of its items, by the search's measure


def title_parts(title: str) -> list[str]:
    """The pieces of a title at ". ": GRAF writes "Title. Artist", and a reply may give either
    piece alone, or both the other way round ("Jordi Alcaraz – Misteri d'una forma"). Not split
    at dashes: "Espiral del temps – Plaça dels Àngels – MACBA" would be named by an address."""
    return [part for part in title.split(". ") if part]


def named_items(reply: str, seen: Mapping[str, SeenItem]) -> list[Named]:
    """The titles of `seen` the reply names, in the order it first names them. A title is named
    by itself, or by one of its parts (`title_parts`) of two words or more that no other title
    seen has, whole or within it: "Jordi Alcaraz" names "Misteri d'una forma. Jordi Alcaraz"
    unless he has a second show among them, and "Aurèlia Muñoz" names neither "Aurèlia Muñoz.
    Ens" nor "Els Ocells-estels d'Aurèlia Muñoz" when both were seen.

    An item is named once, under its first title, though the reply may name it by either of its
    titles (GRAF's own and its English one)."""
    items: dict[str, list[SeenItem]] = {}  # title key -> every item seen under it
    shown: dict[str, str] = {}  # title key -> the title as the tool gave it
    counted_as: dict[str, str] = {}  # title key -> the key it is counted under: its item's first
    owners: dict[str, set[str]] = {}  # part key -> the keys counted for the titles with that part
    for item in seen.values():
        keyed = [(title_key(title), title) for title in item.titles]
        usable = [(key, title) for key, title in keyed if len(key) >= MIN_TITLE_CHARS]
        if not usable:
            continue
        first = usable[0][0]
        for key, title in usable:
            items.setdefault(key, []).append(item)
            shown.setdefault(key, title)
            counted_as.setdefault(key, first)
            for part in title_parts(title):
                owners.setdefault(title_key(part), set()).add(counted_as[key])

    def in_another_title(part: str, owner: str) -> bool:
        return any(_spans(part, f" {key} ") for key in items if counted_as[key] != owner)

    # What a reply may write for each title: its whole key, then the parts only it has.
    names = {key: counted_as[key] for key in items}
    for part, counted in owners.items():
        if part in names or len(counted) != 1 or len(part.split()) < 2:
            continue
        [owner] = counted
        if not in_another_title(part, owner):
            names[part] = owner

    text = f" {title_key(reply)} "
    found = {name: spans for name in names if (spans := _spans(name, text))}

    def only_inside_longer_names(name: str) -> bool:
        longer = [
            span for other, spans in found.items() if len(other) > len(name) for span in spans
        ]
        return all(
            any(outer[0] <= start and end <= outer[1] for outer in longer)
            for start, end in found[name]
        )

    first_named: dict[str, int] = {}  # counted key -> where the reply first names it, by any name
    for name, spans in found.items():
        if not only_inside_longer_names(name):
            key = names[name]
            first_named[key] = min(first_named.get(key, spans[0][0]), spans[0][0])

    result = []
    for counted in sorted(first_named, key=first_named.__getitem__):
        under = [item for key, its in items.items() if counted_as[key] == counted for item in its]
        ids = tuple(dict.fromkeys(item.id for item in under))
        distances = [item.distance_km for item in under if item.distance_km is not None]
        result.append(Named(shown[counted], ids, min(distances) if distances else None))
    return result


def _spans(key: str, text: str) -> list[tuple[int, int]]:
    """Where `key` stands in `text` as whole words, `text` being spaced at both ends. Overlapping
    mentions count: " a a " holds " a " twice."""
    needle = f" {key} "
    return [
        (found.start(), found.start() + len(needle))
        for found in re.finditer(f"(?={re.escape(needle)})", text)
    ]


def in_distance_order(named: Sequence[Named]) -> bool | None:
    """Whether the named items with a distance come nearest first; None with fewer than two."""
    distances = [item.distance_km for item in named if item.distance_km is not None]
    if len(distances) < 2:
        return None
    return distances == sorted(distances)


def search_origins(uses: Iterable[ToolUse]) -> list[dict[str, Any]]:
    """The place each successful search measured from (its result's `near`), in order. A search
    without `near` measured from nowhere and is left out."""
    origins = []
    for use in uses:
        if use.name == "search_events" and not use.is_error:
            near = use.parsed().get("near")
            if near is not None:
                origins.append(near)
    return origins


def check_failures(expect: Expect, uses: Sequence[ToolUse]) -> list[str]:
    """What the turn did that the scenario says it should not have, or did not do that it
    should. `searches_near` needs the database, so the runner checks it (`near_failures`)."""
    failures = []
    searched = any(use.name == "search_events" and not use.is_error for use in uses)
    if expect.searches is True and not searched:
        failures.append("did not search")
    if expect.searches is False and searched:
        failures.append("searched, which it should not have")
    if expect.remembers is not None:
        remembered = {
            str(use.input.get("topic"))
            for use in uses
            if use.name == "remember" and not use.is_error
        }
        missing = [topic for topic in expect.remembers if topic not in remembered]
        extra = sorted(remembered - set(expect.remembers))
        failures += [f"did not remember {topic}" for topic in missing]
        if extra:
            failures.append(f"remembered {', '.join(extra)}, which it should not have")
    return failures


def checks_made(expect: Expect) -> int:
    """How many checks a turn with `expect` gets: one for `searches`, one for `searches_near`,
    and for `remembers` one per topic to remember plus one for "nothing else"."""
    made = 0
    if expect.searches is not None:
        made += 1
    if expect.searches_near is not None:
        made += 1
    if expect.remembers is not None:
        made += len(expect.remembers) + 1
    return made


# --- a run ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallUsage:
    """One model call of a turn, from its `llm_calls` row."""

    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    cost_usd: Decimal
    latency_ms: int
    stop_reason: str | None

    @property
    def prompt_tokens(self) -> int:
        """Everything the call sent, cached or not: the prefix its breakpoint covers."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass(frozen=True)
class OffConstraint:
    title: str
    constraint: str  # dates, area, price, kinds or ends


@dataclass(frozen=True)
class Exchange:
    """One user turn and the curator's reply, as run and checked."""

    user: str
    constraints: Constraints
    expect: Expect
    reply: str
    stop_reason: str | None
    iteration_count: int
    uses: tuple[ToolUse, ...]
    calls: tuple[CallUsage, ...]
    seconds: float  # wall clock, tools included
    named: tuple[Named, ...]
    off: tuple[OffConstraint, ...]
    failures: tuple[str, ...]

    @property
    def cost_usd(self) -> Decimal:
        return sum((call.cost_usd for call in self.calls), Decimal(0))

    @property
    def in_order(self) -> bool | None:
        return in_distance_order(self.named)

    @property
    def searches(self) -> list[dict[str, Any]]:
        """The results of the turn's searches that ran."""
        return [
            use.parsed() for use in self.uses if use.name == "search_events" and not use.is_error
        ]


@dataclass(frozen=True)
class ScenarioRun:
    scenario: Scenario
    context: str  # the block the conversation opened with
    exchanges: tuple[Exchange, ...]
    remembered: tuple[str, ...] = ()  # the profile when it ended, "topic: fact"
    verdicts: tuple[Verdict, ...] | None = None  # one per exchange; None when not judged
    judge_error: str = ""
    judge_calls: tuple[CallUsage, ...] = ()
    stopped: str = ""  # what the endpoint said when it refused a call; the run ends here
    waited_seconds: float = 0.0  # spent waiting for the requests-per-minute quota, judge included

    @property
    def judged(self) -> list[tuple[Exchange, Verdict | None]]:
        verdicts: Sequence[Verdict | None] = self.verdicts or [None] * len(self.exchanges)
        return list(zip(self.exchanges, verdicts, strict=True))


# --- the report ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunInfo:
    """What a run was: printed at the head of the report and the transcript."""

    today: date
    model: str
    effort: str
    arm: str
    judge_model: str | None  # None when the judge was not asked
    embed_model: str | None = None  # semantic ranking's model; None: the filters alone
    per_minute: int | None = None  # calls a minute to one model it was paced to; None: unpaced

    @property
    def ranking(self) -> str:
        return f"semantic ranking {self.embed_model}" if self.embed_model else "filters only"

    @property
    def pacing(self) -> str:
        return f"paced to {self.per_minute} calls a minute a model" if self.per_minute else ""


def percentile(values: Sequence[float], share: float) -> float:
    """The nearest-rank percentile: the smallest value at least `share` of `values` are at."""
    ordered = sorted(values)
    return ordered[max(math.ceil(share * len(ordered)) - 1, 0)]


def cache_share(runs: Sequence[ScenarioRun]) -> tuple[int, int, int]:
    """`(calls, read, sent)` over the calls that could read the cache: those after a call of the
    same conversation whose prefix passed CACHE_MIN_TOKENS. Below it nothing is written, so
    nothing can be read (CLAUDE.md § 6)."""
    calls = read = sent = 0
    for run in runs:
        cached = False
        for exchange in run.exchanges:
            for call in exchange.calls:
                if cached:
                    calls += 1
                    read += call.cache_read_tokens
                    sent += call.prompt_tokens
                cached = cached or call.prompt_tokens >= CACHE_MIN_TOKENS
    return calls, read, sent


def _share(hits: int, total: int) -> str:
    return f"{hits / total:.0%} ({hits}/{total})" if total else "n/a (0/0)"


def report(info: RunInfo, runs: Sequence[ScenarioRun]) -> list[str]:
    """What `eval-curation` prints: the totals the P3 done-when reads, then every exchange with
    what it got wrong. Titles, our replies' own passages and the judge's sentences are printed;
    nothing else from a conversation is."""
    lines: list[str] = []

    def block(label: str, *parts: str) -> None:
        lines.append(f"{label:<12} " + "   ".join(part for part in parts if part))

    def detail(label: str, *parts: str) -> None:
        lines.append(f"{'':<12} {label:<9} " + "   ".join(part for part in parts if part))

    exchanges = [exchange for run in runs for exchange in run.exchanges]
    judged = [(e, v) for run in runs for e, v in run.judged if v is not None]
    block(
        "run",
        f"{info.today}",
        f"arm {info.arm}",
        f"{info.model}",
        f"effort {info.effort}",
        info.ranking,
        f"judge {info.judge_model}" if info.judge_model else "not judged",
        info.pacing,
    )
    stopped = [run for run in runs if run.stopped]
    block(
        "scenarios",
        f"{len(runs)} run",
        f"{len(exchanges)} exchanges",
        f"{len(stopped)} stopped" if stopped else "",
    )
    if not exchanges:
        for run in stopped:
            block("stopped", run.scenario.id, run.stopped)
        return lines

    named = [len(e.named) for e in exchanges]
    block(
        "named",
        f"{sum(named)} items",
        f"median {median(named):g} an exchange, max {max(named)}",
    )
    claims = sum(len(v.claims) for _, v in judged)
    if info.judge_model:
        block(
            "unsupported",
            f"{claims} claims in {sum(bool(v.claims) for _, v in judged)} of {len(judged)} "
            "judged exchanges",
        )
    off = Counter(o.constraint for e in exchanges for o in e.off)
    block(
        "off",
        f"{sum(off.values())} named items off constraint",
        ", ".join(f"{n} {constraint}" for constraint, n in sorted(off.items())),
    )
    ordered = [e.in_order for e in exchanges if e.in_order is not None]
    block("order", f"nearest first in {_share(sum(ordered), len(ordered))}")
    made = sum(checks_made(e.expect) for e in exchanges)
    failed = sum(len(e.failures) for e in exchanges)
    block("checks", f"{_share(made - failed, made)} met")
    if info.judge_model:
        tastes = Counter(v.taste for _, v in judged)
        block("taste", ", ".join(f"{tastes[t]} {t}" for t in TASTES))

    rounds = Counter(e.iteration_count for e in exchanges)
    block("rounds", ", ".join(f"{n}: {rounds[n]}" for n in sorted(rounds)), "(tool rounds a turn)")
    searches = [s for e in exchanges for s in e.searches]
    truncated = [s for s in searches if s["truncated"]]
    cut_runs = [
        run for run in runs if any(s["truncated"] for e in run.exchanges for s in e.searches)
    ]
    block(
        "truncated",
        f"{_share(len(truncated), len(searches))} of searches",
        f"in {len(cut_runs)} of {len(runs)} scenarios",
    )
    if info.embed_model:
        queried = [
            use
            for e in exchanges
            for use in e.uses
            if use.name == "search_events" and not use.is_error and use.input.get("query")
        ]
        block("queries", f"{_share(len(queried), len(searches))} of searches passed a query")
    costs = [e.cost_usd for e in exchanges]
    total = sum(costs, Decimal(0))
    block(
        "cost",
        f"${total:.4f}",
        f"${total / len(costs):.4f} an exchange",
        f"median ${median(costs):.4f}, max ${max(costs):.4f}",
        f"{sum(len(e.calls) for e in exchanges)} model calls",
    )
    if info.judge_model:
        judge_calls = [call for run in runs for call in run.judge_calls]
        judge_cost = sum((call.cost_usd for call in judge_calls), Decimal(0))
        block("judge cost", f"${judge_cost:.4f}", f"{len(judge_calls)} model calls")
    calls, read, sent = cache_share(runs)
    block(
        "cache",
        f"reads {_share(read, sent)} of input tokens" if sent else "no call could read the cache",
        f"on {calls} calls after a prefix of {CACHE_MIN_TOKENS}+ tokens",
    )
    seconds = [e.seconds for e in exchanges]
    block(
        "latency",
        f"p50 {percentile(seconds, 0.5):.1f} s",
        f"p95 {percentile(seconds, 0.95):.1f} s",
        f"max {max(seconds):.1f} s an exchange",
        "(waits for the quota left out)" if info.per_minute else "",
    )
    waited = sum(run.waited_seconds for run in runs)
    if waited:
        block("quota", f"waited {waited:.0f} s in all for the requests-per-minute quota")
    stops = Counter(e.stop_reason for e in exchanges if e.stop_reason != "end_turn")
    if stops:
        block("stops", ", ".join(f"{n} {reason}" for reason, n in stops.items()))
    errors = sum(use.is_error for e in exchanges for use in e.uses)
    if errors:
        block("tool errors", f"{errors} tool results were errors")

    for run in runs:
        block("scenario", run.scenario.id)
        for number, (exchange, verdict) in enumerate(run.judged, start=1):
            order = {True: "nearest first", False: "not nearest first", None: ""}[exchange.in_order]
            block(
                f"  {number}",
                f"{exchange.iteration_count} rounds",
                f"{len(exchange.calls)} calls",
                f"${exchange.cost_usd:.4f}",
                f"{exchange.seconds:.1f} s",
                f"{len(exchange.named)} named",
                order,
                f"taste {verdict.taste}" if verdict else "",
            )
            for item in exchange.off:
                detail("off", item.constraint, repr(item.title), f"asked: {exchange.constraints}")
            for failure in exchange.failures:
                detail("check", failure)
            for claim in verdict.claims if verdict else ():
                detail("claim", repr(claim))
        if run.judge_error:
            detail("unjudged", run.judge_error)
        if run.stopped:
            detail("stopped", f"the endpoint refused a call: {run.stopped}")
    return lines


# --- the transcript ------------------------------------------------------------------------------


def transcript(info: RunInfo, runs: Sequence[ScenarioRun]) -> str:
    """Every conversation as Markdown, for the person who judges whether each one is sensible
    (PLAN.md § 8, P3 done-when). Kept outside the repo: it is a run's output, not a label."""
    out = [
        f"# eval-curation {info.today} · arm {info.arm} · {info.model} · effort {info.effort}"
        f" · {info.ranking}",
        "",
        "For each scenario, read the conversation and mark it sensible or not: would a person "
        "who knows the Barcelona art scene be glad of these answers? The P3 bar is four in five.",
        "",
    ]
    for run in runs:
        out += [f"## {run.scenario.id}", "", f"**Persona:** {run.scenario.persona}", ""]
        out += ["```", run.context, "```", ""]
        for number, (exchange, verdict) in enumerate(run.judged, start=1):
            out += [
                f"### {number}. You: {exchange.user}",
                "",
                f"*Asked:* {exchange.constraints}",
                "",
            ]
            for use in exchange.uses:
                out.append(f"- `{use.name} {json.dumps(use.input, ensure_ascii=False)}` → "
                           f"{_result_line(use)}")  # fmt: skip
            out += ["", exchange.reply or f"*(no reply: {exchange.stop_reason})*", ""]
            notes = [
                f"{exchange.iteration_count} rounds, {len(exchange.calls)} calls, "
                f"${exchange.cost_usd:.4f}, {exchange.seconds:.1f} s",
                "named: " + ("; ".join(n.title for n in exchange.named) or "nothing"),
            ]
            notes += [f"off {o.constraint}: {o.title}" for o in exchange.off]
            notes += [f"check: {failure}" for failure in exchange.failures]
            if verdict is not None:
                notes.append(f"judge: taste {verdict.taste}: {verdict.reason}")
                notes += [f"unsupported: {claim}" for claim in verdict.claims]
            out += [f"> {note}  " for note in notes] + [""]
        if run.remembered:
            out += ["**Remembered at the end:**", ""]
            out += [f"- {fact}" for fact in run.remembered] + [""]
        if run.judge_error:
            out += [f"*Unjudged: {run.judge_error}*", ""]
        if run.stopped:
            out += [f"*Stopped: {run.stopped}*", ""]
        out += ["**Sensible?** yes / no", ""]
    return "\n".join(out)


def _result_line(use: ToolUse) -> str:
    """One line on what a tool call got back: the error, or what it found."""
    if use.is_error:
        return f"error: {use.result}"
    result = use.parsed()
    if use.name == "search_events":
        near = f" near {result['near']['name']}" if result.get("near") else ""
        cut = ", truncated" if result["truncated"] else ""
        return f"{len(result['events'])} of {result['total']}{near}{cut}"
    if use.name == "get_events":
        missing = f", not found {result['not_found']}" if result["not_found"] else ""
        return ", ".join(f"{e['id']} {e['title']}" for e in result["events"]) + missing
    return use.result


def progress_line(run: ScenarioRun) -> str:
    """The line `eval-curation` prints as each scenario ends."""
    cost = sum((e.cost_usd for e in run.exchanges), Decimal(0))
    seconds = sum(e.seconds for e in run.exchanges)
    line = f"{run.scenario.id:<32} {len(run.exchanges)} exchanges   ${cost:.4f}   {seconds:.0f} s"
    if run.waited_seconds:
        line += f"   waited {run.waited_seconds:.0f} s for the quota"
    return line + (f"   stopped: {run.stopped}" if run.stopped else "")
