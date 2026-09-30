"""GRAF payloads in, typed facts out — and third-party prose dropped on the way in.

`scrub()` runs on every response body **before** validation, so prose never reaches a model, a
log line, a span or disk (CLAUDE.md § 1). It is not a formality: 45 of 158 venue profiles carry a
`description` of up to 6330 characters, and every event carries `acf.desc_{ca,es,en}` of up to
~2.3k. The models below simply have nowhere to put any of it.

Two GRAF entities, neither sufficient alone, which is why P1 joins them:

- a **term** (`event-venues`) is a *space* — address, city, lat/long, and no URL;
- a **profile** (`users`) is an *organisation* — a URL, and nothing geographic.

The join is many-to-one: MACBA is one profile and four terms. See `ingest/matching.py`.

Field-level traps, each with a test in `tests/test_graf_parse.py`:

- **`longtitude` is misspelled in their API.** A correctly spelled `longitude` yields no geometry.
- **14 of 570 terms sit at `0.000000 / 0.000000`.** As a POINT those land in the Gulf of Guinea and
  silently corrupt the distance math of CLAUDE.md § 8, so they become no geometry at all.
- Latitude and longitude are **strings**, and `Point.ewkt` is the single place the lon/lat order of
  WKT is applied — the one spot a swap could hide.
- **`_event_price-*` arrive as `""`**, never absent, and are `""` for every event in the live
  window. `register_deadline` arrives as `None`.
- **The list row's `occurrence_id` is a string** (`"23161"`); `/events/{id}/occurrences` returns an
  **int**. The single-event endpoint omits it entirely *and* serves `start`/`end` without a UTC
  offset, so it is unusable for dates — prefer the list row or `/occurrences`.
- **`modified_gmt` is naive UTC; `modified` is site-local.** Use the former and attach UTC.
- Event `date` is absent (CLAUDE.md records it as null — either way, use `start`/`end`).
- **WordPress serves HTML entities in the fields we persist**: `Col·lectiva d&#8217;estiu` as an
  event title, `L&amp;B Gallery` as both a term and a profile name. Titles are dedup identifiers
  (CLAUDE.md § 1), so an escaping change upstream must not mint a second identity for one event —
  every persisted string is unescaped here, once.
"""

import html
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Third-party prose, dropped before validation. Wider than `desc_*` on purpose: `description` is
# on both users and venue terms, and events carry `content` and `excerpt`. `guid` and the yoast
# keys are not prose but are pure noise that would otherwise end up in a recorded fixture.
DROP_KEYS = frozenset(
    {"description", "content", "excerpt", "guid", "yoast_head", "yoast_head_json"}
)
DROP_PREFIXES = ("desc_",)

# A term at exactly the null island is a missing coordinate, not a location in the Gulf of Guinea.
NULL_ISLAND = (0.0, 0.0)


def scrub[T](payload: T) -> T:
    """Recursively drop third-party prose before anything can persist or log it (CLAUDE.md § 1).

    Applied to the raw body, so a key we have never seen cannot smuggle prose past the models by
    being ignored: it is dropped by name, whatever its position in the tree.
    """
    if isinstance(payload, dict):
        return {  # type: ignore[return-value]
            k: scrub(v)
            for k, v in payload.items()
            if k not in DROP_KEYS and not k.startswith(DROP_PREFIXES)
        }
    if isinstance(payload, list):
        return [scrub(v) for v in payload]  # type: ignore[return-value]
    return payload


def banned_keys(payload: Any) -> set[str]:
    """Every prose key present anywhere in `payload`. Empty iff `scrub()` has nothing left to do.

    Used by `--record` before writing a fixture and by `tests/test_no_verbatim.py` after, because
    prose committed to git is permanent — see that module's docstring.
    """
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in DROP_KEYS or key.startswith(DROP_PREFIXES):
                found.add(key)
            found |= banned_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            found |= banned_keys(value)
    return found


class Point(NamedTuple):
    """A venue coordinate. Exists so the lon/lat order of WKT lives in exactly one place."""

    lat: float
    lon: float

    @property
    def ewkt(self) -> str:
        """`SRID=4326;POINT(lon lat)` — WKT is x y, i.e. longitude first."""
        return f"SRID=4326;POINT({self.lon} {self.lat})"


def clean_text(value: Any) -> str | None:
    """Unescape HTML entities, strip, blank to absent. `None` for anything that is not a string."""
    if not isinstance(value, str):
        return None
    cleaned = html.unescape(value).strip()
    return cleaned or None


def parse_point(latitude: Any, longitude: Any) -> Point | None:
    """A `Point`, or `None` when either coordinate is blank, unparseable or at the null island."""
    try:
        lat = float(str(latitude).strip())
        lon = float(str(longitude).strip())
    except (TypeError, ValueError):
        return None
    if (lat, lon) == NULL_ISLAND:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return Point(lat, lon)


def parse_decimal(value: Any) -> Decimal | None:
    """A price. `""` (what GRAF actually sends) and unparseable values are absent, not zero."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_int(value: Any) -> int | None:
    """An id. Accepts the string form GRAF uses for `occurrence_id` in list rows."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def parse_datetime(value: Any, *, assume_utc: bool = False) -> datetime | None:
    """An ISO 8601 timestamp, always returned aware.

    `assume_utc` is for `modified_gmt`, which is UTC but written without an offset. A naive value
    without that flag is rejected rather than guessed at — the single-event endpoint serves
    site-local naive times, and reading those as UTC would shift every date by an hour or two.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC) if assume_utc else None
    return parsed.astimezone(UTC)


class _GrafModel(BaseModel):
    # `extra="ignore"`: GRAF adds fields (`class_list`, `abe_event_venue`, `country`) that we have
    # no column for, and a new one upstream must not fail the sync.
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


class Term(_GrafModel):
    """An `event-venues` taxonomy term: a *space*. Carries geography, never a URL."""

    source_venue_id: int = Field(alias="id")
    name: str
    slug: str
    address: str | None = None
    city: str | None = None
    province: str | None = Field(default=None, alias="state")  # GRAF calls it `state`
    postcode: str | None = None
    point: Point | None = None
    event_count: int = Field(default=0, alias="count")

    @field_validator("name", "slug", mode="before")
    @classmethod
    def _required_text(cls, v: Any) -> Any:
        return clean_text(v) or v

    @field_validator("address", "city", "province", "postcode", mode="before")
    @classmethod
    def _optional_text(cls, v: Any) -> str | None:
        return clean_text(v)

    @model_validator(mode="before")
    @classmethod
    def _geometry(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "point" in data:
            return data
        # `longtitude` (sic) is the real field name; a correctly spelled `longitude` is ignored.
        return {**data, "point": parse_point(data.get("latitude"), data.get("longtitude"))}


class Profile(_GrafModel):
    """A `users` entry: an *organisation*. Carries a URL, never geography."""

    source_profile_id: int = Field(alias="id")
    name: str
    slug: str
    url: str | None = None

    @field_validator("name", "slug", mode="before")
    @classmethod
    def _required_text(cls, v: Any) -> Any:
        return clean_text(v) or v

    @field_validator("url", mode="before")
    @classmethod
    def _optional_text(cls, v: Any) -> str | None:
        return clean_text(v)


class Occurrence(_GrafModel):
    """When an event happens. Both shapes GRAF serves parse into this: the list row (string
    `occurrence_id`, offset-aware `start`) and `/events/{id}/occurrences` (int, also aware)."""

    source_occurrence_id: int = Field(alias="occurrence_id")
    starts_at: datetime = Field(alias="start")
    ends_at: datetime | None = Field(default=None, alias="end")

    @field_validator("source_occurrence_id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> Any:
        return parse_int(v)

    @field_validator("starts_at", "ends_at", mode="before")
    @classmethod
    def _timestamp(cls, v: Any) -> datetime | None:
        return parse_datetime(v)


class Event(_GrafModel):
    """One row of `/events`. That endpoint is per *occurrence*, so a post id can repeat across
    rows and the sync dedupes on `source_event_id` before writing `event_snapshots`."""

    source_event_id: int = Field(alias="id")
    title: str
    title_en: str | None = None
    source_venue_ids: tuple[int, ...] = Field(default=(), alias="event-venues")
    source_category_id: int | None = None
    is_free: bool | None = None
    price_min: Decimal | None = None
    price_max: Decimal | None = None
    is_online: bool | None = None
    source_url: str = Field(alias="link")
    web_url_ca: str | None = None
    web_url_es: str | None = None
    web_url_en: str | None = None
    source_modified_at: datetime | None = None
    occurrence: Occurrence | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        """Lift `title.rendered` and the `acf` block into flat fields."""
        if not isinstance(data, dict) or "acf" not in data:
            return data
        # `acf` is a dict on events but an empty *list* on terms and profiles.
        acf = data["acf"] if isinstance(data.get("acf"), dict) else {}
        title = data.get("title")
        rendered = title.get("rendered") if isinstance(title, dict) else title
        occurrence = data.get("occurrence")
        if occurrence is None and data.get("occurrence_id") is not None:
            occurrence = {k: data.get(k) for k in ("occurrence_id", "start", "end")}
        return {
            **data,
            "title": clean_text(rendered),
            "title_en": clean_text(acf.get("title_en")),
            "source_category_id": parse_int(acf.get("event_category")),
            "is_free": acf.get("_event_free") if isinstance(acf.get("_event_free"), bool) else None,
            "price_min": parse_decimal(acf.get("_event_price-min")),
            "price_max": parse_decimal(acf.get("_event_price-max")),
            "is_online": (
                acf.get("is_online_event") if isinstance(acf.get("is_online_event"), bool) else None
            ),
            "web_url_ca": clean_text(acf.get("_event_web_ca")),
            "web_url_es": clean_text(acf.get("_event_web_es")),
            "web_url_en": clean_text(acf.get("_event_web_en")),
            # `modified_gmt` is UTC written without an offset; `modified` is site-local.
            "source_modified_at": parse_datetime(data.get("modified_gmt"), assume_utc=True),
            "occurrence": occurrence,
        }

    @field_validator("source_venue_ids", mode="before")
    @classmethod
    def _venue_ids(cls, v: Any) -> tuple[int, ...]:
        if not isinstance(v, list):
            return ()
        return tuple(i for i in (parse_int(x) for x in v) if i is not None)

    @property
    def source_venue_id(self) -> int | None:
        """The one term an event names. Every event in the live window names exactly one."""
        return self.source_venue_ids[0] if self.source_venue_ids else None


def parse_terms(rows: Iterable[Any]) -> list[Term]:
    return [Term.model_validate(row) for row in rows]


def parse_profiles(rows: Iterable[Any]) -> list[Profile]:
    return [Profile.model_validate(row) for row in rows]


def parse_events(rows: Iterable[Any]) -> list[Event]:
    return [Event.model_validate(row) for row in rows]


def parse_occurrences(rows: Iterable[Any]) -> list[Occurrence]:
    return [Occurrence.model_validate(row) for row in rows]
