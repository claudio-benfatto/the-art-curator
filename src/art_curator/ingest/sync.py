"""Write one GRAF fetch to Postgres: venues, event snapshots, occurrences. Facts only.

Two steps, split so the dry-run report and the real write cannot disagree:

1. `plan_venues()` — pure. Terms + the profile join + the pilot list -> the desired venue rows.
2. `write_graf()` — applies a fetch inside the caller's transaction and **never commits**. The
   caller commits, or rolls back for `sync-graf --dry-run`: the dry run is the real write undone,
   so every count it prints comes from the same statements a real run executes.

Rules the statements encode:

- **The profile join and `is_pilot` are derived and recomputed from scratch every run**, never
  accumulated. A term that drops out of the fetch is unjoined (no URL, `crawl_enabled=false`), so P2
  stops crawling it; its row and facts stay. Because the join is recomputed, a GRAF editor renaming
  a term can silently move a URL between venues — `joins_before` / `joins_after` are how the report
  surfaces that.
- **A venue upsert only touches a row whose values changed** (`IS DISTINCT FROM`), so
  `venues.updated_at` means "last changed", not "last synced", and the changed-row count is real.
- **Nothing is ever deleted.** An event that leaves the live window is simply not touched, so its
  `last_seen_at` freezes — that is how history survives an API that only serves the present.
- **`now()` is fixed at transaction start**, so every row a run touches gets the same
  `last_seen_at`, and "seen in this run" is an exact equality, not a time window.

All writes are Core statements on the tables, not ORM objects: a bulk `ON CONFLICT` upsert of ~570
rows is one round trip (~8k bind parameters, well under asyncpg's 32,767).
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.db.models import EventOccurrence, EventSnapshot, Venue
from art_curator.ingest.graf import Event, Profile, Term
from art_curator.ingest.matching import MatchResult, classify_url

SOURCE = "graf"

VENUES = Venue.__table__
EVENTS = EventSnapshot.__table__
OCCURRENCES = EventOccurrence.__table__


@dataclass(frozen=True)
class GrafSnapshot:
    """Everything one fetch saw, parsed. Fetching is `sync-graf`'s job, not this module's."""

    terms: Sequence[Term]
    profiles: Sequence[Profile]
    events: Sequence[Event]


class VenueRow(NamedTuple):
    """The desired state of one `venues` row. Field names are column names."""

    source_venue_id: int
    name: str
    slug: str
    address: str | None
    city: str | None
    province: str | None
    postcode: str | None
    geom: str | None  # EWKT, from `Point.ewkt`; None for a missing or null-island coordinate
    source_profile_id: int | None
    website_url: str | None
    instagram_url: str | None
    crawl_enabled: bool
    is_pilot: bool


# Every venue column `plan_venues` decides, i.e. what an upsert may overwrite.
_VENUE_UPDATE_COLUMNS = tuple(c for c in VenueRow._fields if c != "source_venue_id")

# Reset on a venue the fetch no longer returns.
_UNJOINED = dict(
    source_profile_id=None,
    website_url=None,
    instagram_url=None,
    crawl_enabled=False,
    is_pilot=False,
)


@dataclass(frozen=True)
class SyncResult:
    venues: int  # terms in this fetch
    venues_inserted: int
    venues_updated: int  # existing rows whose values changed
    venues_unjoined: int  # rows no longer in the fetch that lost their join or pilot flag
    joins_before: dict[int, int]  # term id -> profile id, as the database held it
    joins_after: dict[int, int]
    events: int  # distinct posts
    occurrences: int
    unknown_venue_events: tuple[int, ...] = field(default=())  # post ids naming an unknown term


def plan_venues(
    snapshot: GrafSnapshot, match: MatchResult, pilot_ids: Collection[int]
) -> list[VenueRow]:
    """One row per distinct term. Later duplicates win — pagination over a list that changes
    mid-fetch can repeat a term, and one `ON CONFLICT` statement cannot touch a row twice."""
    profiles = {p.source_profile_id: p for p in snapshot.profiles}
    rows: dict[int, VenueRow] = {}
    for term in snapshot.terms:
        profile_id = match.profile_id(term.source_venue_id)
        profile = profiles.get(profile_id) if profile_id is not None else None
        website_url, instagram_url, crawl_enabled = classify_url(profile.url if profile else None)
        rows[term.source_venue_id] = VenueRow(
            source_venue_id=term.source_venue_id,
            name=term.name,
            slug=term.slug,
            address=term.address,
            city=term.city,
            province=term.province,
            postcode=term.postcode,
            geom=term.point.ewkt if term.point else None,
            source_profile_id=profile.source_profile_id if profile else None,
            website_url=website_url,
            instagram_url=instagram_url,
            crawl_enabled=crawl_enabled,
            is_pilot=term.source_venue_id in pilot_ids,
        )
    return list(rows.values())


async def write_graf(
    session: AsyncSession, snapshot: GrafSnapshot, venues: Sequence[VenueRow]
) -> SyncResult:
    """Apply one fetch. Runs in the caller's transaction and never commits."""
    if not venues:
        # An empty fetch is a failed fetch, not an empty GRAF — and it would unjoin every venue.
        raise ValueError("refusing to sync zero venues")

    existing = dict(
        (
            await session.execute(
                select(VENUES.c.source_venue_id, VENUES.c.source_profile_id).where(
                    VENUES.c.source == SOURCE
                )
            )
        ).all()
    )
    fetched = [v.source_venue_id for v in venues]

    unjoined = await session.execute(
        update(VENUES)
        .where(
            VENUES.c.source == SOURCE,
            VENUES.c.source_venue_id.not_in(fetched),
            or_(
                VENUES.c.source_profile_id.is_not(None),
                VENUES.c.website_url.is_not(None),
                VENUES.c.instagram_url.is_not(None),
                VENUES.c.crawl_enabled,
                VENUES.c.is_pilot,
            ),
        )
        .values(**_UNJOINED, updated_at=func.now())
    )

    stmt = insert(VENUES).values([{"source": SOURCE, **v._asdict()} for v in venues])
    changed = or_(
        *(
            VENUES.c[c].is_distinct_from(stmt.excluded[c])
            for c in _VENUE_UPDATE_COLUMNS
            if c != "geom"
        ),
        # Compare the serialized point, not the geography: `=` on geography may compare bounding
        # boxes only, which would miss a small move.
        func.ST_AsBinary(VENUES.c.geom).is_distinct_from(func.ST_AsBinary(stmt.excluded.geom)),
    )
    upserted = await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[VENUES.c.source, VENUES.c.source_venue_id],
            set_={c: stmt.excluded[c] for c in _VENUE_UPDATE_COLUMNS} | {"updated_at": func.now()},
            where=changed,
        ).returning(VENUES.c.source_venue_id)
    )
    touched = upserted.scalars().all()
    inserted = sum(1 for term_id in touched if term_id not in existing)

    # Read back rather than use RETURNING: a guarded upsert returns nothing for an unchanged row,
    # and an event may name a venue that only an earlier run saw.
    venue_ids = dict(
        (
            await session.execute(
                select(VENUES.c.source_venue_id, VENUES.c.id).where(VENUES.c.source == SOURCE)
            )
        ).all()
    )

    events, unknown = _event_rows(snapshot.events, venue_ids)
    event_ids = await _upsert_events(session, events)
    occurrences = _occurrence_rows(snapshot.events, event_ids)
    await _upsert_occurrences(session, occurrences)

    return SyncResult(
        venues=len(venues),
        venues_inserted=inserted,
        venues_updated=len(touched) - inserted,
        venues_unjoined=unjoined.rowcount,
        joins_before={t: p for t, p in existing.items() if p is not None},
        joins_after={
            v.source_venue_id: v.source_profile_id
            for v in venues
            if v.source_profile_id is not None
        },
        events=len(events),
        occurrences=len(occurrences),
        unknown_venue_events=unknown,
    )


def _event_rows(
    events: Sequence[Event], venue_ids: dict[int, int]
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    """One row per post: `/events` is per *occurrence*, so a post id can repeat."""
    rows: dict[int, dict[str, Any]] = {}
    unknown: set[int] = set()
    for event in events:
        term_id = event.source_venue_id
        if term_id is not None and term_id not in venue_ids:
            unknown.add(event.source_event_id)
        rows[event.source_event_id] = {
            "source": SOURCE,
            "source_event_id": event.source_event_id,
            "venue_id": venue_ids.get(term_id) if term_id is not None else None,
            "title": event.title,
            "title_en": event.title_en,
            "source_category_id": event.source_category_id,
            "is_free": event.is_free,
            "price_min": event.price_min,
            "price_max": event.price_max,
            "is_online": event.is_online,
            "source_url": event.source_url,
            "web_url_ca": event.web_url_ca,
            "web_url_es": event.web_url_es,
            "web_url_en": event.web_url_en,
            "source_modified_at": event.source_modified_at,
        }
    return list(rows.values()), tuple(sorted(unknown))


async def _upsert_events(session: AsyncSession, rows: list[dict[str, Any]]) -> dict[int, int]:
    """`{post id: event_snapshots.id}`. `first_seen_at` is never in the SET list, so it is
    preserved by construction; `last_seen_at` moves on every run that sees the post."""
    if not rows:  # the live window can legitimately be empty
        return {}
    stmt = insert(EVENTS).values(rows)
    keys = {"source", "source_event_id"}
    result = await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[EVENTS.c.source, EVENTS.c.source_event_id],
            set_={c: stmt.excluded[c] for c in rows[0] if c not in keys}
            | {"last_seen_at": func.now()},
        ).returning(EVENTS.c.source_event_id, EVENTS.c.id)
    )
    return dict(result.tuples().all())


def _occurrence_rows(events: Sequence[Event], event_ids: dict[int, int]) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for event in events:
        occurrence = event.occurrence
        if occurrence is None:
            continue
        rows[occurrence.source_occurrence_id] = {
            "source": SOURCE,
            "event_id": event_ids[event.source_event_id],
            "source_occurrence_id": occurrence.source_occurrence_id,
            "starts_at": occurrence.starts_at,
            "ends_at": occurrence.ends_at,
        }
    return list(rows.values())


async def _upsert_occurrences(session: AsyncSession, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = insert(OCCURRENCES).values(rows)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[OCCURRENCES.c.source, OCCURRENCES.c.source_occurrence_id],
            set_={
                "event_id": stmt.excluded.event_id,
                "starts_at": stmt.excluded.starts_at,
                "ends_at": stmt.excluded.ends_at,
                "last_seen_at": func.now(),
            },
        )
    )
