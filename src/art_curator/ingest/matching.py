"""Join venue terms to venue profiles, and pick out the pilot venues. Pure functions, no I/O.

GRAF splits a venue across two entities and neither is sufficient alone (see `ingest/graf.py`):
a **term** has geography and no URL, a **profile** has a URL and no geography. P2 crawls
`venues.website_url`, so without this join there is nothing to crawl.

The join is **many-to-one**: MACBA is one profile and four terms (`MACBA`, `MACBA, Museu d'Art
Contemporani de Barcelona` with 67 events, `Convent dels Àngels - MACBA`, `Capella MACBA`).
That is why migration 0002 drops `UNIQUE (source, source_profile_id)`.

Slugs differ between the two sides because they are derived from different things: a term slug is
sanitized from the term *name* at creation, with a numeric suffix when the slug is already taken
(`chiquita-room-2`), while a profile slug is WordPress's `user_nicename`, derived from the *login*
(`chiquitaroom`). Hence a cascade rather than a single key.

**Ruled out with data, so nobody retries it:** `event.author` is not a venue↔profile join. Over one
live window, 10 of 38 distinct (term, author) pairs disagreed with the name/slug join, because
`Barcelona Gallery Weekend` authors events at six different galleries.

Rounds run per term, first hit wins, and a round that finds more than one candidate yields nothing
at all — an ambiguous term is reported, never guessed at:

1. exact `slug`
2. `base_slug` — the numeric-suffix case
3. `normalize_name`
4. **containment** — the profile's name appears as a whole-word token run inside the term's

Containment is the round that makes the many-to-one decision pay: it is how MACBA's 67-event term,
both Alzueta spaces, `Espai 13 - Fundació Joan Miró`, both Nogueras Blanchard spaces, `FUGA Gallery`
and `Galeria Zielinsky` get a website at all. It is also the only heuristic here, so it is guarded
three ways (minimum length, whole-word runs, exactly one candidate) and the `sync-graf` dry-run
report lists its joins as their own block, so a human reviews those lines rather than all of them.
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple
from urllib.parse import urlsplit

from art_curator.ingest.graf import Profile, Term, clean_text

# PLAN.md § 7. `tests/test_matching.py` asserts this equals the names parsed out of that prose, so
# the two cannot drift — the same trick `test_config.py` uses against `variables.tf`.
PILOT_VENUES: tuple[str, ...] = (
    "MACBA",
    "Fundació Joan Miró",
    "CaixaForum Barcelona",
    "La Escocesa",
    "ESPRONCEDA",
    "àngels barcelona",
    "ADN Galeria",
    "Chiquita Room",
    "ProjecteSD",
    "Galeria Marc Domènech",
    "RocioSantaCruz",
    "Pigment Gallery",
    "FUGA Gallery",
    "Dilalica",
    "ethall",
    "Sala Parés",
    "House of Chappaz",
    "Galería Alegría",
    "#plantauno",
    "ACVic",
)

# A venue whose only web presence is Instagram stays facts-only: there is no page to extract an
# exhibition from, and crawling it would breach their terms. Only 2 of 158 profiles, not the ~12
# PLAN.md § 9 estimated.
INSTAGRAM_HOSTS = frozenset({"instagram.com", "instagr.am"})

# A profile name shorter than this matches too much as a substring ("ethall" inside a sentence is
# fine; "art" inside "Museu d'Art" is not).
MIN_CONTAINMENT_LENGTH = 4

ROUNDS = ("slug", "base-slug", "name", "contains")

_NUMERIC_SUFFIX = re.compile(r"-\d+$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    """A comparable form of a display name.

    NFKD then drop combining marks, so `àngels barcelona` and `angels barcelona` agree; `&` becomes
    ` and ` rather than vanishing, so `L&B Gallery` does not collapse to `lb gallery` and collide
    with something else; everything else non-alphanumeric becomes a single space.

    HTML entities are unescaped first: GRAF serves `L&amp;B Gallery` as both a term name and a
    profile name, and comparing one escaped form against another is how a join silently misses.
    """
    text = clean_text(name) or ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    return _NON_ALNUM.sub(" ", text).strip()


def base_slug(slug: str) -> str:
    """`chiquita-room-2` -> `chiquita-room`. WordPress appends a counter when a slug is taken."""
    return _NUMERIC_SUFFIX.sub("", (clean_text(slug) or "").lower())


def classify_url(url: str | None) -> tuple[str | None, str | None, bool]:
    """`(website_url, instagram_url, crawl_enabled)`.

    `crawl_enabled` is a *permission*, not a promise: P2 still honours `robots.txt` and the
    ≤8-pages-per-venue cap before fetching anything.
    """
    cleaned = clean_text(url)
    if not cleaned:
        return None, None, False
    parts = urlsplit(cleaned)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None, None, False
    host = parts.netloc.lower().split("@")[-1].split(":")[0].removeprefix("www.")
    if any(host == known or host.endswith(f".{known}") for known in INSTAGRAM_HOSTS):
        return None, cleaned, False
    return cleaned, None, True


class Match(NamedTuple):
    source_venue_id: int
    source_profile_id: int
    round: str  # which of ROUNDS fired; the dry-run report groups by it


@dataclass(frozen=True)
class MatchResult:
    matches: dict[int, Match] = field(default_factory=dict)  # term id -> match
    ambiguous: dict[int, tuple[int, ...]] = field(default_factory=dict)  # term id -> candidates

    @property
    def counts(self) -> dict[str, int]:
        """Terms joined, per round. The dry-run report prints this, and a test asserts it."""
        return {r: sum(1 for m in self.matches.values() if m.round == r) for r in ROUNDS}

    def profile_id(self, source_venue_id: int) -> int | None:
        match = self.matches.get(source_venue_id)
        return match.source_profile_id if match else None


def _unique(candidates: Sequence[Profile]) -> Profile | None:
    """Exactly one candidate, or nothing. Two profiles claiming one term is not a 50/50 guess."""
    return candidates[0] if len(candidates) == 1 else None


def _index(profiles: Iterable[Profile], key) -> dict[str, list[Profile]]:
    index: dict[str, list[Profile]] = {}
    for profile in profiles:
        k = key(profile)
        if k:
            index.setdefault(k, []).append(profile)
    return index


def _contains_run(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Is `needle` a contiguous run of whole tokens in `haystack`?

    Token-wise, not substring: `Espai 13 - Fundació Joan Miró` contains `Fundació Joan Miró`, but
    the term `Sismògraf` must not match the profile `GRAF` — a real pair in the live corpus.
    """
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        list(haystack[i : i + len(needle)]) == list(needle)
        for i in range(len(haystack) - len(needle) + 1)
    )


def match_profiles(terms: Iterable[Term], profiles: Iterable[Profile]) -> MatchResult:
    """Join each term to at most one profile. Several terms may share one — that is the point."""
    profiles = list(profiles)
    by_slug = _index(profiles, lambda p: (clean_text(p.slug) or "").lower())
    by_base_slug = _index(profiles, lambda p: base_slug(p.slug))
    by_name = _index(profiles, lambda p: normalize_name(p.name))

    # Candidates for containment, longest name first so the report reads sensibly. The
    # exactly-one-candidate guard means order cannot change the outcome.
    contained = sorted(
        (
            (normalize_name(p.name).split(), p)
            for p in profiles
            if len(normalize_name(p.name)) >= MIN_CONTAINMENT_LENGTH
        ),
        key=lambda pair: -len(pair[0]),
    )

    matches: dict[int, Match] = {}
    ambiguous: dict[int, tuple[int, ...]] = {}

    for term in terms:
        rounds = (
            ("slug", by_slug.get((clean_text(term.slug) or "").lower(), [])),
            ("base-slug", by_base_slug.get(base_slug(term.slug), [])),
            ("name", by_name.get(normalize_name(term.name), [])),
        )
        for name, candidates in rounds:
            if not candidates:
                continue
            winner = _unique(candidates)
            if winner is None:
                ambiguous[term.source_venue_id] = tuple(
                    sorted(p.source_profile_id for p in candidates)
                )
            else:
                matches[term.source_venue_id] = Match(
                    term.source_venue_id, winner.source_profile_id, name
                )
            break  # first round that has anything to say decides, including "ambiguous"
        else:
            tokens = normalize_name(term.name).split()
            hits = [p for needle, p in contained if _contains_run(tokens, needle)]
            # Several organisations' names inside one term's is not a coin flip.
            winner = _unique(hits)
            if winner is not None:
                matches[term.source_venue_id] = Match(
                    term.source_venue_id, winner.source_profile_id, "contains"
                )
            elif hits:
                ambiguous[term.source_venue_id] = tuple(sorted(p.source_profile_id for p in hits))

    return MatchResult(matches=matches, ambiguous=ambiguous)


def match_pilots(
    terms: Iterable[Term], names: Sequence[str] = PILOT_VENUES
) -> tuple[dict[str, int], tuple[str, ...]]:
    """`({pilot name: term id}, unmatched names)`, matched on `normalize_name`, one hit only.

    Slug matching would fail here: `Chiquita Room`'s term slug is `chiquita-room-2` and
    `#plantauno`'s is `plantauno`. `is_pilot` is what scopes the P2 crawl to 20 venues, so a miss
    matters enough that `sync-graf` exits non-zero over it.
    """
    by_name: dict[str, list[Term]] = {}
    for term in terms:
        by_name.setdefault(normalize_name(term.name), []).append(term)

    matched: dict[str, int] = {}
    unmatched: list[str] = []
    for name in names:
        candidates = by_name.get(normalize_name(name), [])
        if len(candidates) == 1:
            matched[name] = candidates[0].source_venue_id
        else:
            unmatched.append(name)
    return matched, tuple(unmatched)
