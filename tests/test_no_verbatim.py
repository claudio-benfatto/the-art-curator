"""CLAUDE.md § 1: facts from GRAF, prose from nobody. Do not skip or weaken these tests.

Schema half (P0): no source-fact table may carry a prose column, and third-party page text may
live only in `venue_pages.raw_text`. Checked against the models *and* the migrated database, so a
hand-written migration cannot slip a column past the models.

Repository half (P1): **third-party prose must not enter git either.** A recorded GRAF fixture is
the obvious way for it to happen — 45 of 158 venue profiles carry a `description` of up to 6330
characters, and every event carries `acf.desc_{ca,es,en}`. Fixtures are therefore scrubbed at
capture, and `test_fixtures_carry_no_third_party_prose` is the check that they stayed that way.
This one is not like a failing test elsewhere: once prose is in a pushed commit, removing it is a
history rewrite. That is why the scan landed in the same commit as the first fixture.

The extraction half (no output span > ~25 words copied from its source) lands with `extract`
in P2.
"""

import json
import re
from pathlib import Path

import pytest

from art_curator.db.models import Base
from art_curator.ingest.graf import DROP_KEYS, DROP_PREFIXES, banned_keys, scrub
from tests.db import run_sql

FIXTURES = Path(__file__).parent / "fixtures"

# Every table must be classified here. A new table fails `test_every_table_is_classified` until
# someone decides which rules apply to it — that decision is the point.
SOURCE_TABLES = {"venues", "event_snapshots", "event_occurrences"}  # facts only, ever
PROSE_CACHE = {"venue_pages": {"raw_text"}}  # third-party text, 7-day TTL, purged
OWN_TABLES = {"llm_calls"}  # our data; no third-party text

PROSE = re.compile(r"desc|summary|body|content|excerpt|abstract|prose|blurb|text|html", re.I)
THIRD_PARTY_TEXT = re.compile(r"raw|html|body|excerpt|page_text", re.I)

# Base tables only: PostGIS adds views (geometry_columns, ...) and its own spatial_ref_sys.
DB_COLUMNS = (
    "SELECT c.table_name, c.column_name FROM information_schema.columns c "
    "JOIN information_schema.tables t USING (table_schema, table_name) "
    "WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE' "
    "AND c.table_name NOT IN ('spatial_ref_sys', 'alembic_version')"
)


def _model_columns() -> dict[str, set[str]]:
    return {name: {c.name for c in table.columns} for name, table in Base.metadata.tables.items()}


def _db_columns(url) -> dict[str, set[str]]:
    columns: dict[str, set[str]] = {}
    for table, column in run_sql(url, DB_COLUMNS)[0]:
        columns.setdefault(table, set()).add(column)
    return columns


def _violations(columns: dict[str, set[str]]) -> list[str]:
    found = []
    for table, names in columns.items():
        for name in names:
            if table in SOURCE_TABLES and PROSE.search(name):
                found.append(f"{table}.{name}: prose-like column on a source-fact table")
            elif (
                table not in SOURCE_TABLES
                and THIRD_PARTY_TEXT.search(name)
                and name not in PROSE_CACHE.get(table, set())
            ):
                found.append(f"{table}.{name}: third-party text outside venue_pages.raw_text")
    return found


def test_every_table_is_classified():
    classified = SOURCE_TABLES | set(PROSE_CACHE) | OWN_TABLES
    assert set(Base.metadata.tables) == classified


def test_models_carry_no_third_party_prose():
    assert not _violations(_model_columns())


def test_database_carries_no_third_party_prose(migrated_db):
    db = _db_columns(migrated_db)
    assert set(db) == set(Base.metadata.tables), "database has tables the models don't declare"
    assert not _violations(db)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("venues", "description"),
        ("event_snapshots", "desc_en"),
        ("event_snapshots", "summary"),
        ("venues", "body_html"),
        ("llm_calls", "response_body"),
        ("venue_pages", "raw_html"),
    ],
)
def test_planted_column_is_caught(table, column):
    assert _violations({table: {column}})


# --- Recorded fixtures: prose must not enter the repository ---------------------------------------


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES.rglob("*.json"))


def test_there_are_fixtures_to_scan():
    """A scan matching nothing would pass forever. Fail instead if the fixtures ever move."""
    assert _fixture_files(), f"no JSON fixtures under {FIXTURES}"


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda p: p.name)
def test_fixtures_carry_no_third_party_prose(path: Path):
    found = banned_keys(json.loads(path.read_text()))
    assert not found, (
        f"{path.relative_to(FIXTURES.parent)} carries {sorted(found)}. Re-record it with "
        f"`--record`, which scrubs; do not hand-edit, and do not push this."
    )


def test_fixtures_carry_no_long_third_party_text():
    """A second, key-blind pass: the length of the longest string in every fixture.

    `banned_keys` only catches prose in a key we already know about. Anything over ~400 characters
    in a facts-only payload is prose under some name we have not seen — a name, address, title or
    URL is nowhere near that long (the longest in the current fixtures is a 102-character address).
    """

    def longest(payload, path=()):
        if isinstance(payload, str):
            return [(len(payload), ".".join(path), payload[:80])]
        if isinstance(payload, dict):
            return [x for k, v in payload.items() for x in longest(v, (*path, str(k)))]
        if isinstance(payload, list):
            return [x for i, v in enumerate(payload) for x in longest(v, (*path, str(i)))]
        return []

    for path in _fixture_files():
        strings = longest(json.loads(path.read_text()))
        worst = max(strings, default=(0, "", ""))
        assert worst[0] <= 400, f"{path.name}: {worst[1]} holds {worst[0]} chars: {worst[2]!r}"


@pytest.mark.parametrize("key", sorted(DROP_KEYS))
def test_scrub_removes_planted_prose(key):
    """Like `test_planted_column_is_caught`, for the capture path rather than the schema."""
    planted = {"id": 1, "name": "keep", key: "third-party prose", "nested": [{key: "prose"}]}
    clean = scrub(planted)
    assert banned_keys(planted) == {key}
    assert banned_keys(clean) == set()
    assert clean == {"id": 1, "name": "keep", "nested": [{}]}


@pytest.mark.parametrize("prefix", DROP_PREFIXES)
def test_scrub_removes_planted_prose_by_prefix(prefix):
    for lang in ("ca", "es", "en"):
        planted = {"acf": {f"{prefix}{lang}": "prosa", "title_ca": "keep"}}
        assert scrub(planted) == {"acf": {"title_ca": "keep"}}
