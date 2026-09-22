"""CLAUDE.md § 1: facts from GRAF, prose from nobody. Do not skip or weaken these tests.

Schema half (P0): no GRAF-sourced table may carry a prose column, and third-party page text may
live only in `venue_pages.raw_text`. Checked against the models *and* the migrated database, so a
hand-written migration cannot slip a column past the models.

The extraction half (no output span > ~25 words copied from its source) lands with `extract`
in P2.
"""

import re

import pytest

from art_curator.db.models import Base
from tests.db import run_sql

# Every table must be classified here. A new table fails `test_every_table_is_classified` until
# someone decides which rules apply to it — that decision is the point.
GRAF_TABLES = {"venues", "graf_event_snapshots", "graf_event_occurrences"}  # facts only, ever
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
            if table in GRAF_TABLES and PROSE.search(name):
                found.append(f"{table}.{name}: prose-like column on a GRAF-sourced table")
            elif (
                table not in GRAF_TABLES
                and THIRD_PARTY_TEXT.search(name)
                and name not in PROSE_CACHE.get(table, set())
            ):
                found.append(f"{table}.{name}: third-party text outside venue_pages.raw_text")
    return found


def test_every_table_is_classified():
    classified = GRAF_TABLES | set(PROSE_CACHE) | OWN_TABLES
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
        ("graf_event_snapshots", "desc_en"),
        ("graf_event_snapshots", "summary"),
        ("venues", "body_html"),
        ("llm_calls", "response_body"),
        ("venue_pages", "raw_html"),
    ],
)
def test_planted_column_is_caught(table, column):
    assert _violations({table: {column}})
