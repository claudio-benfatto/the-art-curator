"""Database fixtures. Tests never touch the database DATABASE_URL names: they create throwaway
databases on the same server and drop them afterwards.

No reachable server → DB tests skip, so `uv run pytest` works without Compose. CI sets
REQUIRE_DB=1, which turns that skip into a failure — the schema must never go untested there.
"""

import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from alembic import command
from sqlalchemy.engine import URL, make_url

from art_curator.config import Settings
from tests.db import alembic_config, run_sql


@pytest.fixture(scope="session")
def server_url() -> URL:
    url = make_url(Settings().database_url)
    try:
        run_sql(url, "SELECT 1")
    except Exception as exc:  # any connection failure means "no database here"
        if os.environ.get("REQUIRE_DB") == "1":
            pytest.fail(f"REQUIRE_DB=1 but the database is unreachable: {exc}")
        pytest.skip(f"database unreachable ({type(exc).__name__}); start it with Compose")
    return url


@pytest.fixture(scope="session")
def fresh_database(server_url: URL) -> Callable:
    """Context manager yielding the URL of a new, empty database, dropped on exit."""

    @contextmanager
    def _fresh() -> Iterator[URL]:
        name = f"art_curator_test_{uuid.uuid4().hex[:8]}"
        run_sql(server_url, f'CREATE DATABASE "{name}"', autocommit=True)
        try:
            yield server_url.set(database=name)
        finally:
            run_sql(server_url, f'DROP DATABASE "{name}" WITH (FORCE)', autocommit=True)

    return _fresh


@pytest.fixture(scope="session")
def migrated_db(fresh_database: Callable) -> Iterator[URL]:
    """A database at `alembic upgrade head`, shared by read-only schema tests."""
    with fresh_database() as url:
        command.upgrade(alembic_config(url), "head")
        yield url
