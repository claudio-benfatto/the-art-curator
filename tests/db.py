"""Helpers for tests that need a real database (see conftest.py for the fixtures)."""

import asyncio
from pathlib import Path

from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[1]


def alembic_config(url: URL) -> Config:
    # TOML only: loading alembic.ini would run fileConfig and reset pytest's logging.
    cfg = Config(toml_file=ROOT / "pyproject.toml")
    cfg.set_main_option("sqlalchemy.url", url.render_as_string(hide_password=False))
    return cfg


def run_sql(url: URL, *statements: str, autocommit: bool = False) -> list[list[tuple]]:
    async def _run() -> list[list[tuple]]:
        engine = create_async_engine(url, isolation_level="AUTOCOMMIT" if autocommit else None)
        try:
            async with engine.begin() as conn:
                results = []
                for stmt in statements:
                    result = await conn.execute(text(stmt))
                    results.append([tuple(r) for r in result] if result.returns_rows else [])
                return results
        finally:
            await engine.dispose()

    return asyncio.run(_run())
