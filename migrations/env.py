import asyncio
from logging.config import fileConfig

from alembic import context
from geoalchemy2 import alembic_helpers
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from art_curator.config import get_settings
from art_curator.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Tables that extensions create in `public`; they are not ours to migrate.
EXTENSION_TABLES = {"spatial_ref_sys"}


def _database_url() -> str:
    # Tests point Alembic at a throwaway database via sqlalchemy.url; otherwise config.py decides.
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    if type_ == "table" and name in EXTENSION_TABLES:
        return False
    return alembic_helpers.include_object(obj, name, type_, reflected, compare_to)


CONFIGURE_KWARGS = dict(
    target_metadata=target_metadata,
    include_object=_include_object,
    process_revision_directives=alembic_helpers.writer,
    render_item=alembic_helpers.render_item,
)


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **CONFIGURE_KWARGS,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, **CONFIGURE_KWARGS)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        {"sqlalchemy.url": _database_url()}, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    # A caller may hand over its own (sync-adapted) connection, e.g. from inside run_sync.
    connection = config.attributes.get("connection")
    if connection is None:
        asyncio.run(run_async_migrations())
    else:
        do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
