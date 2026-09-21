# Postgres 16 + pgvector + PostGIS. Used by docker-compose locally and as the CI service container.
#
# Base is pgvector's image because postgis/postgis publishes no arm64 builds. PostGIS comes from
# the PGDG apt repo the official Postgres image already configures.
#
# Extensions are only made *available* here. `CREATE EXTENSION` is the first Alembic migration's
# job, so local, CI and RDS all take the same path.
FROM pgvector/pgvector:0.8.6-pg16-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-postgis-3 \
    && rm -rf /var/lib/apt/lists/*
