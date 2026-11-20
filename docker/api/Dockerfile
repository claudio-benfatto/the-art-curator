# The HTTP API (`api/main.py`) for docker-compose's `api` service. Local only through P3: P7
# replaces the mounted ~/.aws with a task role (PLAN.md § 8).
#
# Built from the repository root (`context: .`); `.dockerignore` keeps `.env` and `.venv` out.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 PATH="/app/.venv/bin:$PATH"

# Dependencies first, so a source change does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project
COPY src ./src
RUN uv sync --locked

EXPOSE 8000
# 0.0.0.0 inside the container; Compose publishes the port on the host's 127.0.0.1 only.
CMD ["uvicorn", "art_curator.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
