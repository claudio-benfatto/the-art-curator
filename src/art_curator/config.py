"""All configuration comes from the environment (or `.env`). Nothing else reads env vars."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    aws_region: str = "eu-west-1"

    # Bedrock model IDs. Chat is Claude-only; the others may be any Bedrock model.
    chat_model: str = "anthropic.claude-opus-5"
    extract_model: str = "anthropic.claude-haiku-4-5"
    embed_model: str | None = None  # chosen by measurement in P3

    # Async driver for the app; Alembic uses the same URL. Default is the Compose `db` service.
    database_url: str = "postgresql+asyncpg://art_curator:art_curator@localhost:5432/art_curator"

    @field_validator("embed_model", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: str | None) -> str | None:
        return v or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
