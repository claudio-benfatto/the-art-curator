"""All configuration comes from the environment (or `.env`). Nothing else reads env vars."""

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    aws_region: str = "eu-west-1"

    # Bedrock model IDs. Chat is Claude-only (Mantle); the others may be any Bedrock model.
    chat_model: str = "anthropic.claude-opus-5"
    # A global inference profile, called via Bedrock Converse: Mantle refuses every model
    # for this account until AWS resolves it (CLAUDE.md, Current state). Must match Terraform's
    # extract_model_id with a `global.` prefix, or the app role's IAM denies it.
    extract_model: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    embed_model: str | None = None  # chosen by measurement in P3

    # Async driver for the app; Alembic uses the same URL. Default is the Compose `db` service.
    database_url: str = "postgresql+asyncpg://art_curator:art_curator@localhost:5432/art_curator"

    # Optional span export to Langfuse over OTLP. OTel + llm_calls work without it.
    # The key defaults match the Compose `langfuse` service's seeded project.
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = "pk-lf-local"
    langfuse_secret_key: SecretStr = SecretStr("sk-lf-local")

    @field_validator("embed_model", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: str | None) -> str | None:
        return v or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
