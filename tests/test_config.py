import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from art_curator.cli import app
from art_curator.config import Settings, get_settings

ENV_VARS = ("AWS_REGION", "CHAT_MODEL", "EXTRACT_MODEL", "EMBED_MODEL", "DATABASE_URL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults():
    s = Settings(_env_file=None)
    assert s.aws_region == "eu-west-1"
    assert s.chat_model.startswith("anthropic.")
    assert s.extract_model.startswith("global.anthropic.")
    assert s.embed_model is None
    assert s.database_url.startswith("postgresql+asyncpg://")


def _tf_default(name: str) -> str:
    tf = (Path(__file__).resolve().parents[1] / "infra/terraform/variables.tf").read_text()
    match = re.search(rf'variable "{name}" \{{.*?default\s*=\s*"([^"]+)"', tf, re.DOTALL)
    assert match, f"no default for {name} in variables.tf"
    return match.group(1)


def test_model_defaults_match_terraform():
    # A mismatch isn't caught anywhere else: the app role's IAM just denies the call.
    s = Settings(_env_file=None)
    assert s.chat_model == _tf_default("chat_model_id")
    assert s.extract_model == "global." + _tf_default("extract_model_id")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "anthropic.claude-sonnet-5")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    s = Settings(_env_file=None)
    assert s.chat_model == "anthropic.claude-sonnet-5"
    assert s.aws_region == "us-east-1"


def test_blank_embed_model_is_unset(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "")
    assert Settings(_env_file=None).embed_model is None


def test_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("EXTRACT_MODEL=anthropic.claude-haiku-5\nUNRELATED=1\n")
    assert Settings(_env_file=env).extract_model == "anthropic.claude-haiku-5"


def test_cli_prints_config(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "anthropic.claude-sonnet-5")
    result = CliRunner().invoke(app, ["config"])
    assert result.exit_code == 0
    assert "chat_model=anthropic.claude-sonnet-5" in result.stdout
    assert "embed_model=\n" in result.stdout


def test_cli_hides_database_password(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://app:s3cret@db.example:5432/app")
    result = CliRunner().invoke(app, ["config"])
    assert result.exit_code == 0
    assert "s3cret" not in result.stdout
    assert "database_url=postgresql+asyncpg://app:***@db.example:5432/app" in result.stdout
