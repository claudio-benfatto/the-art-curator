"""`cli smoke` end to end: stubbed model, real database, row checked by trace id."""

import pytest
from typer.testing import CliRunner

from art_curator import cli
from art_curator.config import get_settings
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.llm.client import EMBED_DIMENSIONS, db_recorder
from tests.db import run_sql
from tests.llm_stub import StubLlm

CONVERSE_OK = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 30, "outputTokens": 20, "totalTokens": 50},
    "metrics": {"latencyMs": 5},
    "ResponseMetadata": {"RequestId": "aws-smoke"},
}


def _clear_caches() -> None:
    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()


@pytest.fixture
def stub(migrated_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", migrated_db.render_as_string(hide_password=False))
    for var in ("EXTRACT_MODEL", "LANGFUSE_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    _clear_caches()
    stub = StubLlm(record=db_recorder(get_sessionmaker()))
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)
    yield stub
    _clear_caches()


def _smoke_rows(url) -> list[tuple]:
    [rows] = run_sql(
        url, "SELECT provider, cost_usd, error_type FROM llm_calls WHERE purpose = 'smoke'"
    )
    return rows


def test_smoke_passes_on_converse(stub, migrated_db):
    stub.converse_stub.add_response("converse", CONVERSE_OK)
    result = CliRunner().invoke(cli.app, ["smoke"])

    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "$0.000130" in result.output  # Haiku: 30 × 1 + 20 × 5 = 130 µ$
    assert ("bedrock-converse", pytest.approx(0.00013), None) in [
        (p, float(c), e) for p, c, e in _smoke_rows(migrated_db)
    ]


def test_smoke_fails_and_still_records_a_refused_call(stub, migrated_db):
    stub.converse_stub.add_client_error(
        "converse",
        service_error_code="AccessDeniedException",
        service_message="not available for this account",
        http_status_code=403,
    )
    result = CliRunner().invoke(cli.app, ["smoke"])

    assert result.exit_code == 1
    assert "FAIL: call raised AccessDeniedException" in result.output
    assert any(e == "AccessDeniedException" for _, _, e in _smoke_rows(migrated_db))


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("anthropic.claude-opus-5", "bedrock-mantle"),
        ("global.anthropic.claude-sonnet-4-6", "bedrock-invoke"),
    ],
)
def test_smoke_via_messages(stub, model, provider):
    stub.reply(input_tokens=12, output_tokens=3, model=model)
    result = CliRunner().invoke(cli.app, ["smoke", "--via", "messages", "--model", model])

    assert result.exit_code == 0, result.output
    assert f"{provider} {model}" in result.output


def test_smoke_via_embed(stub, migrated_db):
    stub.embed_reply({"embeddings": [[0.1] * EMBED_DIMENSIONS]}, input_tokens=10)
    result = CliRunner().invoke(
        cli.app, ["smoke", "--via", "embed", "--model", "cohere.embed-v4:0"]
    )

    assert result.exit_code == 0, result.output
    assert "bedrock-invoke cohere.embed-v4:0" in result.output
    assert "$0.000001" in result.output  # 10 × $0.12 / 1M = 1.2 µ$, stored to 6 dp


def test_smoke_fails_an_embedding_of_the_wrong_length(stub):
    stub.embed_reply({"embeddings": [[0.1] * 1536]}, input_tokens=10)
    result = CliRunner().invoke(
        cli.app, ["smoke", "--via", "embed", "--model", "cohere.embed-v4:0"]
    )

    assert result.exit_code == 1
    assert f"embedding has 1536 dimensions, not {EMBED_DIMENSIONS}" in result.output
