"""Spans around model calls: ids on the row, usage on the span, bodies masked by purpose."""

import asyncio
import json

import pytest
from anthropic import BadRequestError
from opentelemetry.trace import StatusCode
from pydantic import SecretStr

from art_curator.config import Settings
from art_curator.obs.telemetry import BODY_KEYS, BODY_PURPOSES, langfuse_exporter
from tests.llm_stub import StubLlm

OPUS = "anthropic.claude-opus-5"
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
SECRET_PAGE_TEXT = "Verbatim third-party venue prose that must never be exported."


def _chat(stub: StubLlm, purpose: str = "chat", content: str = "hi") -> None:
    asyncio.run(
        stub.client.create_message(
            purpose=purpose,
            model=OPUS,
            max_tokens=64,
            messages=[{"role": "user", "content": content}],
        )
    )


def _converse(stub: StubLlm, purpose: str, text: str) -> None:
    stub.converse_stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "summary"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
            "metrics": {"latencyMs": 1},
        },
    )
    asyncio.run(
        stub.client.converse_message(
            purpose=purpose, model=HAIKU, messages=[{"role": "user", "content": [{"text": text}]}]
        )
    )


def test_row_carries_the_span_ids():
    stub = StubLlm()
    stub.reply(input_tokens=10, output_tokens=5)
    _chat(stub)

    [row] = stub.calls
    [span] = stub.spans
    assert row.trace_id == format(span.context.trace_id, "032x")
    assert row.span_id == format(span.context.span_id, "016x")
    assert len(row.trace_id) == 32 and len(row.span_id) == 16


def test_calls_in_one_trace_share_the_trace_id():
    stub = StubLlm()
    tracer = stub.client._tracer
    stub.reply()
    stub.reply()
    with tracer.start_as_current_span("request"):
        _chat(stub)
        _chat(stub)

    first, second = stub.calls
    assert first.trace_id == second.trace_id
    assert first.span_id != second.span_id


def test_span_carries_usage_and_our_cost():
    stub = StubLlm()
    stub.reply(input_tokens=400, output_tokens=600, cache_read_input_tokens=2500)
    _chat(stub)

    attrs = next(iter(stub.spans)).attributes
    assert attrs["art_curator.purpose"] == "chat"
    assert attrs["gen_ai.request.model"] == OPUS
    assert attrs["gen_ai.usage.input_tokens"] == 400
    assert attrs["langfuse.observation.type"] == "generation"
    assert json.loads(attrs["langfuse.observation.cost_details"]) == {"total": 0.01825}
    usage = json.loads(attrs["langfuse.observation.usage_details"])
    assert usage["cache_read_input_tokens"] == 2500


def test_failed_call_has_ids_and_an_error_span():
    stub = StubLlm()
    stub.fail(400)
    with pytest.raises(BadRequestError):
        _chat(stub)

    [row] = stub.calls
    [span] = stub.spans
    assert row.span_id == format(span.context.span_id, "016x")
    assert span.status.status_code is StatusCode.ERROR


# --- Masking (CLAUDE.md § 1) ---------------------------------------------------------------------


def test_chat_bodies_are_exported():
    stub = StubLlm()
    stub.reply("a reply")
    _chat(stub, content="where should I go?")

    attrs = next(iter(stub.spans)).attributes
    assert "where should I go?" in attrs["langfuse.observation.input"]
    assert "a reply" in attrs["langfuse.observation.output"]


@pytest.mark.parametrize("purpose", ["extract", "judge"])
def test_page_text_purposes_never_export_bodies(purpose):
    stub = StubLlm()
    _converse(stub, purpose, SECRET_PAGE_TEXT)

    [span] = stub.spans
    assert not set(BODY_KEYS) & set(span.attributes)
    assert not any(SECRET_PAGE_TEXT in str(v) for v in span.attributes.values())


def test_extract_on_mantle_is_masked_too():
    stub = StubLlm()
    stub.reply()
    _chat(stub, purpose="extract", content=SECRET_PAGE_TEXT)

    [span] = stub.spans
    assert not any(SECRET_PAGE_TEXT in str(v) for v in span.attributes.values())


def test_body_allowlist_is_deliberate():
    # Widening this list is a copyright decision (CLAUDE.md § 1), not a refactor.
    assert {"chat", "smoke"} == BODY_PURPOSES


# --- Langfuse export -----------------------------------------------------------------------------


def test_langfuse_exporter_targets_the_otlp_endpoint():
    settings = Settings(
        _env_file=None,
        langfuse_host="http://localhost:3000/",
        langfuse_public_key="pk",
        langfuse_secret_key=SecretStr("sk"),
    )
    exporter = langfuse_exporter(settings)
    assert exporter._endpoint == "http://localhost:3000/api/public/otel/v1/traces"
    assert exporter._headers["Authorization"] == "Basic cGs6c2s="  # base64("pk:sk")
    assert exporter._headers["x-langfuse-ingestion-version"] == "4"


def test_langfuse_is_off_by_default():
    assert Settings(_env_file=None).langfuse_enabled is False
