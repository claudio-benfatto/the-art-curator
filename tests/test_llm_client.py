"""`llm/client.py`: one `llm_calls` row per call, with the right tokens and cost (CLAUDE.md § 2)."""

import asyncio
from decimal import Decimal

import pytest
from anthropic import BadRequestError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from art_curator.db.models import LLM_PURPOSES
from art_curator.llm.client import (
    AUTO_CACHE,
    PROVIDER_CONVERSE,
    PROVIDER_INVOKE,
    PROVIDER_MANTLE,
    PURPOSES,
    cached_tail,
    db_recorder,
    is_profile_model,
    static_prefix,
)
from art_curator.llm.pricing import UnknownModelError
from tests.db import run_sql
from tests.llm_stub import StubLlm

OPUS = "anthropic.claude-opus-5"  # Mantle ID
CHAT = "global.anthropic.claude-opus-4-6-v1"  # inference profile → InvokeModel
HAIKU = "anthropic.claude-haiku-4-5"
MESSAGES = [{"role": "user", "content": "hi"}]


def test_purposes_match_the_schema_check():
    assert set(PURPOSES) == set(LLM_PURPOSES)


def test_message_writes_one_row_with_cost():
    stub = StubLlm()
    stub.reply(
        input_tokens=400,
        output_tokens=600,
        cache_read_input_tokens=2500,
        request_id="req_1",
    )
    msg = asyncio.run(
        stub.client.create_message(purpose="chat", model=OPUS, max_tokens=64, messages=MESSAGES)
    )

    assert msg.content[0].text == "ok"
    [row] = stub.calls
    assert (row.provider, row.model, row.purpose) == (PROVIDER_MANTLE, OPUS, "chat")
    assert (row.input_tokens, row.output_tokens, row.cache_read_input_tokens) == (400, 600, 2500)
    assert row.cost_usd == Decimal("0.018250")  # see test_pricing
    assert row.request_id == "req_1"
    assert row.stop_reason == "end_turn"
    assert row.error_type is None
    assert row.latency_ms >= 0


def test_cache_writes_are_priced_by_ttl():
    stub = StubLlm()
    stub.reply(
        cache_creation_input_tokens=3000,
        cache_creation={"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 1000},
    )
    asyncio.run(
        stub.client.create_message(purpose="chat", model=OPUS, max_tokens=64, messages=MESSAGES)
    )

    [row] = stub.calls
    assert row.cache_creation_input_tokens == 3000
    # 2000 × 6.25 + 1000 × 10 = 22 500 µ$
    assert row.cost_usd == Decimal("0.022500")


def test_failed_call_is_recorded_and_reraised():
    stub = StubLlm()
    stub.fail(400, request_id="req_bad")
    with pytest.raises(BadRequestError):
        asyncio.run(
            stub.client.create_message(
                purpose="extract", model=HAIKU, max_tokens=64, messages=MESSAGES
            )
        )

    [row] = stub.calls
    assert row.error_type == "BadRequestError"
    assert row.request_id == "req_bad"
    assert row.cost_usd == Decimal(0)
    assert row.input_tokens == 0


def test_unpriced_model_fails_before_calling():
    stub = StubLlm()
    with pytest.raises(UnknownModelError):
        asyncio.run(
            stub.client.create_message(
                purpose="chat", model="anthropic.claude-unpriced", max_tokens=64, messages=MESSAGES
            )
        )
    assert stub.requests == []
    assert stub.calls == []


def test_streaming_is_rejected_until_it_is_recorded():
    stub = StubLlm()
    with pytest.raises(ValueError, match="stream"):
        asyncio.run(
            stub.client.create_message(
                purpose="chat", model=OPUS, max_tokens=64, messages=MESSAGES, stream=True
            )
        )


def test_cache_layout_reaches_the_wire():
    stub = StubLlm()
    stub.reply()
    asyncio.run(
        stub.client.create_message(
            purpose="chat",
            model=OPUS,
            max_tokens=64,
            system=static_prefix("persona", "rules"),
            cache_control=AUTO_CACHE,
            messages=MESSAGES,
        )
    )

    [body] = stub.requests
    assert body["cache_control"] == {"type": "ephemeral"}
    first, last = body["system"]
    assert "cache_control" not in first
    assert last["cache_control"] == {"type": "ephemeral"}


def test_static_prefix_needs_a_block():
    with pytest.raises(ValueError):
        static_prefix()


# --- Transport routing (CLAUDE.md § 4) -----------------------------------------------------------


def test_profile_ids_route_to_invoke_and_bare_ids_to_mantle():
    assert is_profile_model(CHAT) and is_profile_model("eu.anthropic.claude-opus-4-6-v1")
    assert not is_profile_model(OPUS) and not is_profile_model("anthropic.claude-haiku-4-5")


def test_chat_model_goes_through_invoke():
    stub = StubLlm()
    stub.reply(input_tokens=10, output_tokens=5, model=CHAT)
    asyncio.run(
        stub.client.create_message(purpose="chat", model=CHAT, max_tokens=64, messages=MESSAGES)
    )

    [row] = stub.calls
    assert row.provider == PROVIDER_INVOKE
    assert row.cost_usd == Decimal("0.000175")  # Opus pricing: 10 × 5 + 5 × 25


def test_invoke_rejects_top_level_cache_control():
    # It would be accepted-and-ignored or rejected upstream; either way, silently uncached.
    stub = StubLlm()
    with pytest.raises(ValueError, match="cached_tail"):
        asyncio.run(
            stub.client.create_message(
                purpose="chat",
                model=CHAT,
                max_tokens=64,
                messages=MESSAGES,
                cache_control=AUTO_CACHE,
            )
        )
    assert stub.requests == [] and stub.calls == []


# --- Cache breakpoints (CLAUDE.md § 6) -----------------------------------------------------------


def test_cached_tail_marks_the_last_block_only():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
    ]
    out = cached_tail(messages)

    assert out[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out[-1]["content"][0]
    assert out[0] == {"role": "user", "content": "first"}  # input untouched
    assert messages[-1]["content"][-1] == {"type": "text", "text": "b"}


def test_cached_tail_wraps_a_string_message():
    out = cached_tail([{"role": "user", "content": "hi"}])
    assert out[0]["content"] == [
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
    ]


def test_cached_tail_needs_a_message():
    with pytest.raises(ValueError):
        cached_tail([])


def test_cached_tail_reaches_the_wire():
    stub = StubLlm()
    stub.reply(model=CHAT)
    asyncio.run(
        stub.client.create_message(
            purpose="chat",
            model=CHAT,
            max_tokens=64,
            system=static_prefix("persona"),
            messages=cached_tail(MESSAGES),
        )
    )

    [body] = stub.requests
    assert "cache_control" not in body
    assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_converse_writes_one_row():
    stub = StubLlm()
    stub.converse_stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 3300,
                "outputTokens": 600,
                "totalTokens": 3900,
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
            },
            "metrics": {"latencyMs": 10},
            "ResponseMetadata": {"RequestId": "aws-req-1"},
        },
    )
    resp = asyncio.run(
        stub.client.converse_message(
            purpose="judge",
            model=HAIKU,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )
    )

    assert resp["stopReason"] == "end_turn"
    [row] = stub.calls
    assert (row.provider, row.purpose, row.request_id) == (PROVIDER_CONVERSE, "judge", "aws-req-1")
    assert (row.input_tokens, row.output_tokens) == (3300, 600)
    assert row.cost_usd == Decimal("0.006300")


def test_converse_refuses_chat():
    stub = StubLlm()
    with pytest.raises(ValueError, match="Claude-only"):
        asyncio.run(stub.client.converse_message(purpose="chat", model=HAIKU, messages=[]))
    assert stub.calls == []


def test_db_recorder_writes_the_row(migrated_db):
    async def run() -> None:
        engine = create_async_engine(migrated_db)
        try:
            stub = StubLlm(record=db_recorder(async_sessionmaker(engine)))
            stub.reply(input_tokens=10, output_tokens=5, request_id="req_db")
            await stub.client.create_message(
                purpose="smoke", model=OPUS, max_tokens=8, messages=MESSAGES
            )
        finally:
            await engine.dispose()

    asyncio.run(run())
    [rows] = run_sql(
        migrated_db,
        "SELECT purpose, input_tokens, output_tokens, cost_usd FROM llm_calls "
        "WHERE request_id = 'req_db'",
    )
    # 10 × 5 + 5 × 25 = 175 µ$
    assert rows == [("smoke", 10, 5, Decimal("0.000175"))]
