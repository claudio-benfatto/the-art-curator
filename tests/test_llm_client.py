"""`llm/client.py`: one `llm_calls` row per call, with the right tokens and cost (CLAUDE.md § 2)."""

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import pytest
from anthropic import APIStatusError, BadRequestError, PermissionDeniedError
from anthropic.types import Message
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from art_curator.db.models import LLM_PURPOSES
from art_curator.llm.client import (
    AUTO_CACHE,
    EMBED_DIMENSIONS,
    PROVIDER_CONVERSE,
    PROVIDER_INVOKE,
    PROVIDER_MANTLE,
    PURPOSES,
    MissingUsageError,
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


def test_create_message_refuses_to_stream():
    # A stream is recorded only by stream_message, which reads it to the end.
    stub = StubLlm()
    with pytest.raises(ValueError, match="stream_message"):
        asyncio.run(
            stub.client.create_message(
                purpose="chat", model=OPUS, max_tokens=64, messages=MESSAGES, stream=True
            )
        )
    assert stub.requests == [] and stub.calls == []


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


# --- Streaming (PLAN.md § 8, PR 33) --------------------------------------------------------------

USAGE = {
    "input_tokens": 120,
    "output_tokens": 45,
    "cache_read_input_tokens": 4000,
    "cache_creation_input_tokens": 900,
}
REPLY = "Three shows near Poblenou this weekend."


def stream(stub: StubLlm, model: str, **params: Any) -> tuple[list[Any], Message]:
    """Read one stream to the end: its events and its final message."""
    params = {"max_tokens": 64, "messages": MESSAGES, **params}

    async def main() -> tuple[list[Any], Message]:
        async with stub.client.stream_message(purpose="chat", model=model, **params) as s:
            events = [event async for event in s]
            return events, await s.get_final_message()

    return asyncio.run(main())


@pytest.mark.parametrize(
    ("model", "provider", "path"),
    [
        (CHAT, PROVIDER_INVOKE, f"/model/{CHAT}/invoke-with-response-stream"),
        (OPUS, PROVIDER_MANTLE, "/v1/messages"),
    ],
)
def test_a_stream_is_one_row_costed_as_the_same_call_unstreamed(model, provider, path):
    streamed = StubLlm()
    streamed.stream_reply(REPLY, model=model, request_id="req_s", **USAGE)
    events, message = stream(streamed, model)

    unstreamed = StubLlm()
    unstreamed.reply(REPLY, model=model, **USAGE)
    asyncio.run(
        unstreamed.client.create_message(
            purpose="chat", model=model, max_tokens=64, messages=MESSAGES
        )
    )

    deltas = [e.delta.text for e in events if e.type == "content_block_delta"]
    assert len(deltas) > 1 and "".join(deltas) == message.content[0].text == REPLY
    assert streamed.paths[0].endswith(path)
    [row], [same] = streamed.calls, unstreamed.calls
    assert (row.provider, row.request_id, row.stop_reason) == (provider, "req_s", "end_turn")
    tokens = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    assert [getattr(row, t) for t in tokens] == [getattr(same, t) for t in tokens]
    assert row.cost_usd == same.cost_usd > 0
    assert row.error_type is None


def test_a_streamed_request_is_the_unstreamed_one():
    # InvokeModel takes `stream` out of the body and into the URL: the bodies are byte-identical,
    # so a streamed turn reads the cache an unstreamed one wrote.
    streamed, unstreamed = StubLlm(), StubLlm()
    streamed.stream_reply(model=CHAT)
    unstreamed.reply(model=CHAT)
    stream(streamed, CHAT, messages=cached_tail(MESSAGES))
    asyncio.run(
        unstreamed.client.create_message(
            purpose="chat", model=CHAT, max_tokens=64, messages=cached_tail(MESSAGES)
        )
    )
    assert streamed.requests == unstreamed.requests


def test_a_streamed_message_comes_back_whole():
    blocks = [
        {"type": "thinking", "thinking": "Weighing it.", "signature": "sig-1"},
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "t1", "name": "search_events", "input": {"near": "Gràcia"}},
    ]
    stub = StubLlm()
    stub.stream_reply(content=blocks, stop_reason="tool_use", model=CHAT)
    _, message = stream(stub, CHAT)

    assert [b.to_dict(exclude_none=True) for b in message.content] == blocks
    assert message.stop_reason == "tool_use"


@pytest.mark.parametrize("model", [CHAT, OPUS])
def test_a_broken_stream_is_a_row_with_the_tokens_it_reported(model):
    stub = StubLlm()
    stub.stream_reply(REPLY, model=model, break_after=4, **USAGE)  # start, block start, 2 words
    with pytest.raises(APIStatusError):
        stream(stub, model)

    [row] = stub.calls
    assert row.error_type is not None
    # message_start's input and cache tokens are billed; the output count never arrived.
    assert (row.input_tokens, row.cache_read_input_tokens, row.output_tokens) == (120, 4000, 0)
    assert row.cost_usd > 0


def test_a_stream_refused_before_it_starts_is_a_row():
    stub = StubLlm()
    stub.fail(403, request_id="req_denied")
    with pytest.raises(PermissionDeniedError):
        stream(stub, CHAT)

    [row] = stub.calls
    assert (row.error_type, row.request_id, row.cost_usd) == (
        "PermissionDeniedError",
        "req_denied",
        Decimal(0),
    )


def test_a_stream_the_caller_abandons_is_a_row():
    # What a client disconnecting mid-reply does to /chat: the generator reading the stream is
    # closed at the event it was waiting on.
    stub = StubLlm()
    stub.stream_reply(REPLY, model=CHAT, **USAGE)

    async def words() -> AsyncIterator[str]:
        async with stub.client.stream_message(
            purpose="chat", model=CHAT, max_tokens=64, messages=MESSAGES
        ) as s:
            async for event in s:
                if event.type == "content_block_delta":
                    yield event.delta.text

    async def main() -> str:
        reader = words()
        first = await anext(reader)
        await reader.aclose()
        return first

    assert asyncio.run(main()) == "Three "
    [row] = stub.calls
    assert row.error_type == "GeneratorExit"
    assert row.input_tokens == 120


def test_a_stream_left_unread_is_read_to_the_end_for_its_row():
    stub = StubLlm()
    stub.stream_reply(REPLY, model=CHAT, **USAGE)

    async def main() -> None:
        async with stub.client.stream_message(
            purpose="chat", model=CHAT, max_tokens=64, messages=MESSAGES
        ):
            pass

    asyncio.run(main())
    [row] = stub.calls
    assert (row.error_type, row.output_tokens, row.stop_reason) == (None, 45, "end_turn")


def test_stream_message_refuses_a_stream_flag_and_unpriced_models():
    stub = StubLlm()
    with pytest.raises(ValueError, match="stream"):
        stream(stub, CHAT, stream=True)
    with pytest.raises(UnknownModelError):
        stream(stub, "global.anthropic.claude-unpriced")
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


TITAN = "amazon.titan-embed-text-v2:0"
COHERE = "cohere.embed-v4:0"
VECTOR = [0.5] * EMBED_DIMENSIONS


def test_titan_embedding_writes_one_row():
    stub = StubLlm()
    stub.embed_reply({"embedding": VECTOR, "inputTextTokenCount": 7}, input_tokens=7)
    vector = asyncio.run(
        stub.client.embed(purpose="embed", model=TITAN, text="a show", input_type="document")
    )

    assert vector == VECTOR
    assert stub.embed_requests == [
        {"inputText": "a show", "dimensions": EMBED_DIMENSIONS, "normalize": True}
    ]
    [row] = stub.calls
    assert (row.provider, row.model, row.purpose) == (PROVIDER_INVOKE, TITAN, "embed")
    assert (row.input_tokens, row.output_tokens, row.request_id) == (7, 0, "aws-embed-stub")
    assert row.stop_reason is None
    assert row.cost_usd == Decimal("0.000000")  # 7 × $0.026004 / 1M rounds to nothing


@pytest.mark.parametrize(
    "embeddings",
    [[VECTOR], {"float": [VECTOR]}],
    ids=["list", "by-type"],
)
def test_cohere_embedding_reads_either_response_shape(embeddings):
    stub = StubLlm()
    stub.embed_reply({"id": "e1", "embeddings": embeddings}, input_tokens=50_000)
    vector = asyncio.run(
        stub.client.embed(purpose="embed", model=COHERE, text="sculpture", input_type="query")
    )

    assert vector == VECTOR
    assert stub.embed_requests == [
        {
            "texts": ["sculpture"],
            "input_type": "search_query",
            "embedding_types": ["float"],
            "output_dimension": EMBED_DIMENSIONS,
        }
    ]
    [row] = stub.calls
    assert row.input_tokens == 50_000
    assert row.cost_usd == Decimal("0.006000")  # 50k × $0.12 / 1M


def test_cohere_documents_are_embedded_as_documents():
    stub = StubLlm()
    stub.embed_reply({"embeddings": [VECTOR]}, input_tokens=3)
    asyncio.run(stub.client.embed(purpose="embed", model=COHERE, text="x", input_type="document"))
    assert stub.embed_requests[0]["input_type"] == "search_document"


def test_titan_count_in_the_body_is_enough():
    stub = StubLlm()
    stub.embed_reply({"embedding": VECTOR, "inputTextTokenCount": 9})
    asyncio.run(stub.client.embed(purpose="embed", model=TITAN, text="x", input_type="query"))
    assert stub.calls[0].input_tokens == 9


def test_embedding_without_a_token_count_is_an_error_row():
    """A call that cannot be priced is recorded as failed, never as free."""
    stub = StubLlm()
    stub.embed_reply({"embeddings": [VECTOR]})
    with pytest.raises(MissingUsageError):
        asyncio.run(stub.client.embed(purpose="embed", model=COHERE, text="x", input_type="query"))
    [row] = stub.calls
    assert (row.error_type, row.input_tokens) == ("MissingUsageError", 0)


def test_unknown_embedding_model_fails_before_calling():
    stub = StubLlm()
    with pytest.raises(ValueError, match="not an embedding model"):
        asyncio.run(stub.client.embed(purpose="embed", model=CHAT, text="x", input_type="query"))
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
