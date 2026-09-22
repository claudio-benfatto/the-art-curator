"""The only place a model is called (CLAUDE.md § 2). Every call — success or failure — writes one
`llm_calls` row with tokens, cost and latency before control returns to the caller.

Two transports, both Bedrock:
- Mantle (the Messages-API endpoint) for Claude. Chat is Claude-only, so chat goes here.
- `bedrock-runtime` Converse for any other Bedrock model (extraction, judge).

Streaming is not wrapped yet: `/chat` lands in P3 and will add a recorded stream here.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any, Literal, get_args

import boto3
from anthropic import AsyncAnthropicBedrockMantle
from anthropic.types import Message, TextBlockParam
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from art_curator.config import get_settings
from art_curator.db.models import LlmCall
from art_curator.db.session import get_sessionmaker
from art_curator.llm.pricing import Usage, cost_usd, price_for

Purpose = Literal["chat", "extract", "embed", "judge", "smoke"]
PURPOSES: tuple[str, ...] = get_args(Purpose)

PROVIDER_MANTLE = "bedrock-mantle"
PROVIDER_CONVERSE = "bedrock-converse"

# Top-level automatic caching for the conversation tail (CLAUDE.md § 6).
AUTO_CACHE = {"type": "ephemeral"}

Recorder = Callable[[LlmCall], Awaitable[None]]


def static_prefix(*texts: str) -> list[TextBlockParam]:
    """System prompt blocks carrying the one explicit cache breakpoint, on the last block.

    The API renders tools → system → messages, so this breakpoint caches the tool definitions
    too. Pass alongside `cache_control=AUTO_CACHE` so the conversation tail is cached as well.
    Nothing per-request (timestamps, ids) may go in `texts` — it would break the prefix.
    """
    if not texts:
        raise ValueError("static_prefix needs at least one block")
    blocks: list[TextBlockParam] = [{"type": "text", "text": t} for t in texts]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def db_recorder(sessionmaker: async_sessionmaker[AsyncSession]) -> Recorder:
    """Write each row in its own transaction, so a caller's rollback can't erase spend."""

    async def record(row: LlmCall) -> None:
        async with sessionmaker() as session, session.begin():
            session.add(row)

    return record


@dataclass(frozen=True)
class _Outcome:
    usage: Usage
    request_id: str | None
    stop_reason: str | None


class LlmClient:
    """Wraps the model transports. Recording failures propagate: a call that can't be recorded
    is an error, not a warning."""

    def __init__(
        self,
        *,
        mantle: AsyncAnthropicBedrockMantle,
        runtime: Any,  # boto3 bedrock-runtime client (untyped)
        record: Recorder,
    ) -> None:
        self._mantle = mantle
        self._runtime = runtime
        self._record = record

    async def create_message(self, *, purpose: Purpose, model: str, **params: Any) -> Message:
        """Messages API on Mantle. `params` are passed through (system, messages, tools,
        thinking, output_config, cache_control, max_tokens, ...)."""
        if params.get("stream"):
            raise ValueError("streaming is not recorded yet; call without stream=True")

        async def call() -> tuple[Message, _Outcome]:
            msg = await self._mantle.messages.create(model=model, **params)
            return msg, _outcome_from_message(msg)

        return await self._recorded(PROVIDER_MANTLE, purpose, model, call)

    async def converse(self, *, purpose: Purpose, model: str, **params: Any) -> dict[str, Any]:
        """Bedrock Converse for non-Claude models. `params` use Converse's own field names
        (messages, system, inferenceConfig, toolConfig, ...)."""
        if purpose == "chat":
            raise ValueError("chat is Claude-only and goes through create_message (CLAUDE.md § 4)")

        async def call() -> tuple[dict[str, Any], _Outcome]:
            resp = await asyncio.to_thread(self._runtime.converse, modelId=model, **params)
            return resp, _outcome_from_converse(resp)

        return await self._recorded(PROVIDER_CONVERSE, purpose, model, call)

    async def _recorded[T](
        self,
        provider: str,
        purpose: Purpose,
        model: str,
        call: Callable[[], Awaitable[tuple[T, _Outcome]]],
    ) -> T:
        if purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {purpose!r}")
        price_for(model)  # unpriced model: fail before spending anything

        start = time.perf_counter()
        try:
            result, outcome = await call()
        except Exception as exc:
            await self._record(
                LlmCall(
                    provider=provider,
                    model=model,
                    purpose=purpose,
                    request_id=_request_id_from_error(exc),
                    input_tokens=0,
                    output_tokens=0,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    cost_usd=Decimal(0),
                    latency_ms=_elapsed_ms(start),
                    error_type=type(exc).__name__,
                )
            )
            raise
        latency_ms = _elapsed_ms(start)

        usage = outcome.usage
        await self._record(
            LlmCall(
                provider=provider,
                model=model,
                purpose=purpose,
                request_id=outcome.request_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=usage.cache_write_tokens,
                cache_read_input_tokens=usage.cache_read_tokens,
                cost_usd=cost_usd(model, usage),
                latency_ms=latency_ms,
                stop_reason=outcome.stop_reason,
            )
        )
        return result


def _elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _outcome_from_message(msg: Message) -> _Outcome:
    u = msg.usage
    written = u.cache_creation_input_tokens or 0
    if u.cache_creation is not None:
        write_5m = u.cache_creation.ephemeral_5m_input_tokens
        write_1h = u.cache_creation.ephemeral_1h_input_tokens
    else:  # no TTL breakdown: every write used the default 5-minute TTL
        write_5m, write_1h = written, 0
    return _Outcome(
        usage=Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_write_5m_tokens=write_5m,
            cache_write_1h_tokens=write_1h,
            cache_read_tokens=u.cache_read_input_tokens or 0,
        ),
        request_id=msg._request_id,
        stop_reason=msg.stop_reason,
    )


def _outcome_from_converse(resp: dict[str, Any]) -> _Outcome:
    # Converse's inputTokens excludes cache reads/writes; writes carry no TTL split (5m default).
    u = resp.get("usage", {})
    return _Outcome(
        usage=Usage(
            input_tokens=u.get("inputTokens", 0),
            output_tokens=u.get("outputTokens", 0),
            cache_write_5m_tokens=u.get("cacheWriteInputTokens", 0),
            cache_read_tokens=u.get("cacheReadInputTokens", 0),
        ),
        request_id=resp.get("ResponseMetadata", {}).get("RequestId"),
        stop_reason=resp.get("stopReason"),
    )


def _request_id_from_error(exc: Exception) -> str | None:
    # anthropic.APIStatusError carries request_id; botocore ClientError carries ResponseMetadata.
    if request_id := getattr(exc, "request_id", None):
        return request_id
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return response.get("ResponseMetadata", {}).get("RequestId")
    return None


@lru_cache
def get_llm_client() -> LlmClient:
    settings = get_settings()
    return LlmClient(
        mantle=AsyncAnthropicBedrockMantle(aws_region=settings.aws_region),
        runtime=boto3.client("bedrock-runtime", region_name=settings.aws_region),
        record=db_recorder(get_sessionmaker()),
    )
