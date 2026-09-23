"""The only place a model is called (CLAUDE.md § 2). Every call — success or failure — writes one
`llm_calls` row with tokens, cost and latency before control returns to the caller.

Two transports, both Bedrock:
- Mantle (the Messages-API endpoint) for Claude. Chat is Claude-only, so chat goes here.
- Converse (the runtime endpoint) for any Bedrock model — extraction, judge, and for now
  Claude too, while Mantle is blocked for this account (CLAUDE.md, Current state).

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
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from art_curator.config import get_settings
from art_curator.db.models import LlmCall
from art_curator.db.session import get_sessionmaker
from art_curator.llm.pricing import Usage, cost_usd, price_for
from art_curator.obs import telemetry

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
    output: Any  # completion body, for the span (masked per purpose)


class LlmClient:
    """Wraps the model transports. Each call runs in its own OTel span, whose ids land on the
    `llm_calls` row. Recording failures propagate: a call that can't be recorded is an error,
    not a warning."""

    def __init__(
        self,
        *,
        mantle: AsyncAnthropicBedrockMantle,
        runtime: Any,  # boto3 bedrock-runtime client (untyped)
        record: Recorder,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self._mantle = mantle
        self._runtime = runtime
        self._record = record
        self._tracer = tracer or telemetry.get_tracer()

    async def create_message(self, *, purpose: Purpose, model: str, **params: Any) -> Message:
        """Messages API on Mantle. `params` are passed through (system, messages, tools,
        thinking, output_config, cache_control, max_tokens, ...)."""
        if params.get("stream"):
            raise ValueError("streaming is not recorded yet; call without stream=True")

        async def call() -> tuple[Message, _Outcome]:
            msg = await self._mantle.messages.create(model=model, **params)
            return msg, _outcome_from_message(msg)

        return await self._recorded(PROVIDER_MANTLE, purpose, model, params, call)

    @property
    def tracer(self) -> trace.Tracer:
        """For callers that open a parent span, so their calls share its trace id."""
        return self._tracer

    async def converse_message(
        self, *, purpose: Purpose, model: str, **params: Any
    ) -> dict[str, Any]:
        """Bedrock runtime Converse: any Bedrock model, including Claude via an inference
        profile. `params` use Converse's own field names (messages, system, inferenceConfig,
        toolConfig, ...). Not named `converse` so the architecture grep can tell it apart from
        a direct SDK call."""
        if purpose == "chat":
            raise ValueError("chat is Claude-only and goes through create_message (CLAUDE.md § 4)")

        async def call() -> tuple[dict[str, Any], _Outcome]:
            resp = await asyncio.to_thread(self._runtime.converse, modelId=model, **params)
            return resp, _outcome_from_converse(resp)

        return await self._recorded(PROVIDER_CONVERSE, purpose, model, params, call)

    async def _recorded[T](
        self,
        provider: str,
        purpose: Purpose,
        model: str,
        request: dict[str, Any],
        call: Callable[[], Awaitable[tuple[T, _Outcome]]],
    ) -> T:
        if purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {purpose!r}")
        price_for(model)  # unpriced model: fail before spending anything

        # The span records and re-raises exceptions itself (status ERROR).
        with self._tracer.start_as_current_span(
            f"llm.{purpose}", kind=trace.SpanKind.CLIENT
        ) as span:
            span.set_attributes(
                telemetry.request_attributes(provider=provider, model=model, purpose=purpose)
            )
            trace_id, span_id = telemetry.span_ids(span)
            row = LlmCall(
                trace_id=trace_id,
                span_id=span_id,
                provider=provider,
                model=model,
                purpose=purpose,
                input_tokens=0,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                cost_usd=Decimal(0),
            )

            start = time.perf_counter()
            try:
                result, outcome = await call()
            except Exception as exc:
                row.request_id = _request_id_from_error(exc)
                row.latency_ms = _elapsed_ms(start)
                row.error_type = type(exc).__name__
                await self._record(row)
                raise
            row.latency_ms = _elapsed_ms(start)

            usage = outcome.usage
            row.request_id = outcome.request_id
            row.input_tokens = usage.input_tokens
            row.output_tokens = usage.output_tokens
            row.cache_creation_input_tokens = usage.cache_write_tokens
            row.cache_read_input_tokens = usage.cache_read_tokens
            row.cost_usd = cost_usd(model, usage)
            row.stop_reason = outcome.stop_reason

            span.set_attributes(
                telemetry.usage_attributes(
                    input_tokens=row.input_tokens,
                    output_tokens=row.output_tokens,
                    cache_creation_input_tokens=row.cache_creation_input_tokens,
                    cache_read_input_tokens=row.cache_read_input_tokens,
                    cost_usd=row.cost_usd,
                    stop_reason=row.stop_reason,
                )
            )
            span.set_attributes(
                telemetry.body_attributes(purpose, input=request, output=outcome.output)
            )
            await self._record(row)
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
        output=msg.content,
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
        output=resp.get("output"),
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
    telemetry.setup_telemetry()
    return LlmClient(
        mantle=AsyncAnthropicBedrockMantle(aws_region=settings.aws_region),
        runtime=boto3.client("bedrock-runtime", region_name=settings.aws_region),
        record=db_recorder(get_sessionmaker()),
    )
