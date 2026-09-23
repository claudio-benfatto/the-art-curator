"""OpenTelemetry setup and the span attributes `llm/client.py` puts on every model call.

Spans always exist once `setup_telemetry()` has run, so every `llm_calls` row carries a trace
and span id. Export to Langfuse (OTLP/HTTP) is optional and off by default; without it, spans
still correlate rows within a request.

Bodies (prompt and completion text) go on spans only for purposes in `BODY_PURPOSES`. Extraction
prompts contain venue-page text, and judge prompts will compare against it — third-party prose
that must not reach a persistent store (CLAUDE.md § 1). Langfuse is one, so those are masked.
"""

import base64
import json
from collections.abc import Mapping
from decimal import Decimal
from functools import lru_cache
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span

from art_curator.config import Settings, get_settings

TRACER_NAME = "art_curator"
SERVICE_NAME = "art-curator"

# Allowlist, not denylist: a new purpose stays masked until someone decides otherwise.
BODY_PURPOSES = frozenset({"chat", "smoke"})
BODY_KEYS = ("langfuse.observation.input", "langfuse.observation.output")


def langfuse_exporter(settings: Settings) -> OTLPSpanExporter:
    creds = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key.get_secret_value()}"
    return OTLPSpanExporter(
        endpoint=f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
        headers={
            "Authorization": f"Basic {base64.b64encode(creds.encode()).decode()}",
            "x-langfuse-ingestion-version": "4",
        },
    )


@lru_cache
def setup_telemetry() -> TracerProvider:
    """Install the global tracer provider, once per process."""
    settings = get_settings()
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    if settings.langfuse_enabled:
        provider.add_span_processor(BatchSpanProcessor(langfuse_exporter(settings)))
    trace.set_tracer_provider(provider)
    return provider


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def span_ids(span: Span) -> tuple[str | None, str | None]:
    """Hex trace and span id, as stored in `llm_calls`; None for a non-recording span."""
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None, None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def request_attributes(*, provider: str, model: str, purpose: str) -> dict[str, Any]:
    return {
        "art_curator.purpose": purpose,
        "art_curator.provider": provider,
        "gen_ai.system": "aws.bedrock",
        "gen_ai.request.model": model,
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": model,
    }


def usage_attributes(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    cost_usd: Decimal,
    stop_reason: str | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "langfuse.observation.usage_details": json.dumps(
            {
                "input": input_tokens,
                "output": output_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            }
        ),
        # Our cost, from pricing.yaml — Langfuse doesn't know Bedrock model ids.
        "langfuse.observation.cost_details": json.dumps({"total": float(cost_usd)}),
    }
    if stop_reason is not None:
        attrs["art_curator.stop_reason"] = stop_reason
    return attrs


def body_attributes(purpose: str, *, input: Any, output: Any) -> dict[str, str]:
    """Prompt/completion as span attributes — empty unless `purpose` may carry bodies."""
    if purpose not in BODY_PURPOSES:
        return {}
    return {BODY_KEYS[0]: _to_json(input), BODY_KEYS[1]: _to_json(output)}


def _to_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def _json_default(value: Any) -> Any:
    # SDK content blocks (pydantic) appear in histories that echo `response.content` back.
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return str(value)
