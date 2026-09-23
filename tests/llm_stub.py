"""A model-free `LlmClient` for tests. CI never calls a model (CLAUDE.md § 10).

The real `LlmClient` runs unchanged; only the wire is fake:
- Both Messages transports (Mantle and InvokeModel) share an httpx2 `MockTransport`, so the
  SDK's own request building and response parsing run as in production. Queue responses with
  `StubLlm.reply` / `StubLlm.fail`.
- Converse gets a botocore `Stubber` on a real (credential-less) boto3 client.

Rows go to `StubLlm.calls` instead of the database, unless a recorder is passed. Spans go to an
in-memory exporter (`StubLlm.spans`) — exactly what Langfuse would receive.
"""

import json
from collections import deque
from typing import Any

import boto3
import httpx2
from anthropic import AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from botocore.stub import Stubber
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from art_curator.db.models import LlmCall
from art_curator.llm.client import LlmClient, Recorder

REGION = "eu-west-1"


class StubLlm:
    def __init__(self, record: Recorder | None = None) -> None:
        self.calls: list[LlmCall] = []
        self.requests: list[dict[str, Any]] = []  # Mantle request bodies, as sent
        self._responses: deque[httpx2.Response] = deque()

        transport = httpx2.MockTransport(self._handle)
        mantle = AsyncAnthropicBedrockMantle(
            aws_region=REGION,
            skip_auth=True,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=transport),
        )
        # The InvokeModel path (chat on an inference profile). A bearer token keeps the SDK from
        # reaching for AWS credentials; the request never leaves the mock transport.
        invoke = AsyncAnthropicBedrock(
            aws_region=REGION,
            api_key="test",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=transport),
        )
        runtime = boto3.client(
            "bedrock-runtime",
            region_name=REGION,
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        self.converse_stub = Stubber(runtime)
        self.converse_stub.activate()

        # A private provider, so tests never touch the global one.
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.client = LlmClient(
            mantle=mantle,
            invoke=invoke,
            runtime=runtime,
            record=record or self._capture,
            tracer=provider.get_tracer("test"),
        )

    @property
    def spans(self) -> tuple[ReadableSpan, ...]:
        return self.exporter.get_finished_spans()

    async def _capture(self, row: LlmCall) -> None:
        self.calls.append(row)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return self._responses.popleft()

    def reply(
        self,
        text: str = "ok",
        *,
        model: str = "anthropic.claude-opus-5",
        stop_reason: str = "end_turn",
        request_id: str = "req_stub",
        **usage: Any,
    ) -> None:
        body = {
            "id": "msg_stub",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, **usage},
        }
        self._responses.append(httpx2.Response(200, json=body, headers={"request-id": request_id}))

    def fail(self, status: int = 400, *, request_id: str = "req_err") -> None:
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": "stub"}}
        self._responses.append(
            httpx2.Response(status, json=body, headers={"request-id": request_id})
        )
