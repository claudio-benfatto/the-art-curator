"""A model-free `LlmClient` for tests. CI never calls a model (CLAUDE.md § 10).

The real `LlmClient` runs unchanged; only the wire is fake:
- Both Messages transports (Mantle and InvokeModel) share an httpx2 `MockTransport`, so the
  SDK's own request building and response parsing run as in production. Queue responses with
  `StubLlm.reply` / `StubLlm.fail`, and streams with `StubLlm.stream_reply`: InvokeModel's as
  AWS event-stream frames, Mantle's as server-sent events, the encodings each endpoint sends.
- Converse gets a botocore `Stubber` on a real (credential-less) boto3 client, so request
  parameters are validated against the service model. Queue a forced-tool answer with
  `StubLlm.tool_reply`; what was sent is in `StubLlm.converse_requests`.
- Embeddings (InvokeModel on the same boto3 client) share that `Stubber`. Queue a response
  with `StubLlm.embed_reply`; the request bodies sent are in `StubLlm.embed_requests`.

Rows go to `StubLlm.calls` instead of the database, unless a recorder is passed. Spans go to an
in-memory exporter (`StubLlm.spans`) — exactly what Langfuse would receive.
"""

import base64
import binascii
import io
import json
import re
import struct
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import boto3
import httpx2
from anthropic import AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
from botocore.response import StreamingBody
from botocore.stub import Stubber
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from art_curator.db.models import LlmCall
from art_curator.llm.client import INPUT_TOKENS_HEADER, LlmClient, Recorder

REGION = "eu-west-1"
STRING_HEADER = b"\x07"  # an event-stream header's value type: string


class StubLlm:
    def __init__(self, record: Recorder | None = None) -> None:
        self.calls: list[LlmCall] = []
        self.requests: list[dict[str, Any]] = []  # Messages request bodies (both transports)
        self.paths: list[str] = []  # their URL paths: which endpoint each request reached
        # A stream is queued as a function of the request, since its encoding depends on the
        # endpoint the request reached.
        self._responses: deque[httpx2.Response | Callable[[httpx2.Request], httpx2.Response]] = (
            deque()
        )

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
        self.converse_requests: list[dict[str, Any]] = []  # Converse parameters, as sent
        runtime.meta.events.register(
            "provide-client-params.bedrock-runtime.Converse",
            lambda params, **_: self.converse_requests.append(params),
        )
        self.embed_requests: list[dict[str, Any]] = []  # InvokeModel bodies, decoded
        runtime.meta.events.register(
            "provide-client-params.bedrock-runtime.InvokeModel",
            lambda params, **_: self.embed_requests.append(json.loads(params["body"])),
        )

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
        self.paths.append(request.url.path)
        response = self._responses.popleft()
        return response if isinstance(response, httpx2.Response) else response(request)

    def reply(
        self,
        text: str = "ok",
        *,
        model: str = "anthropic.claude-opus-5",
        stop_reason: str = "end_turn",
        request_id: str = "req_stub",
        content: list[dict[str, Any]] | None = None,
        **usage: Any,
    ) -> None:
        """Queue a Messages response. `content` replaces the single text block `text` makes:
        thinking blocks and tool calls, as the API would return them."""
        body = _message(text, model=model, stop_reason=stop_reason, content=content, usage=usage)
        self._responses.append(httpx2.Response(200, json=body, headers={"request-id": request_id}))

    def stream_reply(
        self,
        text: str = "ok",
        *,
        model: str = "anthropic.claude-opus-5",
        stop_reason: str = "end_turn",
        request_id: str = "req_stub",
        content: list[dict[str, Any]] | None = None,
        break_after: int | None = None,
        **usage: Any,
    ) -> None:
        """Queue the stream of the message `reply` would queue with the same arguments, event by
        event: text a word a delta, a tool call's input as one JSON delta. `message_start` carries
        the input and cache tokens, the closing `message_delta` the output tokens, as the API
        sends them. `break_after=n` sends the first n events, then an error, as an endpoint
        does when a stream fails partway."""
        body = _message(text, model=model, stop_reason=stop_reason, content=content, usage=usage)
        events = _stream_events(body)
        if break_after is not None:
            events = events[:break_after]

        def respond(request: httpx2.Request) -> httpx2.Response:
            if request.url.path.endswith("/invoke-with-response-stream"):
                frames = [_chunk_frame(event) for event in events]
                if break_after is not None:
                    frames.append(_exception_frame("modelStreamErrorException"))
                media_type = "application/vnd.amazon.eventstream"
            else:
                frames = [_sse(event["type"], event) for event in events]
                if break_after is not None:
                    error = {"type": "overloaded_error", "message": "stub"}
                    frames.append(_sse("error", {"type": "error", "error": error}))
                media_type = "text/event-stream"
            headers = {"request-id": request_id, "content-type": media_type}
            return httpx2.Response(200, content=b"".join(frames), headers=headers)

        self._responses.append(respond)

    def fail(self, status: int = 400, *, request_id: str = "req_err") -> None:
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": "stub"}}
        self._responses.append(
            httpx2.Response(status, json=body, headers={"request-id": request_id})
        )

    def tool_reply(
        self,
        name: str,
        tool_input: dict[str, Any] | None,
        *,
        tool_use_id: str = "tooluse_stub",
        request_id: str = "aws-req-stub",
        input_tokens: int = 0,
        output_tokens: int = 0,
        stop_reason: str | None = None,
        more_calls: Sequence[dict[str, Any]] = (),
    ) -> None:
        """Queue a Converse response that calls tool `name` with `tool_input`. `None` queues a
        plain-text answer instead: the model ignoring a forced tool. `stop_reason` overrides
        the one that follows from that, e.g. `max_tokens` for an answer cut short. `more_calls`
        are further calls of the same tool in the same reply, with ids `<tool_use_id>_2`, …"""
        if tool_input is None:
            content: list[dict[str, Any]] = [{"text": "stub"}]
        else:
            content = [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input}}]
            content += [
                {"toolUse": {"toolUseId": f"{tool_use_id}_{n}", "name": name, "input": more}}
                for n, more in enumerate(more_calls, start=2)
            ]
        self.converse_stub.add_response(
            "converse",
            {
                "output": {"message": {"role": "assistant", "content": content}},
                "stopReason": stop_reason or ("end_turn" if tool_input is None else "tool_use"),
                "usage": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                },
                "metrics": {"latencyMs": 10},
                "ResponseMetadata": {"RequestId": request_id},
            },
        )

    def embed_reply(
        self,
        body: dict[str, Any],
        *,
        input_tokens: int | None = None,
        request_id: str = "aws-embed-stub",
    ) -> None:
        """Queue an InvokeModel response carrying `body`, the embedding model's own JSON.
        `input_tokens` goes in Bedrock's token-count header; None leaves the header out."""
        raw = json.dumps(body).encode()
        headers = {} if input_tokens is None else {INPUT_TOKENS_HEADER: str(input_tokens)}
        self.converse_stub.add_response(
            "invoke_model",
            {
                "body": StreamingBody(io.BytesIO(raw), len(raw)),
                "contentType": "application/json",
                "ResponseMetadata": {"RequestId": request_id, "HTTPHeaders": headers},
            },
        )


def _message(
    text: str,
    *,
    model: str,
    stop_reason: str,
    content: list[dict[str, Any]] | None,
    usage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": "msg_stub",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content if content is not None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, **usage},
    }


def _stream_events(message: dict[str, Any]) -> list[dict[str, Any]]:
    """`message` as the Messages streaming events that add up to it."""
    usage = message["usage"]
    start = {**message, "content": [], "stop_reason": None, "usage": {**usage, "output_tokens": 0}}
    events: list[dict[str, Any]] = [{"type": "message_start", "message": start}]
    for index, block in enumerate(message["content"]):
        match block["type"]:
            case "text":
                opening = {"type": "text", "text": ""}
                deltas = [
                    {"type": "text_delta", "text": word}
                    for word in re.findall(r"\S+\s*", block["text"])
                ]
            case "thinking":
                opening = {"type": "thinking", "thinking": "", "signature": ""}
                deltas = [
                    {"type": "thinking_delta", "thinking": block["thinking"]},
                    {"type": "signature_delta", "signature": block["signature"]},
                ]
            case "tool_use":
                opening = {**block, "input": {}}
                deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
            case other:
                raise ValueError(f"the stub cannot stream a {other} block")
        events.append({"type": "content_block_start", "index": index, "content_block": opening})
        events += [{"type": "content_block_delta", "index": index, "delta": d} for d in deltas]
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": usage["output_tokens"]},
        }
    )
    events.append({"type": "message_stop"})
    return events


def _sse(name: str, data: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


def _chunk_frame(event: dict[str, Any]) -> bytes:
    """One Messages event as InvokeModelWithResponseStream sends it: a `chunk` frame whose JSON
    payload carries the event's bytes, base64-encoded."""
    payload = {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}
    headers = {
        ":event-type": "chunk",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    return _frame(headers, json.dumps(payload).encode())


def _exception_frame(exception_type: str) -> bytes:
    headers = {
        ":exception-type": exception_type,
        ":content-type": "application/json",
        ":message-type": "exception",
    }
    return _frame(headers, json.dumps({"message": "stub"}).encode())


def _frame(headers: dict[str, str], payload: bytes) -> bytes:
    """An AWS event-stream message: prelude (total and header lengths, their CRC32), headers
    (all strings here), payload, and the CRC32 of everything before it."""
    encoded = b"".join(
        bytes([len(name)])
        + name.encode()
        + STRING_HEADER
        + struct.pack(">H", len(value))
        + value.encode()
        for name, value in headers.items()
    )
    lengths = struct.pack(">II", 12 + len(encoded) + len(payload) + 4, len(encoded))
    message = lengths + struct.pack(">I", binascii.crc32(lengths)) + encoded + payload
    return message + struct.pack(">I", binascii.crc32(message))
