"""The curator's turn: a manual tool loop, capped at MAX_TOOL_ROUNDS (CLAUDE.md § 7).

Each call sends the whole conversation with one cache breakpoint on its last block
(`cached_tail`, CLAUDE.md § 6), adaptive thinking and a fixed effort. After the last permitted
round, one more call with `tool_choice: none` makes the model answer with what it has, so the cap
bounds cost without leaving the user unanswered.

History is append-only and replayed exactly as the API returned it, thinking blocks and their
signatures included: Opus 4.5+ keeps earlier thinking in context, and any edit to an earlier block
breaks the cache from that block on. A turn that fails leaves the conversation as it was.

`iteration_count` (tool rounds) goes on the turn's span and the returned `Turn`; it is the
evidence for whether the loop is needed at all.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from anthropic.types import Message

from art_curator.agent.prompt import SYSTEM_PROMPT
from art_curator.agent.tools import TOOL_DEFINITIONS, ToolResult
from art_curator.llm.client import LlmClient, cached_tail, static_prefix
from art_curator.obs import telemetry

MAX_TOOL_ROUNDS = 4
# Room for adaptive thinking plus a reply, and under the SDK's limit for a call that is not
# streamed (it refuses a non-streaming request it expects to run past 10 minutes).
MAX_TOKENS = 16000
CAPPED = "Not run: this turn used all its tool calls. Answer with what you have."

RunTool = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


@dataclass
class Conversation:
    """One conversation, in memory: its opening context and every message as sent."""

    context: str  # frozen at the start: the first block of the first user turn
    messages: list[dict[str, Any]] = field(default_factory=list)

    def with_user_text(self, text: str) -> list[dict[str, Any]]:
        """The messages plus `text` as the user's next turn. Nothing earlier changes: a turn cut off
        after its tool results already ends with a user message, and `text` joins it."""
        block = {"type": "text", "text": text}
        if not self.messages:
            return [{"role": "user", "content": [{"type": "text", "text": self.context}, block]}]
        last = self.messages[-1]
        if last["role"] == "user":
            return [*self.messages[:-1], {"role": "user", "content": [*last["content"], block]}]
        return [*self.messages, {"role": "user", "content": [block]}]


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, Any]
    is_error: bool


@dataclass(frozen=True)
class Turn:
    reply: str  # the text of the last response; empty when it had none (a refusal, say)
    iteration_count: int  # tool rounds run, at most MAX_TOOL_ROUNDS
    stop_reason: str | None  # of the last response
    trace_id: str | None  # shared by every llm_calls row of the turn
    tool_calls: list[ToolCall]


class Curator:
    def __init__(self, llm: LlmClient, run_tool: RunTool, *, model: str, effort: str) -> None:
        self._llm = llm
        self._run_tool = run_tool
        self._model = model
        self._effort = effort

    async def turn(self, conversation: Conversation, text: str) -> Turn:
        """Answer `text`, calling tools as the model asks, and append the exchange to
        `conversation`. If any call fails the exception propagates and nothing is appended."""
        messages = conversation.with_user_text(text)
        tool_calls: list[ToolCall] = []
        rounds = 0
        with self._llm.tracer.start_as_current_span("chat.turn") as span:
            trace_id, _ = telemetry.span_ids(span)
            while True:
                final = rounds == MAX_TOOL_ROUNDS
                response = await self._call(messages, final=final)
                messages.append({"role": "assistant", "content": _replayable(response)})
                calls = [block for block in response.content if block.type == "tool_use"]
                if not calls:
                    break
                # Every call is answered, or the next request is refused. A call made despite
                # `tool_choice: none` is answered without running, so the cap holds.
                results = []
                for call in calls:
                    result = ToolResult(CAPPED, True) if final else await self._run(call)
                    tool_calls.append(ToolCall(call.name, call.input, result.is_error))
                    results.append(result.block(call.id))
                messages.append({"role": "user", "content": results})
                if final:
                    break
                rounds += 1
                if response.stop_reason != "tool_use":  # cut off (max_tokens) after the calls
                    break
            span.set_attributes(
                {
                    "art_curator.iteration_count": rounds,
                    "art_curator.stop_reason": response.stop_reason or "",
                }
            )
        conversation.messages = messages
        reply = "\n\n".join(block.text for block in response.content if block.type == "text")
        return Turn(reply, rounds, response.stop_reason, trace_id, tool_calls)

    async def _call(self, messages: list[dict[str, Any]], *, final: bool) -> Message:
        params: dict[str, Any] = {
            "system": static_prefix(SYSTEM_PROMPT),
            "tools": TOOL_DEFINITIONS,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort},
            "max_tokens": MAX_TOKENS,
            "messages": cached_tail(messages),
        }
        if final:
            params["tool_choice"] = {"type": "none"}
        return await self._llm.create_message(purpose="chat", model=self._model, **params)

    async def _run(self, call: Any) -> ToolResult:
        with self._llm.tracer.start_as_current_span(f"tool.{call.name}") as span:
            result = await self._run_tool(call.name, call.input)
            span.set_attribute("art_curator.tool.is_error", result.is_error)
            return result


def _replayable(response: Message) -> list[dict[str, Any]]:
    """The response's content blocks as request params, as the API returned them: a thinking
    block keeps its text and signature, a tool call its id and input."""
    return [block.to_dict(mode="json", exclude_none=True) for block in response.content]
