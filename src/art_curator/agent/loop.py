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

`events` is the turn as it happens, for `/chat`: each tool as it starts, the reply's text as it
streams, then the `Turn`. Text is forwarded from every response, so words the model writes
before a tool call arrive too, ahead of that tool's `ToolRunning`. `Turn.reply` is the same
text, every response's, so `cli chat` and the eval read what `/chat` showed: a model that
answers alongside a tool call and then ends with nothing more has still answered. Thinking is
not forwarded. `turn` runs the same loop unstreamed and
returns the `Turn`: what `cli chat` and the eval use, and the only path open to a chat model
without the streaming grant.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from anthropic.types import Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from art_curator.agent.prompt import SYSTEM_PROMPT
from art_curator.agent.tools import TOOL_DEFINITIONS, CuratorTools, ToolResult, query_embedder
from art_curator.llm.client import LlmClient, cached_tail, static_prefix
from art_curator.obs import telemetry

MAX_TOOL_ROUNDS = 4
# Room for adaptive thinking plus a reply, and under the SDK's limit for a call that is not
# streamed (it refuses a non-streaming request it expects to run past 10 minutes). Streamed calls
# send the same, so both paths make the same request.
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
    # The text of every response of the turn, in order: what `/chat` streamed. Empty when none
    # had any (a refusal, say).
    reply: str
    iteration_count: int  # tool rounds run, at most MAX_TOOL_ROUNDS
    stop_reason: str | None  # of the last response
    trace_id: str | None  # shared by every llm_calls row of the turn
    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class ToolRunning:
    """A tool call is about to run (SSE `status`). Not sent for a call the cap leaves unrun."""

    name: str


@dataclass(frozen=True)
class TextDelta:
    """The next piece of a response's text (SSE `text`)."""

    text: str


@dataclass(frozen=True)
class TurnDone:
    """The turn is over and appended to the conversation (SSE `done`). Always the last event."""

    turn: Turn


TurnEvent = ToolRunning | TextDelta | TurnDone


class Curator:
    def __init__(
        self,
        llm: LlmClient,
        run_tool: RunTool,
        *,
        model: str,
        effort: str,
        tool_definitions: list[dict[str, Any]] = TOOL_DEFINITIONS,
    ) -> None:
        """`tool_definitions` are what the model is sent; they must match what `run_tool` runs
        (`CuratorTools.definitions`)."""
        self._llm = llm
        self._run_tool = run_tool
        self._model = model
        self._effort = effort
        self._tool_definitions = tool_definitions

    async def turn(self, conversation: Conversation, text: str) -> Turn:
        """Answer `text`, calling tools as the model asks, and append the exchange to
        `conversation`. If any call fails the exception propagates and nothing is appended."""
        events = [event async for event in self.events(conversation, text, stream=False)]
        done = events[-1]
        assert isinstance(done, TurnDone)  # always the last event
        return done.turn

    async def events(
        self, conversation: Conversation, text: str, *, stream: bool = True
    ) -> AsyncIterator[TurnEvent]:
        """`turn`, as events. With `stream`, each model call is streamed and its text yielded
        as it arrives; without, only `ToolRunning` and the final `TurnDone` are. The exchange is
        appended just before `TurnDone`: a turn that fails, or that the reader stops reading,
        leaves the conversation as it was, and its model calls are recorded all the same."""
        messages = conversation.with_user_text(text)
        texts: list[str] = []  # every response's text blocks, in order: the reply
        tool_calls: list[ToolCall] = []
        rounds = 0
        with self._llm.tracer.start_as_current_span("chat.turn") as span:
            trace_id, _ = telemetry.span_ids(span)
            while True:
                final = rounds == MAX_TOOL_ROUNDS
                params = self._params(messages, final=final)
                if stream:
                    async with self._llm.stream_message(
                        purpose="chat", model=self._model, **params
                    ) as response_stream:
                        async for event in response_stream:
                            if (
                                event.type == "content_block_delta"
                                and event.delta.type == "text_delta"
                            ):
                                yield TextDelta(event.delta.text)
                        response = await response_stream.get_final_message()
                else:
                    response = await self._llm.create_message(
                        purpose="chat", model=self._model, **params
                    )
                messages.append({"role": "assistant", "content": _replayable(response)})
                texts += [block.text for block in response.content if block.type == "text"]
                calls = [block for block in response.content if block.type == "tool_use"]
                if not calls:
                    break
                # Every call is answered, or the next request is refused. A call made despite
                # `tool_choice: none` is answered without running, so the cap holds.
                results = []
                for call in calls:
                    if final:
                        result = ToolResult(CAPPED, True)
                    else:
                        yield ToolRunning(call.name)
                        result = await self._run(call)
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
        yield TurnDone(Turn("\n\n".join(texts), rounds, response.stop_reason, trace_id, tool_calls))

    def _params(self, messages: list[dict[str, Any]], *, final: bool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "system": static_prefix(SYSTEM_PROMPT),
            "tools": self._tool_definitions,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort},
            "max_tokens": MAX_TOKENS,
            "messages": cached_tail(messages),
        }
        if final:
            params["tool_choice"] = {"type": "none"}
        return params

    async def _run(self, call: Any) -> ToolResult:
        with self._llm.tracer.start_as_current_span(f"tool.{call.name}") as span:
            result = await self._run_tool(call.name, call.input)
            span.set_attribute("art_curator.tool.is_error", result.is_error)
            return result


def curator_for(
    llm: LlmClient,
    sessions: async_sessionmaker[AsyncSession],
    user_id: int,
    *,
    model: str,
    effort: str,
    embed_model: str | None,
) -> Curator:
    """The curator over its own tools, bound to one user: what `cli chat`, `/chat` and the eval
    run. With `embed_model`, `search_events` takes a `query` ranked by that model's vectors."""
    tools = CuratorTools(sessions, user_id, query_embedder(llm, embed_model))
    return Curator(llm, tools, model=model, effort=effort, tool_definitions=tools.definitions)


def _replayable(response: Message) -> list[dict[str, Any]]:
    """The response's content blocks as request params, as the API returned them: a thinking
    block keeps its text and signature, a tool call its id and input."""
    return [block.to_dict(mode="json", exclude_none=True) for block in response.content]
