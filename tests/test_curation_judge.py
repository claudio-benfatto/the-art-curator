"""`llm/curation_judge.py`: the curation judge's forced tool, its validation and its one retry.

No model is called: `StubLlm` runs the real `LlmClient` over a botocore `Stubber`, which also
validates the Converse parameters against the service model. Every conversation here is invented.
"""

import asyncio
import json

import pytest

from art_curator.llm import curation_judge
from art_curator.llm.curation_judge import (
    TOOL,
    CurationJudgeError,
    Exchange,
    ToolUse,
    Verdict,
)
from tests.llm_stub import StubLlm

OPUS = "global.anthropic.claude-opus-4-6-v1"
PERSONA = "Lives near Poblenou; likes video art."
CONTEXT = "<context>\nToday is Wednesday 2026-11-04.\n</context>"
SEARCH = ToolUse(
    "search_events",
    {"date_from": "2026-11-07", "date_to": "2026-11-07", "near": "Poblenou"},
    '{"total":1,"events":[{"id":"x:1","title":"Prova U","end":"2026-11-30"}]}',
    False,
)
WALK = "It is a five-minute walk."
FIRST = Exchange("Saturday?", (SEARCH,), f"See Prova U at Galeria Exemple. {WALK}")
SECOND = Exchange("Until when?", (), "Prova U runs until 30 November.")


def verdict(number: int, taste: str = "fits", claims: tuple[str, ...] = ()) -> dict:
    return {
        "exchange": number,
        "taste": taste,
        "reason": "Video, nearby.",
        "unsupported_claims": list(claims),
    }


def _judge(stub: StubLlm, exchanges: list[Exchange]) -> list[Verdict]:
    return asyncio.run(
        curation_judge.judge_conversation(
            stub.client, OPUS, persona=PERSONA, context=CONTEXT, exchanges=exchanges
        )
    )


def _prompt(request: dict) -> str:
    blocks = [*request["system"], *(b for m in request["messages"] for b in m["content"])]
    return "\n".join(b["text"] for b in blocks if "text" in b)


def _retry_feedback(stub: StubLlm) -> str:
    result = stub.converse_requests[1]["messages"][2]["content"][0]["toolResult"]
    return result["content"][0]["text"]


def test_each_exchange_gets_its_verdict_in_order():
    stub = StubLlm()
    stub.tool_reply(TOOL, {"verdicts": [verdict(2), verdict(1, "partly", (WALK,))]})

    got = _judge(stub, [FIRST, SECOND])

    assert got == [Verdict("partly", "Video, nearby.", (WALK,)), Verdict("fits", "Video, nearby.")]
    [row] = stub.calls
    assert (row.purpose, row.model) == ("judge", OPUS)
    request = stub.converse_requests[0]
    assert request["inferenceConfig"]["temperature"] == 0
    assert request["toolConfig"]["toolChoice"] == {"tool": {"name": TOOL}}


def test_the_prompt_holds_the_person_the_context_every_tool_call_and_every_reply():
    stub = StubLlm()
    stub.tool_reply(TOOL, {"verdicts": [verdict(1), verdict(2)]})

    _judge(stub, [FIRST, SECOND])

    prompt = _prompt(stub.converse_requests[0])
    assert PERSONA in prompt and CONTEXT in prompt
    assert f"Tool call: search_events {json.dumps(SEARCH.input)}" in prompt
    assert f"Result: {SEARCH.result}" in prompt
    assert "## Exchange 2" in prompt and SECOND.reply in prompt
    assert "word for word" in prompt and "not instructions" in prompt


def test_no_exchanges_no_call():
    stub = StubLlm()
    assert _judge(stub, []) == []
    assert stub.calls == []


@pytest.mark.parametrize(
    ("first", "reason"),
    [
        ([verdict(1, claims=("a walk along the beach",))], "claim 1 is not copied from the reply"),
        ([verdict(1), verdict(1)], "exchanges judged more than once: [1]"),
        ([], "exchanges not judged: [1]"),
        ([verdict(3)], "no exchange numbered 3"),
        ([{**verdict(1), "taste": "great"}], "taste"),
    ],
)
def test_a_bad_answer_is_retried_with_the_reason_and_can_succeed(first, reason):
    stub = StubLlm()
    stub.tool_reply(TOOL, {"verdicts": first})
    stub.tool_reply(TOOL, {"verdicts": [verdict(1)]})

    assert _judge(stub, [FIRST]) == [Verdict("fits", "Video, nearby.")]
    feedback = _retry_feedback(stub)
    assert reason in feedback and "a walk along the beach" not in feedback


def test_an_answer_invalid_twice_fails_the_conversation():
    stub = StubLlm()
    stub.tool_reply(TOOL, {"verdicts": []})
    stub.tool_reply(TOOL, {"verdicts": []})

    with pytest.raises(CurationJudgeError, match="after one retry"):
        _judge(stub, [FIRST])
    assert len(stub.calls) == 2


def test_an_answer_split_over_two_calls_is_read_as_one():
    stub = StubLlm()
    stub.tool_reply(TOOL, {"verdicts": [verdict(1)]}, more_calls=[{"verdicts": [verdict(2)]}])

    assert len(_judge(stub, [FIRST, SECOND])) == 2


def test_an_answer_cut_off_fails_without_a_retry():
    stub = StubLlm()
    stub.tool_reply(TOOL, {"verdicts": []}, stop_reason="max_tokens")

    with pytest.raises(CurationJudgeError, match="cut off"):
        _judge(stub, [FIRST])
    assert len(stub.calls) == 1


def test_a_tool_error_reaches_the_judge_as_an_error():
    failed = ToolUse("search_events", {"near": "Atlantis"}, "No place is called 'Atlantis'.", True)
    stub = StubLlm()
    stub.tool_reply(TOOL, {"verdicts": [verdict(1, "none")]})

    _judge(stub, [Exchange("Atlantis?", (failed,), "Where is that?")])

    assert "Error: No place is called 'Atlantis'." in _prompt(stub.converse_requests[0])
    assert failed.parsed() is None and SEARCH.parsed()["total"] == 1
