"""`llm/extract.py`: the forced tool and its one retry, link numbers, and date resolution.

No model is called: `StubLlm` runs the real `LlmClient` over a botocore `Stubber`, which also
validates the Converse parameters against the service model. Every page here is invented.

The verbatim guard is tested in `tests/test_no_verbatim.py`, with the rest of CLAUDE.md § 1.
"""

import asyncio
import hashlib
import json
from datetime import date, time
from types import SimpleNamespace

import pytest

from art_curator.llm import extract
from art_curator.llm.extract import (
    TOOL,
    ArtEvent,
    DetailDraft,
    Exhibition,
    ExtractionError,
    ListingDraft,
    WrittenDate,
    resolve_dates,
)
from art_curator.llm.tools import flat_schema
from tests.llm_stub import StubLlm

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
VENUE = "Galeria Exemple"
URL = "https://www.exemple.cat/ca/exposicions"
FETCHED = date(2026, 10, 10)
PAGE = "Exposicions Prova U Maria Exemple 12.09 – 30.11 Inauguració dijous 12 de setembre, 19 h"
LINKS = [
    SimpleNamespace(number=1, url="https://www.exemple.cat/ca/exposicions/prova-u", text="Prova U"),
    SimpleNamespace(number=2, url="https://www.exemple.cat/ca/avis-legal", text="Avís legal"),
]


def exhibition_input(**changes) -> dict:
    return {
        "title": "Prova U",
        "artists": ["Maria Exemple"],
        "start": {"day": 12, "month": 9, "year": None},
        "end": {"day": 30, "month": 11, "year": None},
        "at_venue": True,
        "link": None,
        "summary_en": "Small paintings on reclaimed wood recall a village lost to a reservoir.",
        "themes": ["memory", "landscape"],
        "media": ["painting"],
        "confidence": "high",
        **changes,
    }


def event_input(**changes) -> dict:
    return {
        **exhibition_input(
            title="Inauguració: Prova U", artists=[], end=None, summary_en=None, themes=[], media=[]
        ),
        "kind": "opening",
        "start_time": "19:00",
        "exhibition_title": "Prova U",
        **changes,
    }


def listed(item: dict) -> dict:
    """`item` as a listing records it: facts only."""
    return {k: v for k, v in item.items() if k not in {"summary_en", "themes", "media"}}


def answer(exhibitions: list[dict] | None = None, events: list[dict] | None = None) -> dict:
    return {
        "exhibitions": [exhibition_input()] if exhibitions is None else exhibitions,
        "events": events or [],
    }


def run_extract(
    stub: StubLlm, *, text: str = PAGE, links=(), fetched_on: date = FETCHED, listing=False
):
    """A show's own page unless `listing`: most tests here are about what a description holds."""
    return asyncio.run(
        extract.extract_page(
            stub.client,
            HAIKU,
            venue_name=VENUE,
            url=URL,
            text=text,
            fetched_on=fetched_on,
            listing=listing,
            links=links,
        )
    )


def _prompt(request: dict) -> str:
    """Everything the model was sent: system prompt plus every text block."""
    blocks = [*request["system"], *(b for m in request["messages"] for b in m["content"])]
    return "\n".join(b["text"] for b in blocks if "text" in b)


def _feedback(stub: StubLlm) -> dict:
    """The failed tool result the retry was sent."""
    return stub.converse_requests[1]["messages"][2]["content"][0]["toolResult"]


# --- the call -----------------------------------------------------------------------------------


def test_extracts_exhibitions_and_events():
    stub = StubLlm()
    stub.tool_reply(TOOL, answer(events=[event_input()]), input_tokens=3300, output_tokens=600)

    result = run_extract(stub)

    assert result.exhibitions == (
        Exhibition(
            title="Prova U",
            artists=("Maria Exemple",),
            start_date=date(2026, 9, 12),
            end_date=date(2026, 11, 30),
            at_venue=True,
            detail_url=None,
            summary_en="Small paintings on reclaimed wood recall a village lost to a reservoir.",
            themes=("memory", "landscape"),
            media=("painting",),
            confidence="high",
        ),
    )
    assert result.events == (
        ArtEvent(
            title="Inauguració: Prova U",
            artists=(),
            start_date=date(2026, 9, 12),
            end_date=None,
            at_venue=True,
            detail_url=None,
            summary_en=None,
            themes=(),
            media=(),
            confidence="high",
            kind="opening",
            start_time=time(19, 0),
            exhibition_title="Prova U",
        ),
    )
    assert result.summaries_dropped == 0

    [request] = stub.converse_requests
    assert request["system"] == [{"text": extract.DETAIL_SYSTEM}]
    assert request["toolConfig"] == extract.DETAIL_TOOL_CONFIG
    assert request["toolConfig"]["toolChoice"] == {"tool": {"name": TOOL}}
    assert request["inferenceConfig"] == {"maxTokens": extract.MAX_OUTPUT_TOKENS, "temperature": 0}
    assert "additionalModelRequestFields" not in request  # thinking stays off (CLAUDE.md § 5)
    assert f"Venue: {VENUE}\nURL: {URL}\n\nPage text:\n<page>\n{PAGE}\n</page>" in _prompt(request)
    [row] = stub.calls
    assert (row.purpose, row.model, row.input_tokens, row.output_tokens) == (
        "extract",
        HAIKU,
        3300,
        600,
    )


def test_a_listing_records_facts_only():
    """A listing is mostly archive: describing every show on it is what cut four pilot listings
    off at the token limit (2026-10-29). The description comes from the show's own page."""
    stub = StubLlm()
    stub.tool_reply(TOOL, answer([listed(exhibition_input(link=1))], [listed(event_input())]))

    result = run_extract(stub, listing=True, links=LINKS)

    [exhibition], [event] = result.exhibitions, result.events
    assert (exhibition.title, exhibition.start_date) == ("Prova U", date(2026, 9, 12))
    assert exhibition.detail_url == "https://www.exemple.cat/ca/exposicions/prova-u"
    assert (exhibition.summary_en, exhibition.themes, exhibition.media) == (None, (), ())
    assert (event.kind, event.start_time, event.summary_en) == ("opening", time(19, 0), None)
    [request] = stub.converse_requests
    assert request["system"] == [{"text": extract.LISTING_SYSTEM}]
    assert request["toolConfig"] == extract.LISTING_TOOL_CONFIG
    assert request["inferenceConfig"] == {
        "maxTokens": extract.MAX_LISTING_OUTPUT_TOKENS,
        "temperature": 0,
    }


def test_a_listing_has_nowhere_to_put_a_description():
    stub = StubLlm()
    stub.tool_reply(TOOL, answer())
    stub.tool_reply(TOOL, answer())

    with pytest.raises(ExtractionError, match="exhibitions.0.summary_en: Extra inputs"):
        run_extract(stub, listing=True)


def test_a_listing_may_name_more_shows_than_a_show_page():
    """Pigment and RocioSantaCruz list 43 and 52 shows; 40 was a show page's cap."""
    many = [listed(exhibition_input(title=f"Prova {n}")) for n in range(extract.MAX_ITEMS + 10)]
    stub = StubLlm()
    stub.tool_reply(TOOL, answer(many))

    assert len(run_extract(stub, listing=True).exhibitions) == extract.MAX_ITEMS + 10


def test_a_listing_longer_than_its_cap_fails():
    too_many = [listed(exhibition_input())] * (extract.MAX_LISTED_ITEMS + 1)
    stub = StubLlm()
    stub.tool_reply(TOOL, answer(too_many))
    stub.tool_reply(TOOL, answer(too_many))

    with pytest.raises(ExtractionError, match="at most 200 items"):
        run_extract(stub, listing=True)


def test_a_page_that_announces_nothing_is_a_valid_answer():
    stub = StubLlm()
    stub.tool_reply(TOOL, answer(exhibitions=[]))

    result = run_extract(stub)

    assert (result.exhibitions, result.events) == ((), ())
    assert len(stub.calls) == 1


def test_bad_answer_is_retried_with_the_reason_and_can_succeed():
    stub = StubLlm()
    stub.tool_reply(TOOL, answer([exhibition_input(confidence="0.9")]), tool_use_id="use_1")
    stub.tool_reply(TOOL, answer())

    result = run_extract(stub)

    assert [e.title for e in result.exhibitions] == ["Prova U"]
    assert len(stub.calls) == 2  # both attempts are recorded spend
    assert [m["role"] for m in stub.converse_requests[1]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    feedback = _feedback(stub)
    assert (feedback["toolUseId"], feedback["status"]) == ("use_1", "error")
    assert "exhibitions.0.confidence" in feedback["content"][0]["text"]


@pytest.mark.parametrize(
    "tool_input",
    [
        {"exhibitions": "none"},  # a string that is not a JSON list
        {"exhibitions": exhibition_input()},  # one item, not a list of one
        {"exhibitions": json.dumps({"exhibitions": [exhibition_input()]})},  # the whole answer
        answer([{"title": "Prova U"}]),
        answer([exhibition_input(title="")]),
        answer([exhibition_input(title="t" * (extract.TITLE_CHARS + 1))]),
        answer([exhibition_input(summary_en="s" * (extract.SUMMARY_CHARS + 1))]),
        answer([exhibition_input(themes=["t"] * (extract.MAX_THEMES + 1))]),
        answer([exhibition_input(media=["oil on canvas"])]),  # not one of the allowed values
        answer([exhibition_input(start={"day": 12, "month": 13, "year": None})]),
        answer([exhibition_input(start="12.09")]),  # dates are transcribed as numbers
        answer([exhibition_input(start={"day": 12, "month": 9})]),  # the year is null, not absent
        answer([exhibition_input(source_text="The page says: ...")]),  # nowhere for a quote to go
        answer([exhibition_input()] * (extract.MAX_ITEMS + 1)),
        answer(events=[event_input(start_time="7pm")]),
        answer(events=[event_input(kind="vernissage")]),
    ],
)
def test_answer_invalid_twice_fails_the_page(tool_input):
    stub = StubLlm()
    stub.tool_reply(TOOL, tool_input)
    stub.tool_reply(TOOL, tool_input)

    with pytest.raises(ExtractionError, match="no usable answer after one retry"):
        run_extract(stub)
    assert len(stub.calls) == 2


def test_an_answer_split_over_two_calls_is_read_as_one():
    """Miró, 2026-10-30: exhibitions in one call, events in the next, in the same reply."""
    stub = StubLlm()
    stub.tool_reply(
        TOOL,
        {"exhibitions": [exhibition_input()]},
        more_calls=[{"events": [event_input()]}, {"events": [event_input(title="Xerrada")]}],
    )

    result = run_extract(stub)

    assert [e.title for e in result.exhibitions] == ["Prova U"]
    assert [e.title for e in result.events] == ["Inauguració: Prova U", "Xerrada"]
    assert len(stub.calls) == 1


def test_a_retry_answers_every_call_in_the_reply():
    """Converse refuses a conversation that leaves any call unanswered (ValidationException,
    pilot run 2026-10-30)."""
    stub = StubLlm()
    stub.tool_reply(
        TOOL,
        {"exhibitions": [exhibition_input(confidence="0.9")]},
        tool_use_id="use_1",
        more_calls=[{"events": [event_input()]}],
    )
    stub.tool_reply(TOOL, answer())

    run_extract(stub)

    results = [block["toolResult"] for block in stub.converse_requests[1]["messages"][2]["content"]]
    assert [(r["toolUseId"], r["status"]) for r in results] == [
        ("use_1", "error"),
        ("use_1_2", "error"),
    ]
    assert all("exhibitions.0.confidence" in r["content"][0]["text"] for r in results)


@pytest.mark.parametrize("listing", [False, True])
def test_a_list_sent_as_json_text_or_left_out_is_read_as_meant(listing):
    """Miró, 2026-10-30: `exhibitions` as a JSON string and no `events`, twice at temperature 0."""
    item = listed(exhibition_input()) if listing else exhibition_input()
    stub = StubLlm()
    stub.tool_reply(TOOL, {"exhibitions": json.dumps([item])})

    result = run_extract(stub, listing=listing)

    assert ([e.title for e in result.exhibitions], result.events) == (["Prova U"], ())
    assert len(stub.calls) == 1


def test_an_empty_answer_is_a_page_that_announces_nothing():
    stub = StubLlm()
    stub.tool_reply(TOOL, {})

    result = run_extract(stub)

    assert (result.exhibitions, result.events) == ((), ())
    assert len(stub.calls) == 1


@pytest.mark.parametrize("listing", [False, True])
def test_a_year_without_a_month_is_no_date(listing):
    """What the prompt asks for a year alone, applied when the model writes it as a date."""
    year_alone = {"day": None, "month": None, "year": 2027}
    item = exhibition_input(start={"day": 12, "month": 9, "year": 2026}, end=year_alone)
    stub = StubLlm()
    stub.tool_reply(TOOL, answer([listed(item) if listing else item]))

    [exhibition] = run_extract(stub, listing=listing).exhibitions

    assert (exhibition.start_date, exhibition.end_date) == (date(2026, 9, 12), None)
    assert len(stub.calls) == 1


def test_validation_feedback_does_not_echo_the_answer():
    """The retry names the field and the rule, not what the model wrote there."""
    stub = StubLlm()
    stub.tool_reply(TOOL, answer([exhibition_input(source_text="COPIED SPAN")]))
    stub.tool_reply(TOOL, answer())

    run_extract(stub)

    text = _feedback(stub)["content"][0]["text"]
    assert "source_text" in text
    assert "COPIED SPAN" not in text


def test_a_model_that_ignores_the_tool_is_retried_then_fails():
    stub = StubLlm()
    stub.tool_reply(TOOL, None)
    stub.tool_reply(TOOL, None)

    with pytest.raises(ExtractionError, match="did not call the tool"):
        run_extract(stub)
    # Nothing to answer with a tool result, so the retry repeats the original request.
    assert len(stub.converse_requests[1]["messages"]) == 1


@pytest.mark.parametrize(
    ("listing", "limit"),
    [(False, extract.MAX_OUTPUT_TOKENS), (True, extract.MAX_LISTING_OUTPUT_TOKENS)],
)
def test_an_answer_cut_off_by_the_token_limit_fails_without_a_retry(listing, limit):
    """At temperature 0 the second answer would be cut at the same place, and the error does not
    claim a retry that was never made."""
    stub = StubLlm()
    stub.tool_reply(TOOL, {"exhibitions": [exhibition_input()]}, stop_reason="max_tokens")

    with pytest.raises(ExtractionError) as failed:
        run_extract(stub, listing=listing)
    assert str(failed.value) == (
        f"record_programme: no usable answer (the answer was cut off at {limit} output tokens)"
    )
    assert len(stub.calls) == 1


def test_whitespace_and_repeated_tags_are_tidied():
    stub = StubLlm()
    stub.tool_reply(
        TOOL,
        answer(
            [
                exhibition_input(
                    title="  Prova U\n",
                    artists=[" Maria Exemple "],
                    themes=["Memory", "memory", "Landscape"],
                    media=["painting", "video", "painting"],
                    summary_en="   ",
                )
            ]
        ),
    )

    [exhibition] = run_extract(stub).exhibitions

    assert (exhibition.title, exhibition.artists) == ("Prova U", ("Maria Exemple",))
    assert exhibition.themes == ("memory", "landscape")
    assert exhibition.media == ("painting", "video")
    assert exhibition.summary_en is None


# --- links --------------------------------------------------------------------------------------


def test_link_number_becomes_the_detail_url():
    stub = StubLlm()
    stub.tool_reply(TOOL, answer([exhibition_input(link=1)], [event_input(link=None)]))

    result = run_extract(stub, links=LINKS)

    assert result.exhibitions[0].detail_url == "https://www.exemple.cat/ca/exposicions/prova-u"
    assert result.events[0].detail_url is None
    prompt = _prompt(stub.converse_requests[0])
    assert "[1] Prova U — https://www.exemple.cat/ca/exposicions/prova-u" in prompt
    assert prompt.index("Links on the page:") < prompt.index("Page text:")


def test_invented_link_number_is_rejected():
    """The model answers by number so that it cannot invent a URL."""
    stub = StubLlm()
    stub.tool_reply(TOOL, answer([exhibition_input(link=7)]))
    stub.tool_reply(TOOL, answer([exhibition_input(link=0)]))

    with pytest.raises(ExtractionError, match="no link numbered 0"):
        run_extract(stub, links=LINKS)
    assert "no link numbered 7; the list runs from 1 to 2" in _feedback(stub)["content"][0]["text"]


def test_a_link_is_refused_when_none_were_listed():
    stub = StubLlm()
    stub.tool_reply(TOOL, answer([exhibition_input(link=1)]))
    stub.tool_reply(TOOL, answer([exhibition_input(link=None)]))

    [exhibition] = run_extract(stub).exhibitions

    assert exhibition.detail_url is None
    assert "no links were listed" in _feedback(stub)["content"][0]["text"]
    assert "Links on the page" not in _prompt(stub.converse_requests[0])


def test_link_text_and_url_are_capped_in_the_prompt():
    long = SimpleNamespace(number=1, url="https://www.exemple.cat/" + "u" * 500, text="t" * 500)
    prompt = extract.extract_prompt(VENUE, URL, PAGE, [long])

    assert "t" * extract.LINK_TEXT_CHARS in prompt
    assert "t" * (extract.LINK_TEXT_CHARS + 1) not in prompt
    assert "u" * extract.LINK_URL_CHARS not in prompt


# --- dates --------------------------------------------------------------------------------------


def written(day: int | None, month: int, year: int | None = None) -> WrittenDate:
    return WrittenDate(day=day, month=month, year=year)


@pytest.mark.parametrize(
    ("start", "end", "fetched_on", "expected"),
    [
        # Both years written: nothing to infer.
        ((12, 9, 2026), (30, 11, 2026), FETCHED, ("2026-09-12", "2026-11-30")),
        ((12, 9, 2026), (12, 9, 2026), FETCHED, ("2026-09-12", "2026-09-12")),
        ((20, 11, 2025), (15, 2, 2026), FETCHED, ("2025-11-20", "2026-02-15")),
        # No year at all: the run nearest to the day the page was fetched.
        ((12, 9), (30, 11), FETCHED, ("2026-09-12", "2026-11-30")),  # on now
        ((15, 1), (20, 3), FETCHED, ("2027-01-15", "2027-03-20")),  # coming, not nine months gone
        ((1, 9), (30, 9), FETCHED, ("2026-09-01", "2026-09-30")),  # closed ten days ago
        ((20, 11), (15, 2), FETCHED, ("2026-11-20", "2027-02-15")),  # runs over New Year
        ((20, 11), (15, 2), date(2026, 1, 10), ("2025-11-20", "2026-02-15")),  # read in January
        ((1, 12), (31, 12), date(2026, 1, 2), ("2025-12-01", "2025-12-31")),
        # The year on one end only: the nearest year that keeps end >= start.
        ((12, 9), (30, 11, 2026), FETCHED, ("2026-09-12", "2026-11-30")),
        ((12, 9), (30, 11, 2026), date(2026, 3, 1), ("2026-09-12", "2026-11-30")),
        ((20, 11), (15, 2, 2027), FETCHED, ("2026-11-20", "2027-02-15")),
        ((12, 9, 2026), (30, 11), FETCHED, ("2026-09-12", "2026-11-30")),
        ((20, 11, 2026), (15, 2), FETCHED, ("2026-11-20", "2027-02-15")),
        ((20, 11, 2024), (15, 2), FETCHED, ("2024-11-20", "2025-02-15")),  # an archive page
        # A month without a day: its first day for a start, its last for an end.
        ((None, 11, 2026), (None, 11, 2026), FETCHED, ("2026-11-01", "2026-11-30")),
        ((None, 10), (None, 12), FETCHED, ("2026-10-01", "2026-12-31")),
        ((None, 2), (None, 2), date(2028, 1, 15), ("2028-02-01", "2028-02-29")),  # leap year
        ((12, 9, 2026), (None, 1), FETCHED, ("2026-09-12", "2027-01-31")),
        # One end only.
        (None, (30, 11), FETCHED, (None, "2026-11-30")),
        (None, (15, 1), FETCHED, (None, "2027-01-15")),
        (None, (None, 11, 2026), FETCHED, (None, "2026-11-30")),
        ((12, 9), None, FETCHED, ("2026-09-12", None)),  # an opening four weeks ago
        ((20, 12), None, date(2026, 1, 5), ("2025-12-20", None)),
        ((12, 9, 2024), None, FETCHED, ("2024-09-12", None)),
        ((None, 3), None, FETCHED, ("2027-03-01", None)),
        ((29, 2), None, date(2027, 12, 1), ("2028-02-29", None)),  # the only year that has one
        (None, None, FETCHED, (None, None)),
        # Not repaired: a day that does not exist, and an end before its start.
        ((31, 6, 2026), (30, 9, 2026), FETCHED, (None, None)),
        ((1, 6), (31, 6), FETCHED, (None, None)),
        ((31, 6), None, FETCHED, (None, None)),
        ((30, 11, 2026), (12, 9, 2026), FETCHED, (None, None)),
    ],
)
def test_year_inference(start, end, fetched_on, expected):
    resolved = resolve_dates(
        written(*start) if start else None, written(*end) if end else None, fetched_on
    )
    assert tuple(day.isoformat() if day else None for day in resolved) == expected


def test_dates_are_resolved_against_the_fetch_day_not_today():
    stub = StubLlm()
    stub.tool_reply(TOOL, answer())

    [exhibition] = run_extract(stub, fetched_on=date(2031, 10, 1)).exhibitions

    assert (exhibition.start_date, exhibition.end_date) == (date(2031, 9, 12), date(2031, 11, 30))


def test_the_model_is_not_told_the_date():
    """It transcribes; it does not compute (CLAUDE.md § 8). Nothing per-request but the page."""
    assert "2031" not in extract.extract_prompt(VENUE, URL, PAGE, LINKS)
    stub = StubLlm()
    stub.tool_reply(TOOL, answer())
    run_extract(stub, fetched_on=date(2031, 10, 1))
    assert "2031" not in _prompt(stub.converse_requests[0])


# --- schema and prompt --------------------------------------------------------------------------


def _strings(node, path=""):
    """Every string-typed node of a JSON schema, with the path that leads to it."""
    if isinstance(node, dict):
        if node.get("type") == "string":
            yield path, node
        for key, value in node.items():
            yield from _strings(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _strings(value, f"{path}/{index}")


@pytest.mark.parametrize(
    ("draft", "tool_config"),
    [(ListingDraft, extract.LISTING_TOOL_CONFIG), (DetailDraft, extract.DETAIL_TOOL_CONFIG)],
)
def test_tool_schema_is_one_flat_strict_object(draft, tool_config):
    schema = flat_schema(draft)
    dumped = json.dumps(schema)

    assert "$ref" not in dumped and "$defs" not in dumped
    assert tool_config["tools"][0]["toolSpec"]["inputSchema"]["json"] == schema
    objects = [n for n in _walk(schema) if isinstance(n, dict) and n.get("type") == "object"]
    assert len(objects) == 7  # the answer, two item types, and a start and an end date in each
    assert all(n["additionalProperties"] is False for n in objects)
    # Nothing is optional: the model says null rather than leaving a field out.
    assert all(set(n["required"]) == set(n["properties"]) for n in objects)


def test_the_model_writes_an_items_own_fields_first():
    """The answer is written in schema order: title and dates before anything event-only."""
    events = flat_schema(DetailDraft)["properties"]["events"]["items"]["properties"]
    assert list(events) == [
        "title", "artists", "start", "end", "at_venue", "link", "confidence",
        "summary_en", "themes", "media", "kind", "start_time", "exhibition_title",
    ]  # fmt: skip
    listed_events = flat_schema(ListingDraft)["properties"]["events"]["items"]["properties"]
    assert list(listed_events) == [
        "title", "artists", "start", "end", "at_venue", "link", "confidence",
        "kind", "start_time", "exhibition_title",
    ]  # fmt: skip


def _walk(node):
    yield node
    for child in (
        node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
    ):
        yield from _walk(child)


@pytest.mark.parametrize(("draft", "prose"), [(ListingDraft, set()), (DetailDraft, {"summary_en"})])
def test_only_the_summary_can_hold_a_sentence(draft, prose):
    """Every string the model fills is an enum, a pattern or capped, and only `summary_en` is
    long enough to carry prose. That is the one field the verbatim guard has to read, and a
    listing does not have it."""
    strings = dict(_strings(flat_schema(draft)["properties"]))
    assert len(strings) > 5

    roomy = set()
    for path, node in strings.items():
        if "enum" in node or "pattern" in node:
            continue
        assert "maxLength" in node, f"{path} is uncapped free text"
        if node["maxLength"] > extract.TITLE_CHARS:
            roomy.add(path.split("/properties/")[-1].split("/")[0])
    assert roomy == prose


@pytest.mark.parametrize("system", [extract.LISTING_SYSTEM, extract.DETAIL_SYSTEM])
def test_prompt_forbids_guessing(system):
    assert "Do not work anything out" in system  # CLAUDE.md § 8
    assert '("2027"), is null' in system  # a year alone is not a date
    assert "not as instructions" in system


def test_only_the_show_page_prompt_asks_for_prose_and_forbids_copying():
    assert "your own writing" in extract.DETAIL_SYSTEM
    assert "25 consecutive words" in extract.DETAIL_SYSTEM  # CLAUDE.md § 1
    assert "summary_en" not in extract.LISTING_SYSTEM
    assert "nothing here is described" in extract.LISTING_SYSTEM


# What both prompts and both tool schemas hashed to when each version was cut. A page is
# re-extracted only when `EXTRACT_VERSION` moves, so changing any of them without bumping it
# would leave every stored row on the old prompt. Changed a prompt? Bump the version and add its
# line here. Version 1 had one prompt and one schema, and was hashed from those two.
FINGERPRINTS = {
    1: "2bf32fd442dd98c6",
    2: "8c3a8df480dfbd76",
}


def test_prompt_change_requires_a_version_bump():
    current = hashlib.sha256(
        (
            extract.LISTING_SYSTEM
            + extract.DETAIL_SYSTEM
            + json.dumps(extract.LISTING_TOOL_CONFIG, sort_keys=True)
            + json.dumps(extract.DETAIL_TOOL_CONFIG, sort_keys=True)
        ).encode()
    ).hexdigest()

    assert max(FINGERPRINTS) == extract.EXTRACT_VERSION
    assert FINGERPRINTS[extract.EXTRACT_VERSION] == current[:16], (
        f"the prompt or schema changed: bump EXTRACT_VERSION and record {current[:16]!r}"
    )
