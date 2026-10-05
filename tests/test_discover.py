"""`ingest/discover.py`: link collection, the two model passes, and routing.

No model is called: `StubLlm` runs the real `LlmClient` over a botocore `Stubber`, which also
validates the Converse parameters (tool spec, forced `toolChoice`) against the service model.
The HTML is hand-written (`tests/fixtures/crawl/`), never a recorded page (CLAUDE.md § 1).
"""

import asyncio
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from art_curator.ingest import discover
from art_curator.ingest.discover import (
    PASS1_TOOL,
    PASS2_TOOL,
    DiscoveryError,
    GrafHints,
    Link,
    LinkChoice,
    PageVerdict,
    SeedStatus,
    collect_links,
    route,
    route_site,
)
from art_curator.ingest.graf import DROP_KEYS
from tests.llm_stub import StubLlm

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
HOMEPAGE = (Path(__file__).parent / "fixtures" / "crawl" / "homepage.html").read_bytes()
SITE = "https://www.exemple.cat/"
VENUE = "Galeria Exemple"

LINKS = [
    Link(1, "https://www.exemple.cat/ca/exposicions", "Exposicions", "nav"),
    Link(2, "https://www.exemple.cat/ca/agenda", "Agenda", "nav"),
    Link(3, "https://www.exemple.cat/ca/avis-legal", "Avís legal", "footer"),
]
HINTS = GrafHints(
    titles=("Prova U", "Segona mostra"), urls=("https://www.exemple.cat/ca/exposicions/prova-u",)
)


def _verdict(page_type: str, confidence: str = "high", dated_items: int = 3) -> PageVerdict:
    return PageVerdict(
        page_type=page_type, dated_items=dated_items, language="ca", confidence=confidence
    )


def _pick(stub: StubLlm, links=LINKS, hints: GrafHints = HINTS) -> list[Link]:
    return asyncio.run(
        discover.pick_candidates(
            stub.client, HAIKU, venue_name=VENUE, page_url=SITE, links=links, hints=hints
        )
    )


def _classify(stub: StubLlm, text: str = "Exposicions actuals", hints=HINTS) -> PageVerdict:
    return asyncio.run(
        discover.classify_page(
            stub.client, HAIKU, venue_name=VENUE, url=LINKS[0].url, text=text, hints=hints
        )
    )


def _prompt(request: dict) -> str:
    """Everything the model was sent: system prompt plus every text block."""
    blocks = [*request["system"], *(b for m in request["messages"] for b in m["content"])]
    return "\n".join(b["text"] for b in blocks if "text" in b)


# --- link collection ----------------------------------------------------------------------------


def test_links_are_same_site_deduplicated_and_numbered():
    links = collect_links(HOMEPAGE, SITE)

    assert [(link.number, link.url.removeprefix("https://www.exemple.cat")) for link in links] == [
        (1, "/"),
        (2, "/ca/exposicions/"),
        (3, "/ca/agenda"),
        (4, "/ca/artistes"),
        (5, "/ca/visita"),  # fragment dropped
        (6, "/en/exhibitions"),
        (7, "/ca/exposicions/prova-u"),
        (8, "/ca/exposicions/prova-u?vista=sala"),  # a query makes it a different page
        (9, "/ca/avis-legal"),
        (10, "/ca/arxiu"),
    ]
    # Gone: the in-page anchor, the PDF, the shop subdomain, Instagram, mailto:, javascript:,
    # the anchor with no href, and the two repeats (`/ca/exposicions`, `exemple.cat/ca/agenda/`).


def test_links_carry_their_region():
    regions = {link.url.rsplit("/", 1)[-1]: link.region for link in collect_links(HOMEPAGE, SITE)}

    assert regions["agenda"] == "nav"
    assert regions["prova-u"] == "main"
    assert regions["avis-legal"] == "footer"
    assert regions["arxiu"] == "nav"  # role="navigation" inside the footer: nearest wins


def test_link_text_is_normalised_with_fallbacks():
    text = {link.number: link.text for link in collect_links(HOMEPAGE, SITE)}

    assert text[1] == "Galeria Exemple"  # image alt
    assert text[7] == "Prova U 12.09 – 30.11"  # whitespace collapsed across elements
    assert text[8] == "Vistes de sala"  # aria-label


def test_link_text_and_count_are_capped():
    anchors = "".join(f'<a href="/p{i}">{"x" * 300}</a>' for i in range(200))
    links = collect_links(f"<html><body>{anchors}</body></html>", SITE)

    assert len(links) == discover.MAX_LINKS
    assert links[-1].number == discover.MAX_LINKS
    assert all(len(link.text) == discover.LINK_TEXT_CHARS for link in links)


def test_redirected_homepage_and_base_href():
    """`page_url` is the final URL, and `<base href>` moves where relative links resolve."""
    html = '<html><head><base href="/ca/"></head><body><a href="agenda">Agenda</a></body></html>'
    [link] = collect_links(html, "https://exemple.cat/")
    assert link.url == "https://exemple.cat/ca/agenda"


@pytest.mark.parametrize("html", ["", "<html><body><p>cap enllaç</p></body></html>", b"\x00\x01"])
def test_pages_without_links_give_nothing(html):
    assert collect_links(html, SITE) == []


# --- GRAF hints ---------------------------------------------------------------------------------


def _event(title: str, **fields) -> SimpleNamespace:
    urls = {"web_url_ca": None, "web_url_es": None, "web_url_en": None}
    return SimpleNamespace(title=title, **{**urls, **fields})


def test_hints_keep_titles_and_on_site_urls_only():
    hints = GrafHints.from_events(
        [
            _event("Prova U", web_url_ca="https://exemple.cat/ca/exposicions/prova-u"),
            _event("Prova U", web_url_en="https://www.exemple.cat/ca/exposicions/prova-u/"),
            _event("Fira", web_url_ca="https://entrades.altre.cat/fira"),  # a ticketing site
            _event("  Segona\n mostra ", web_url_es="no és una URL"),
        ],
        SITE,
    )

    assert hints.titles == ("Prova U", "Fira", "Segona mostra")
    assert hints.urls == ("https://exemple.cat/ca/exposicions/prova-u",)


def test_hints_are_capped():
    events = [_event("t" * 500 + str(i), web_url_ca=f"{SITE}e/{i}") for i in range(40)]
    hints = GrafHints.from_events(events, SITE)

    assert len(hints.titles) <= discover.MAX_HINT_TITLES
    assert len(hints.urls) == discover.MAX_HINT_URLS
    assert all(len(title) <= discover.HINT_TITLE_CHARS for title in hints.titles)


def test_hints_can_hold_nothing_but_titles_and_urls():
    # Adding a field here is a copyright decision (CLAUDE.md § 1), not a refactor.
    assert [f.name for f in dataclasses.fields(GrafHints)] == ["titles", "urls"]


def test_no_graf_prose_reaches_either_prompt():
    """An event object that still carried every scrubbed GRAF key would leak none of them."""
    prose = {key: f"PROSA-{key}" for key in (*DROP_KEYS, "desc_ca", "desc_es", "desc_en")}
    event = _event("Prova U", web_url_ca=f"{SITE}ca/exposicions/prova-u", **prose)
    hints = GrafHints.from_events([event], SITE)

    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [1]})
    stub.tool_reply(PASS2_TOOL, _verdict("current_listing").model_dump())
    _pick(stub, hints=hints)
    _classify(stub, hints=hints)

    pass1, pass2 = (_prompt(request) for request in stub.converse_requests)
    for prompt in (pass1, pass2):
        assert "Prova U" in prompt
        assert f"{SITE}ca/exposicions/prova-u" in prompt
        assert "PROSA" not in prompt


def test_prompts_without_hints_say_nothing_about_them():
    assert "Hints" not in discover.pass1_prompt(VENUE, SITE, LINKS, GrafHints())
    assert "Hints" not in discover.pass2_prompt(VENUE, SITE, "text", GrafHints())


# --- pass 1 -------------------------------------------------------------------------------------


def test_pass1_returns_the_chosen_links_in_order():
    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [2, 1]}, input_tokens=5000, output_tokens=20)

    assert _pick(stub) == [LINKS[1], LINKS[0]]

    [request] = stub.converse_requests
    assert request["toolConfig"]["toolChoice"] == {"tool": {"name": PASS1_TOOL}}
    assert request["inferenceConfig"]["temperature"] == 0
    assert "[2] (nav) Agenda — https://www.exemple.cat/ca/agenda" in _prompt(request)
    [row] = stub.calls
    assert (row.purpose, row.model, row.input_tokens) == ("discover", HAIKU, 5000)


def test_pass1_invented_link_number_is_rejected():
    """The model answers by number so it cannot invent a URL; an invented number is refused,
    told why, and refused again."""
    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [1, 99]}, tool_use_id="use_1")
    stub.tool_reply(PASS1_TOOL, {"links": [0]}, tool_use_id="use_2")

    with pytest.raises(DiscoveryError, match="no link numbered"):
        _pick(stub)

    assert len(stub.calls) == 2  # both attempts are recorded spend
    retry = stub.converse_requests[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    result = retry[2]["content"][0]["toolResult"]
    assert (result["toolUseId"], result["status"]) == ("use_1", "error")
    assert "no link numbered [99]; the list runs from 1 to 3" in result["content"][0]["text"]


def test_pass1_retry_can_succeed():
    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": [4]})
    stub.tool_reply(PASS1_TOOL, {"links": [1]})

    assert _pick(stub) == [LINKS[0]]
    assert len(stub.calls) == 2


@pytest.mark.parametrize(
    "tool_input",
    [
        {"links": [1, 2, 3, 1]},  # more than MAX_CANDIDATES
        {"links": [1, 1]},
        {"links": ["https://www.exemple.cat/inventat"]},
        {"links": [1], "url": "https://www.exemple.cat/inventat"},
        {},
    ],
)
def test_pass1_malformed_answers_are_invalid(tool_input):
    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, tool_input)
    stub.tool_reply(PASS1_TOOL, tool_input)

    with pytest.raises(DiscoveryError):
        _pick(stub)


def test_pass1_empty_answer_is_valid():
    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, {"links": []})
    assert _pick(stub) == []


def test_pass1_without_links_makes_no_call():
    stub = StubLlm()
    assert _pick(stub, links=[]) == []
    assert stub.calls == []


def test_a_model_that_ignores_the_tool_is_retried_then_fails():
    stub = StubLlm()
    stub.tool_reply(PASS1_TOOL, None)
    stub.tool_reply(PASS1_TOOL, None)

    with pytest.raises(DiscoveryError, match="did not call the tool"):
        _pick(stub)
    # Nothing to answer with a tool result, so the retry repeats the original request.
    assert len(stub.converse_requests[1]["messages"]) == 1


# --- pass 2 -------------------------------------------------------------------------------------


def test_pass2_returns_the_verdict():
    stub = StubLlm()
    stub.tool_reply(
        PASS2_TOOL,
        {"page_type": "agenda", "dated_items": 7, "language": "ca", "confidence": "medium"},
    )

    verdict = _classify(stub, text="Agenda d'octubre")

    assert verdict == _verdict("agenda", "medium", 7)
    [request] = stub.converse_requests
    assert request["toolConfig"]["toolChoice"] == {"tool": {"name": PASS2_TOOL}}
    assert "<page>\nAgenda d'octubre\n</page>" in _prompt(request)
    assert stub.calls[0].purpose == "discover"


def test_pass2_page_text_is_capped():
    stub = StubLlm()
    stub.tool_reply(PASS2_TOOL, _verdict("other").model_dump())
    _classify(stub, text="a" * 50_000 + "FINAL")

    prompt = _prompt(stub.converse_requests[0])
    assert "a" * discover.PASS2_TEXT_CHARS in prompt
    assert "a" * (discover.PASS2_TEXT_CHARS + 1) not in prompt
    assert "FINAL" not in prompt


@pytest.mark.parametrize(
    "change",
    [
        {"page_type": "listing"},  # not one of the five
        {"confidence": "0.9"},
        {"dated_items": -1},
        {"language": "català"},
        {"reason": "The page says: ..."},  # free text has nowhere to go
    ],
)
def test_pass2_invalid_answer_retries_once_then_fails(change):
    bad = {**_verdict("current_listing").model_dump(), **change}
    stub = StubLlm()
    stub.tool_reply(PASS2_TOOL, bad)
    stub.tool_reply(PASS2_TOOL, bad)

    with pytest.raises(DiscoveryError):
        _classify(stub)
    assert len(stub.calls) == 2


def test_validation_feedback_does_not_echo_the_answer():
    """The retry message names the field and the rule, not what the model wrote: a rejected
    free-text field could be a quote from the page."""
    stub = StubLlm()
    stub.tool_reply(PASS2_TOOL, {**_verdict("other").model_dump(), "reason": "COPIED SPAN"})
    stub.tool_reply(PASS2_TOOL, _verdict("other").model_dump())

    _classify(stub)

    feedback = stub.converse_requests[1]["messages"][2]["content"][0]["toolResult"]
    assert "reason" in feedback["content"][0]["text"]
    assert "COPIED SPAN" not in feedback["content"][0]["text"]


@pytest.mark.parametrize("model", [LinkChoice, PageVerdict])
def test_tool_schemas_have_no_free_text_field(model):
    """Pass 2 reads page text, so a string it could fill freely is a way to store a quote
    (CLAUDE.md § 1). Every property is an enum, an integer or a list of integers."""
    schema = model.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "$defs" not in schema  # Bedrock gets one flat object
    for name, prop in schema["properties"].items():
        item = prop.get("items", prop)
        assert item["type"] == "integer" or "enum" in item, f"{name} is free text"


# --- routing ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("page_type", "confidence", "expected"),
    [
        ("current_listing", "high", SeedStatus.ACCEPTED),
        ("agenda", "high", SeedStatus.ACCEPTED),
        ("other", "high", SeedStatus.REJECTED),
        ("past_archive", "high", SeedStatus.REJECTED),
        ("single_show", "high", SeedStatus.AMBIGUOUS),
        ("current_listing", "medium", SeedStatus.AMBIGUOUS),
        ("agenda", "low", SeedStatus.AMBIGUOUS),
        ("other", "medium", SeedStatus.AMBIGUOUS),
        ("past_archive", "low", SeedStatus.AMBIGUOUS),
        ("single_show", "low", SeedStatus.AMBIGUOUS),
    ],
)
def test_route_one_candidate(page_type, confidence, expected):
    assert route(_verdict(page_type, confidence)) is expected


def test_route_ignores_the_item_count():
    """A listing with nothing on right now is still the listing."""
    assert route(_verdict("current_listing", dated_items=0)) is SeedStatus.ACCEPTED


def test_site_with_an_accepted_candidate_is_accepted():
    routing = route_site(
        [
            (LINKS[0].url, _verdict("current_listing")),
            (LINKS[1].url, _verdict("agenda", "medium")),
            (LINKS[2].url, _verdict("other")),
        ]
    )

    assert routing.status is SeedStatus.ACCEPTED
    assert [c.status for c in routing.candidates] == [
        SeedStatus.ACCEPTED,
        SeedStatus.AMBIGUOUS,
        SeedStatus.REJECTED,
    ]
    assert [c.url for c in routing.accepted] == [LINKS[0].url]


@pytest.mark.parametrize(
    "verdicts",
    [
        [],  # pass 1 chose nothing
        [(LINKS[2].url, _verdict("other")), (LINKS[1].url, _verdict("past_archive"))],
        [(LINKS[0].url, _verdict("single_show"))],
        [(LINKS[0].url, _verdict("current_listing", "medium"))],
    ],
)
def test_site_without_an_accepted_candidate_is_ambiguous(verdicts):
    """Even when every candidate was confidently rejected: no listing page is a question for a
    person, not an answer."""
    assert route_site(verdicts).status is SeedStatus.AMBIGUOUS


def test_proposal_that_differs_from_the_seed_in_use_is_ambiguous():
    routing = route_site([(LINKS[1].url, _verdict("agenda"))], seeds_in_use=[LINKS[0].url])

    assert routing.status is SeedStatus.AMBIGUOUS
    assert [c.status for c in routing.candidates] == [SeedStatus.AMBIGUOUS]


def test_seed_in_use_is_confirmed_whatever_its_spelling():
    routing = route_site(
        [(LINKS[0].url, _verdict("current_listing")), (LINKS[1].url, _verdict("agenda"))],
        seeds_in_use=["http://exemple.cat/ca/exposicions/"],
    )

    assert routing.status is SeedStatus.ACCEPTED
    assert [c.status for c in routing.candidates] == [SeedStatus.ACCEPTED, SeedStatus.AMBIGUOUS]


def test_seed_in_use_does_not_rescue_a_rejected_page():
    routing = route_site([(LINKS[0].url, _verdict("other"))], seeds_in_use=[LINKS[0].url])
    assert [c.status for c in routing.candidates] == [SeedStatus.REJECTED]
