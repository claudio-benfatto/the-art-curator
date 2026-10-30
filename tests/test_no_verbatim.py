"""CLAUDE.md § 1: facts from GRAF, prose from nobody. Do not skip or weaken these tests.

Schema half (P0): no source-fact table may carry a prose column, and third-party page text may
live only in `venue_pages.raw_text` and, for link text, `venue_pages.links`. Checked against the
models *and* the migrated database, so a hand-written migration cannot slip a column past the
models.

Repository half (P1): **third-party prose must not enter git either.** A recorded GRAF fixture is
the obvious way for it to happen — 45 of 158 venue profiles carry a `description` of up to 6330
characters, and every event carries `acf.desc_{ca,es,en}`. Fixtures are therefore scrubbed at
capture, and `test_fixtures_carry_no_third_party_prose` is the check that they stayed that way.
This one is not like a failing test elsewhere: once prose is in a pushed commit, removing it is a
history rewrite. That is why the scan landed in the same commit as the first fixture.

Crawl fixtures (P2): tests for discovery and crawling need HTML, and a recorded venue page is
third-party prose in git. Every HTML fixture is therefore hand-written, says so with a
`<!-- synthetic -->` marker on its first line, and stays small enough to read in review.

Extraction half (P2): `summary_en` is the only prose that outlives a venue page, so it has to be
ours. The prompt forbids reproducing ~25 consecutive words; `extract.longest_shared_run` enforces
20. A summary that copies is sent back once, and one that copies twice is dropped while the facts
are kept. The tests here plant a copied span and check each step.
"""

import asyncio
import json
import re
from datetime import date
from pathlib import Path

import pytest

from art_curator.db.models import Base
from art_curator.ingest.graf import DROP_KEYS, DROP_PREFIXES, banned_keys, scrub
from art_curator.llm import extract
from art_curator.llm.extract import MAX_SHARED_WORDS, longest_shared_run
from tests.db import run_sql
from tests.llm_stub import StubLlm

FIXTURES = Path(__file__).parent / "fixtures"

# Every table must be classified here. A new table fails `test_every_table_is_classified` until
# someone decides which rules apply to it — that decision is the point.
SOURCE_TABLES = {"venues", "event_snapshots", "event_occurrences"}  # facts only, ever
# Third-party text, 7-day TTL, purged. `links` carries each link's text, and a CHECK ties it to
# `raw_text`, so one purge covers both (tests/test_crawl.py).
PROSE_CACHE = {"venue_pages": {"raw_text", "links"}}
# Our data; no third-party text. `venue_seeds` is enums, counts and URLs (tests/test_seeds.py).
OWN_TABLES = {"llm_calls", "venue_seeds"}
# Facts read off a page, plus prose that is ours: `summary_en`, and nothing else that could hold
# a sentence, passed `longest_shared_run` before it was written (`test_extraction_tables_*`).
OWN_PROSE = {"exhibitions": {"summary_en"}, "art_events": {"summary_en"}}

PROSE = re.compile(r"desc|summary|body|content|excerpt|abstract|prose|blurb|text|html", re.I)
THIRD_PARTY_TEXT = re.compile(r"raw|html|body|excerpt|page_text", re.I)

# Base tables only: PostGIS adds views (geometry_columns, ...) and its own spatial_ref_sys.
DB_COLUMNS = (
    "SELECT c.table_name, c.column_name FROM information_schema.columns c "
    "JOIN information_schema.tables t USING (table_schema, table_name) "
    "WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE' "
    "AND c.table_name NOT IN ('spatial_ref_sys', 'alembic_version')"
)


def _model_columns() -> dict[str, set[str]]:
    return {name: {c.name for c in table.columns} for name, table in Base.metadata.tables.items()}


def _db_columns(url) -> dict[str, set[str]]:
    columns: dict[str, set[str]] = {}
    for table, column in run_sql(url, DB_COLUMNS)[0]:
        columns.setdefault(table, set()).add(column)
    return columns


def _violations(columns: dict[str, set[str]]) -> list[str]:
    found = []
    for table, names in columns.items():
        for name in names:
            if table in SOURCE_TABLES and PROSE.search(name):
                found.append(f"{table}.{name}: prose-like column on a source-fact table")
            elif (
                table not in SOURCE_TABLES
                and THIRD_PARTY_TEXT.search(name)
                and name not in PROSE_CACHE.get(table, set())
            ):
                found.append(f"{table}.{name}: third-party text outside venue_pages.raw_text")
    return found


def test_every_table_is_classified():
    classified = SOURCE_TABLES | set(PROSE_CACHE) | OWN_TABLES | set(OWN_PROSE)
    assert set(Base.metadata.tables) == classified


# Every column that can hold free text the model wrote, beyond `summary_en`, is capped by the
# extraction schema at a title's length or less (`test_extract.py::test_only_the_summary_*`).
SHORT_TEXT = {"title", "title_key", "artists", "themes"}


@pytest.mark.parametrize("table", sorted(OWN_PROSE))
def test_extraction_tables_hold_no_sentence_but_our_summary(table):
    """Adding a column here is a copyright decision (CLAUDE.md § 1), not a refactor."""
    shared = {
        "id", "venue_id", "start_date", "end_date", "at_venue", "detail_url", "media",
        "confidence", "source_url", "content_hash", "model", "extract_version", "first_seen_at",
        "last_seen_at",
    }  # fmt: skip
    event_only = {"kind", "start_time", "exhibition_id"} if table == "art_events" else set()
    columns = set(Base.metadata.tables[table].columns.keys())
    assert columns == shared | event_only | SHORT_TEXT | OWN_PROSE[table]


def test_models_carry_no_third_party_prose():
    assert not _violations(_model_columns())


def test_database_carries_no_third_party_prose(migrated_db):
    db = _db_columns(migrated_db)
    assert set(db) == set(Base.metadata.tables), "database has tables the models don't declare"
    assert not _violations(db)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("venues", "description"),
        ("event_snapshots", "desc_en"),
        ("event_snapshots", "summary"),
        ("venues", "body_html"),
        ("llm_calls", "response_body"),
        ("venue_pages", "raw_html"),
    ],
)
def test_planted_column_is_caught(table, column):
    assert _violations({table: {column}})


# --- Recorded fixtures: prose must not enter the repository ---------------------------------------


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES.rglob("*.json"))


def test_there_are_fixtures_to_scan():
    """A scan matching nothing would pass forever. Fail instead if the fixtures ever move."""
    assert _fixture_files(), f"no JSON fixtures under {FIXTURES}"


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda p: p.name)
def test_fixtures_carry_no_third_party_prose(path: Path):
    found = banned_keys(json.loads(path.read_text()))
    assert not found, (
        f"{path.relative_to(FIXTURES.parent)} carries {sorted(found)}. Re-record it with "
        f"`--record`, which scrubs; do not hand-edit, and do not push this."
    )


def test_fixtures_carry_no_long_third_party_text():
    """A second, key-blind pass: the length of the longest string in every fixture.

    `banned_keys` only catches prose in a key we already know about. Anything over ~400 characters
    in a facts-only payload is prose under some name we have not seen — a name, address, title or
    URL is nowhere near that long (the longest in the current fixtures is a 102-character address).
    """

    def longest(payload, path=()):
        if isinstance(payload, str):
            return [(len(payload), ".".join(path), payload[:80])]
        if isinstance(payload, dict):
            return [x for k, v in payload.items() for x in longest(v, (*path, str(k)))]
        if isinstance(payload, list):
            return [x for i, v in enumerate(payload) for x in longest(v, (*path, str(i)))]
        return []

    for path in _fixture_files():
        strings = longest(json.loads(path.read_text()))
        worst = max(strings, default=(0, "", ""))
        assert worst[0] <= 400, f"{path.name}: {worst[1]} holds {worst[0]} chars: {worst[2]!r}"


@pytest.mark.parametrize("key", sorted(DROP_KEYS))
def test_scrub_removes_planted_prose(key):
    """Like `test_planted_column_is_caught`, for the capture path rather than the schema."""
    planted = {"id": 1, "name": "keep", key: "third-party prose", "nested": [{key: "prose"}]}
    clean = scrub(planted)
    assert banned_keys(planted) == {key}
    assert banned_keys(clean) == set()
    assert clean == {"id": 1, "name": "keep", "nested": [{}]}


@pytest.mark.parametrize("prefix", DROP_PREFIXES)
def test_scrub_removes_planted_prose_by_prefix(prefix):
    for lang in ("ca", "es", "en"):
        planted = {"acf": {f"{prefix}{lang}": "prosa", "title_ca": "keep"}}
        assert scrub(planted) == {"acf": {"title_ca": "keep"}}


# --- HTML fixtures: synthetic only ----------------------------------------------------------------

SYNTHETIC_MARKER = "<!-- synthetic -->"
# A hand-written page has no reason to be long; a saved one is rarely under 50 KB.
MAX_HTML_FIXTURE_BYTES = 8_000


def _html_fixtures() -> list[Path]:
    return sorted(p for p in FIXTURES.rglob("*") if p.suffix.lower() in (".html", ".htm"))


def _html_fixture_problem(content: bytes) -> str | None:
    if not content.startswith(SYNTHETIC_MARKER.encode()):
        return f"does not start with {SYNTHETIC_MARKER}"
    if len(content) > MAX_HTML_FIXTURE_BYTES:
        return f"is {len(content)} bytes, over the {MAX_HTML_FIXTURE_BYTES} cap"
    return None


def test_there_are_html_fixtures_to_scan():
    assert _html_fixtures(), f"no HTML fixtures under {FIXTURES}"


@pytest.mark.parametrize("path", _html_fixtures(), ids=lambda p: p.name)
def test_html_fixtures_are_synthetic(path: Path):
    problem = _html_fixture_problem(path.read_bytes())
    assert problem is None, (
        f"{path.relative_to(FIXTURES.parent)} {problem}. HTML fixtures are written by hand, "
        f"never saved from a venue site; do not push a recorded page."
    )


@pytest.mark.parametrize(
    "content",
    [
        b"<!DOCTYPE html><html><body>a saved page</body></html>",
        b"<html><!-- synthetic --></html>",  # the marker leads the file or does not count
        SYNTHETIC_MARKER.encode() + b"x" * MAX_HTML_FIXTURE_BYTES,
    ],
)
def test_planted_html_fixture_is_caught(content):
    assert _html_fixture_problem(content)


# --- Extraction: the summary is ours --------------------------------------------------------------

# An invented page. `PASSAGE` is the sentence a lazy summary would lift.
PASSAGE = (
    "The exhibition gathers forty small paintings made on salvaged shutters during two winters "
    "in a village that the reservoir later covered, shown beside the notebooks in which the "
    "painter logged the weather of every single working day."
)
PAGE_TEXT = f"Galeria Exemple Exposicions Agenda Prova U Maria Exemple 12.09 – 30.11 {PASSAGE} Avís"
OWN_SUMMARY = (
    "Maria Exemple paints on reclaimed window shutters. The works remember a village drowned "
    "by a reservoir, and hang alongside her weather diaries."
)


def _span(words: int) -> str:
    """The first `words` words of `PASSAGE`."""
    return " ".join(PASSAGE.split()[:words])


def _answer(summary: str | None) -> dict:
    return {
        "exhibitions": [
            {
                "title": "Prova U",
                "artists": ["Maria Exemple"],
                "start": {"day": 12, "month": 9, "year": None},
                "end": {"day": 30, "month": 11, "year": None},
                "at_venue": True,
                "link": None,
                "summary_en": summary,
                "themes": ["memory"],
                "media": ["painting"],
                "confidence": "high",
            }
        ],
        "events": [],
    }


def _extract(stub: StubLlm) -> extract.Extraction:
    return asyncio.run(
        extract.extract_page(
            stub.client,
            "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            venue_name="Galeria Exemple",
            url="https://www.exemple.cat/ca/exposicions",
            text=PAGE_TEXT,
            fetched_on=date(2026, 10, 10),
        )
    )


def test_guard_is_stricter_than_the_prompt():
    assert MAX_SHARED_WORDS == 20  # PLAN.md § 8; the prompt says ~25
    assert len(PASSAGE.split()) > 25


def test_shared_run_is_counted_in_words():
    assert longest_shared_run(_span(21), PAGE_TEXT) == 21
    assert longest_shared_run(f"Our lead-in. {_span(12)} and our own ending.", PAGE_TEXT) == 12
    assert longest_shared_run(OWN_SUMMARY, PAGE_TEXT) <= 3
    assert longest_shared_run("", PAGE_TEXT) == 0
    assert longest_shared_run(OWN_SUMMARY, "") == 0


def test_shared_run_ignores_case_punctuation_and_line_breaks():
    disguised = _span(21).upper().replace(" ", ",\n", 5).replace("PAINTINGS", "«paintings»")
    assert longest_shared_run(disguised, PAGE_TEXT) == 21


def test_shared_run_is_the_longest_not_the_total():
    """Two short borrowings do not add up to a long one."""
    words = PASSAGE.split()
    stitched = f"{' '.join(words[:15])} — nothing of theirs here — {' '.join(words[15:30])}"
    assert longest_shared_run(stitched, PAGE_TEXT) == 15


def test_shared_run_finds_a_span_that_starts_late_in_a_repetitive_page():
    """The page repeats its opening words; the copy must still be matched from the right place."""
    page = f"The exhibition gathers. The exhibition gathers forty. {PASSAGE}"
    assert longest_shared_run(_span(25), page) == 25


@pytest.mark.parametrize(("words", "allowed"), [(20, True), (21, False), (33, False)])
def test_planted_copied_span_is_caught(words, allowed):
    """Like `test_planted_column_is_caught`, for what the model writes."""
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, _answer(_span(words)))
    stub.tool_reply(extract.TOOL, _answer(_span(words)))

    result = _extract(stub)

    [exhibition] = result.exhibitions
    if allowed:
        assert exhibition.summary_en == _span(words)
        assert (result.summaries_dropped, len(stub.calls)) == (0, 1)
    else:
        assert exhibition.summary_en is None
        assert (result.summaries_dropped, len(stub.calls)) == (1, 2)
    # Either way the facts survive: a copied summary costs the summary, not the exhibition.
    assert (exhibition.title, exhibition.start_date) == ("Prova U", date(2026, 9, 12))


def test_copied_summary_is_sent_back_once_and_a_rewrite_is_kept():
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, _answer(PASSAGE), tool_use_id="use_1")
    stub.tool_reply(extract.TOOL, _answer(OWN_SUMMARY))

    result = _extract(stub)

    assert result.exhibitions[0].summary_en == OWN_SUMMARY
    assert result.summaries_dropped == 0
    feedback = stub.converse_requests[1]["messages"][2]["content"][0]["toolResult"]
    assert (feedback["toolUseId"], feedback["status"]) == ("use_1", "error")
    [told] = feedback["content"]
    assert "exhibitions.0.summary_en" in told["text"]
    assert f"repeats {len(PASSAGE.split())} consecutive words" in told["text"]
    assert _span(5) not in told["text"]  # where and how long, never the span


def test_summary_may_come_back_null_instead():
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, _answer(PASSAGE))
    stub.tool_reply(extract.TOOL, _answer(None))

    result = _extract(stub)

    assert result.exhibitions[0].summary_en is None
    assert result.summaries_dropped == 0  # the model's own choice, not the guard's


@pytest.mark.parametrize("second", [{"exhibitions": "none"}, None])
def test_facts_survive_when_the_retry_after_a_copy_is_unusable(second):
    """The first answer was valid apart from its prose, so it is not thrown away."""
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, _answer(PASSAGE))
    stub.tool_reply(extract.TOOL, second)

    result = _extract(stub)

    assert [e.title for e in result.exhibitions] == ["Prova U"]
    assert result.exhibitions[0].summary_en is None
    assert result.summaries_dropped == 1


def test_only_the_copied_summary_is_dropped():
    both = _answer(PASSAGE)
    both["exhibitions"].append(
        {**both["exhibitions"][0], "title": "Altra", "summary_en": OWN_SUMMARY}
    )
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, both)
    stub.tool_reply(extract.TOOL, both)

    result = _extract(stub)

    assert [e.summary_en for e in result.exhibitions] == [None, OWN_SUMMARY]
    assert result.summaries_dropped == 1


def test_a_copy_after_an_invalid_answer_is_dropped_without_a_third_call():
    """One retry in all: validation and the guard share it."""
    stub = StubLlm()
    stub.tool_reply(extract.TOOL, {})
    stub.tool_reply(extract.TOOL, _answer(PASSAGE))

    result = _extract(stub)

    assert result.exhibitions[0].summary_en is None
    assert (result.summaries_dropped, len(stub.calls)) == (1, 2)
