"""`ingest/embedding.py` and `cli embed`: what an item is embedded from, which items are sent, and
that a second run sends none. The model is stubbed; that only our own text is embedded is checked
in `tests/test_no_verbatim.py`."""

import asyncio

import pytest
from typer.testing import CliRunner

from art_curator import cli
from art_curator.db.session import get_sessionmaker
from art_curator.ingest import embedding
from art_curator.llm.client import db_recorder
from tests.corpus import EMBEDDED_BY, OTHER_EMBEDDER, vector
from tests.db import run_sql
from tests.llm_stub import StubLlm
from tests.test_smoke import _clear_caches


def _item(document: str = "A show.", model: str | None = None, hashed: str | None = None):
    return embedding.Item("x:1", "Galeria Exemple", "Prova U", document, model, hashed)


def test_the_document_is_our_summary_then_its_tags():
    assert embedding.document(
        " Drawings on paper. ", ["memory", "archive"], ["drawing", "mixed_media"]
    ) == ("Drawings on paper.\nThemes: memory, archive.\nMedia: drawing, mixed media.")
    assert embedding.document("Drawings on paper.", [], []) == "Drawings on paper."


def test_only_new_changed_or_other_model_items_are_planned():
    fresh = _item()
    done = _item(model=EMBEDDED_BY, hashed=fresh.hash)
    changed = _item(model=EMBEDDED_BY, hashed=embedding.text_hash("An older summary."))
    other = _item(model=OTHER_EMBEDDER, hashed=fresh.hash)

    planned = embedding.plan([fresh, done, changed, other], EMBEDDED_BY)

    assert planned.to_embed == (fresh, changed, other)
    assert planned.unchanged == (done,)


def test_items_are_embedded_as_documents_and_a_refusal_stops_the_run():
    stub = StubLlm()
    stub.embed_reply({"embeddings": [vector(1.0)]}, input_tokens=4)
    stub.converse_stub.add_client_error("invoke_model", service_error_code="ThrottlingException")
    first, second, third = _item("One."), _item("Two."), _item("Three.")

    run = asyncio.run(embedding.embed_items(stub.client, EMBEDDED_BY, [first, second, third]))

    assert [(done.item, done.vector) for done in run.embedded] == [(first, vector(1.0))]
    assert run.stopped.startswith("ThrottlingException")
    assert [r["input_type"] for r in stub.embed_requests] == ["search_document"] * 2
    assert [r["texts"] for r in stub.embed_requests] == [["One."], ["Two."]]
    assert [(c.purpose, c.error_type) for c in stub.calls] == [
        ("embed", None),
        ("embed", "ThrottlingException"),
    ]
    lines = embedding.report(EMBEDDED_BY, embedding.Plan((first, second, third)), run)
    assert "embedded    1   2 left for the next run" in lines


@pytest.fixture
def embed_cli(writable_corpus_db, monkeypatch):
    """`cli embed` over a corpus of its own, the model stubbed and every call recorded there."""
    monkeypatch.setenv("DATABASE_URL", writable_corpus_db.render_as_string(hide_password=False))
    monkeypatch.setenv("EMBED_MODEL", EMBEDDED_BY)
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    _clear_caches()
    stub = StubLlm(record=db_recorder(get_sessionmaker()))
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)
    yield stub
    _clear_caches()


def test_cli_embed_embeds_a_changed_summary_once(embed_cli, writable_corpus_db):
    """Prova U is the corpus's one item with a summary; its stored vector was made from other
    text, so it is embedded again, and a second run sends nothing."""
    embed_cli.embed_reply({"embeddings": [vector(0.0, 0.0, 1.0)]}, input_tokens=9)

    first = CliRunner().invoke(cli.app, ["embed"])
    second = CliRunner().invoke(cli.app, ["embed"])

    assert first.exit_code == 0, first.output
    assert "1 to embed" in first.output
    assert second.exit_code == 0, second.output
    assert "0 to embed   1 unchanged" in second.output
    [[(model, hashed, third)]] = run_sql(
        writable_corpus_db,
        "SELECT embed_model, embedded_hash, (embedding::real[])[3] FROM exhibitions WHERE id = 1",
    )
    assert (model, third) == (EMBEDDED_BY, 1.0)
    assert hashed == embedding.text_hash("Drawings on paper, made for this show.")
    [[(calls,)]] = run_sql(
        writable_corpus_db, "SELECT count(*) FROM llm_calls WHERE purpose = 'embed'"
    )
    assert calls == 1


def test_cli_embed_needs_a_model(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "")
    _clear_caches()
    result = CliRunner().invoke(cli.app, ["embed"])
    _clear_caches()

    assert result.exit_code == 1
    assert "EMBED_MODEL is not set" in result.output
