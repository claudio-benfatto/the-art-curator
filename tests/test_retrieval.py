"""`eval/retrieval.py` and `cli eval-retrieval`: how a ranking is scored against the labels, what
the report says, and what a draft offers a person to label. The model is stubbed."""

from decimal import Decimal

import pytest
import yaml
from typer.testing import CliRunner

from art_curator import cli
from art_curator.db.session import get_sessionmaker
from art_curator.eval import retrieval
from art_curator.eval.retrieval import Label, ModelRun, Query, RetrievalGold, key
from art_curator.ingest.embedding import Item
from art_curator.llm.client import db_recorder
from tests.corpus import EMBEDDED_BY, vector
from tests.llm_stub import StubLlm
from tests.test_smoke import _clear_caches


def _item(venue: str, title: str) -> Item:
    return Item("x:1", venue, title, f"{title}.", None, None)


GALLERY, ANNEX = "Galeria Exemple", "Annex Exemple"
POOL = [_item(GALLERY, "Prova U"), _item(GALLERY, "Col·lecció"), _item(ANNEX, "Assaig U")]


def test_items_are_ranked_by_cosine_and_a_repeated_title_counts_once():
    tour = _item(GALLERY, "Visita")
    vectors = [
        (POOL[0], [0.0, 1.0]),
        (tour, [0.6, 0.8]),
        (POOL[1], [1.0, 0.0]),
        (tour, [0.0, -1.0]),  # the same tour on another day, far away: its best rank stands
    ]

    assert retrieval.ranked_keys([2.0, 0.0], vectors) == [
        key(GALLERY, "Col·lecció"),
        key(GALLERY, "Visita"),
        key(GALLERY, "Prova U"),
    ]


def test_a_query_is_scored_on_the_labels_in_the_pool():
    query = Query(
        query="drawings",
        confirmed=True,
        relevant=(
            Label(venue="GALERIA EXEMPLE", title="prova u."),  # case and punctuation ignored
            Label(venue=ANNEX, title="Assaig U"),
            Label(venue=GALLERY, title="Tancada"),  # no longer stored
        ),
    )
    ranked = [key(GALLERY, "Col·lecció"), key(GALLERY, "Prova U")] + [
        key(f"Venue {n}", "Other") for n in range(10)
    ]
    ranked.append(key(ANNEX, "Assaig U"))

    scored = retrieval.score(query, ranked)

    assert scored.ranks == (2, 13)
    assert scored.stale == (Label(venue=GALLERY, title="Tancada"),)
    assert (scored.recall(5), scored.recall(10), scored.reciprocal_rank) == (0.5, 0.5, 0.5)


def _gold() -> RetrievalGold:
    return RetrievalGold(
        queries=(
            Query(
                query="drawings", confirmed=True, relevant=(Label(venue=GALLERY, title="Prova U"),)
            ),
            Query(query="collections"),
        )
    )


def _run(model: str, *, drawings: list[Item], collections: list[Item]) -> ModelRun:
    rankings = {
        "drawings": [key(i.venue, i.title) for i in drawings],
        "collections": [key(i.venue, i.title) for i in collections],
    }
    return ModelRun(model, rankings, calls=6, cost_usd=Decimal("0.000012"))


def test_the_report_scores_each_model_on_the_confirmed_queries():
    first = _run("model-a", drawings=POOL, collections=POOL)
    second = _run("model-b", drawings=[POOL[1], POOL[2], POOL[0]], collections=POOL)
    stopped = ModelRun("model-c", {}, calls=1, cost_usd=Decimal(0), stopped="AccessDenied: no")

    lines = retrieval.report(_gold(), POOL, [first, second, stopped])

    assert "queries      2   1 confirmed   1 labels" in lines
    assert "model-a      1.00         1.00         1.00   $0.000012   6" in lines
    assert "model-b      1.00         1.00         0.33   $0.000012   6" in lines
    assert "model-c      stopped: the endpoint refused a call: AccessDenied: no" in lines
    assert f"{'':<12} {'model-b':<32} ranks 3" in lines
    assert not any("collections" in line for line in lines)  # unconfirmed: not scored


def test_unconfirmed_queries_are_never_scored():
    gold = RetrievalGold(
        queries=(Query(query="drawings", relevant=(Label(venue=GALLERY, title="Prova U"),)),)
    )

    assert retrieval.scores(gold, _run("m", drawings=POOL, collections=POOL)) == []


def test_a_draft_pools_each_models_top_items_in_turn():
    first = _run("model-a", drawings=POOL, collections=[POOL[0], POOL[1], POOL[2]])
    second = _run("model-b", drawings=POOL, collections=[POOL[2], POOL[1]])

    drafted = yaml.safe_load(retrieval.draft(_gold(), POOL, [first, second], depth=2))

    assert drafted == {
        "queries": [
            {
                "query": "collections",
                "confirmed": False,
                "relevant": [
                    {"venue": GALLERY, "title": "Prova U"},
                    {"venue": ANNEX, "title": "Assaig U"},
                    {"venue": GALLERY, "title": "Col·lecció"},
                ],
            }
        ]
    }
    assert retrieval.load_gold(retrieval.draft(_gold(), POOL, [first, second]))


def test_a_query_asked_twice_is_refused():
    with pytest.raises(ValueError, match="asked more than once"):
        retrieval.load_gold("queries:\n  - query: drawings\n  - query: drawings\n")


@pytest.fixture
def retrieval_cli(writable_corpus_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", writable_corpus_db.render_as_string(hide_password=False))
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    _clear_caches()
    stub = StubLlm(record=db_recorder(get_sessionmaker()))
    monkeypatch.setattr(cli, "get_llm_client", lambda: stub.client)
    yield stub
    _clear_caches()


def test_cli_eval_retrieval_embeds_in_memory_and_drafts(
    retrieval_cli, writable_corpus_db, tmp_path
):
    """The corpus's one summarised item, then every committed query, by one model. Nothing is
    confirmed yet, so the report scores nothing and the draft offers that item for each query."""
    queries = retrieval.load_gold().queries
    for _ in range(1 + len(queries)):
        retrieval_cli.embed_reply({"embeddings": [vector(1.0)]}, input_tokens=5)
    draft = tmp_path / "draft.yaml"

    result = CliRunner().invoke(
        cli.app, ["eval-retrieval", "--model", EMBEDDED_BY, "--draft", str(draft)]
    )

    assert result.exit_code == 0, result.output
    assert "pool         1 items with a summary" in result.output
    assert f"{EMBEDDED_BY:<12} no confirmed query to score" in result.output
    drafted = yaml.safe_load(draft.read_text())["queries"]
    assert len(drafted) == len(queries)
    assert drafted[0]["relevant"] == [{"venue": GALLERY, "title": "Prova U"}]
    assert [r["input_type"] for r in retrieval_cli.embed_requests] == ["search_document"] + [
        "search_query"
    ] * len(queries)
