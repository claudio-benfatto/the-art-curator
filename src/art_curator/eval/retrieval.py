"""Choose the embedding model for semantic ranking (PLAN.md § 8, P3): a retrieval check over
~20 queries, run against each candidate.

The queries, `eval/gold/retrieval.yaml`, are what the curator would pass as `search_events`'s
`query`: a few English words about what someone is after ("video art about migration"). Each is
labelled with the items that answer it, by venue and title, once a person has read their
summaries (`confirmed`). Labels are facts: a venue's name and a title, nothing else.

For each candidate model, every item that has a summary is embedded as `cli embed` would embed it
(`ingest/embedding.document`) and every query as a query, and the items are ranked by cosine
similarity to each query. The scores, over the confirmed queries:

- **recall@5 and recall@10**: the share of a query's labelled items ranked in its top 5 or 10.
- **MRR**: the mean of 1 / the rank of a query's best-ranked labelled item.

The pool is every item with a summary, not only the current ones: items are never deleted, so a
label never goes stale while its item is still stored. An item added after a query was labelled
may answer it without being labelled, which costs that query recall it deserved: re-draft and
re-label when the pool has grown. A label whose item is not in the pool is reported stale and not
scored.

This module is pure: the gold set, the ranking, the scores, the report and the draft. Embedding
is `eval/retrieval_runner.py`.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from importlib.resources import files
from itertools import zip_longest
from statistics import mean

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from art_curator.ingest.embedding import Item
from art_curator.ingest.extraction import title_key
from art_curator.ingest.matching import normalize_name

GOLD_FILE = "gold/retrieval.yaml"

# The candidates the app role may call (Terraform `embed_candidate_model_ids`, PR 32);
# `tests/test_config.py` keeps the two lists equal.
CANDIDATES = ("amazon.titan-embed-text-v2:0", "cohere.embed-v4:0")

RECALL_AT = (5, 10)
DRAFT_DEPTH = 8  # each model's top items per query that a draft offers a person to label

_GOLD = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Label(BaseModel):
    """An item that answers a query: its venue (`venues.name`) and title, as stored. Case,
    accents and punctuation do not matter."""

    model_config = _GOLD

    venue: str = Field(min_length=1)
    title: str = Field(min_length=1)


class Query(BaseModel):
    model_config = _GOLD

    query: str = Field(min_length=1, max_length=200)
    # True once a person has read every pooled item's summary and kept the ones that answer it.
    # Unconfirmed queries are never scored; `--draft` offers them candidates.
    confirmed: bool = False
    relevant: tuple[Label, ...] = ()


class RetrievalGold(BaseModel):
    model_config = _GOLD

    queries: tuple[Query, ...]

    @field_validator("queries")
    @classmethod
    def _unique(cls, value: tuple[Query, ...]) -> tuple[Query, ...]:
        repeated = [q for q, n in Counter(q.query for q in value).items() if n > 1]
        if repeated:
            raise ValueError(f"queries asked more than once: {repeated}")
        return value


def load_gold(text: str | None = None) -> RetrievalGold:
    """Parse the queries (the committed file unless `text` is given)."""
    if text is None:
        text = files("art_curator.eval").joinpath(GOLD_FILE).read_text(encoding="utf-8")
    return RetrievalGold.model_validate(yaml.safe_load(text) or {})


# --- ranking -------------------------------------------------------------------------------------

Key = tuple[str, str]  # (venue, title), each normalised: what a label and an item are matched on


def key(venue: str, title: str) -> Key:
    return normalize_name(venue) or venue.casefold(), title_key(title)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.hypot(*a) * math.hypot(*b))


def ranked_keys(
    query_vector: Sequence[float], vectors: Sequence[tuple[Item, Sequence[float]]]
) -> list[Key]:
    """The pool's items most similar first, as keys. Items that share a key (a tour repeated on
    several days) count once, at the best rank any of them reached."""
    ordered = sorted(vectors, key=lambda pair: -cosine(query_vector, pair[1]))
    return list(dict.fromkeys(key(item.venue, item.title) for item, _ in ordered))


# --- scores --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryScore:
    query: str
    ranks: tuple[int, ...]  # 1-based, one per labelled item in the pool, best first
    stale: tuple[Label, ...]  # labelled items not in the pool

    def recall(self, at: int) -> float:
        return sum(rank <= at for rank in self.ranks) / len(self.ranks)

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.ranks[0]


def score(query: Query, ranked: Sequence[Key]) -> QueryScore:
    position = {k: rank for rank, k in enumerate(ranked, start=1)}
    ranks, stale = [], []
    for label in query.relevant:
        found = position.get(key(label.venue, label.title))
        if found is None:
            stale.append(label)
        else:
            ranks.append(found)
    return QueryScore(query.query, tuple(sorted(ranks)), tuple(stale))


@dataclass(frozen=True)
class ModelRun:
    """One candidate over the whole gold set."""

    model: str
    rankings: Mapping[str, list[Key]]  # query -> the pool, most similar first
    calls: int
    cost_usd: Decimal
    stopped: str = ""  # what the endpoint said when it refused a call; nothing is scored then


def scores(gold: RetrievalGold, run: ModelRun) -> list[QueryScore]:
    """The confirmed queries' scores. A query none of whose labels is in the pool has no score."""
    found = [
        score(query, run.rankings[query.query])
        for query in gold.queries
        if query.confirmed and query.query in run.rankings
    ]
    return [s for s in found if s.ranks]


# --- the report ----------------------------------------------------------------------------------


def report(gold: RetrievalGold, pool: Sequence[Item], runs: Sequence[ModelRun]) -> list[str]:
    """What `eval-retrieval` prints: the pool and gold set, one line of scores per model, then
    each confirmed query's ranks under every model. Titles and venue names only."""
    lines: list[str] = []

    def block(label: str, *parts: str) -> None:
        lines.append(f"{label:<12} " + "   ".join(part for part in parts if part))

    confirmed = [q for q in gold.queries if q.confirmed]
    keys = {key(item.venue, item.title) for item in pool}
    stale = [
        label
        for q in confirmed
        for label in q.relevant
        if key(label.venue, label.title) not in keys
    ]
    block("pool", f"{len(pool)} items with a summary", f"{len(keys)} distinct venue and title")
    block(
        "queries",
        f"{len(gold.queries)}",
        f"{len(confirmed)} confirmed",
        f"{sum(len(q.relevant) for q in confirmed)} labels",
        f"{len(stale)} stale" if stale else "",
    )
    for label in stale:
        block("stale", f"{label.venue} · {label.title}", "not in the pool")

    header = "   ".join(f"recall@{at:<3}" for at in RECALL_AT)
    block("model", f"{header}   MRR    cost        calls")
    by_model = {}
    for run in runs:
        if run.stopped:
            block(run.model, f"stopped: the endpoint refused a call: {run.stopped}")
            continue
        found = scores(gold, run)
        by_model[run.model] = {s.query: s for s in found}
        if not found:
            block(run.model, "no confirmed query to score", f"${run.cost_usd:.6f}", f"{run.calls}")
            continue
        recalls = "   ".join(f"{mean(s.recall(at) for s in found):<10.2f}" for at in RECALL_AT)
        mrr = mean(s.reciprocal_rank for s in found)
        block(run.model, f"{recalls}   {mrr:.2f}   ${run.cost_usd:.6f}   {run.calls}")

    for query in confirmed:
        block("query", repr(query.query))
        for model, found in by_model.items():
            result = found.get(query.query)
            ranks = ", ".join(map(str, result.ranks)) if result else "nothing labelled in the pool"
            block("", f"{model:<32} ranks {ranks}")
    return lines


def draft(
    gold: RetrievalGold, pool: Sequence[Item], runs: Sequence[ModelRun], depth: int = DRAFT_DEPTH
) -> str:
    """The unconfirmed queries as YAML, each with every model's top `depth` items as candidate
    labels, taken in turn from each model so that no model's view of relevance leads (pooling).
    A person reads each candidate's summary, keeps the ones that answer the query, and sets
    `confirmed: true`."""
    shown = {key(item.venue, item.title): (item.venue, item.title) for item in pool}
    usable = [run for run in runs if not run.stopped]
    queries = []
    for query in gold.queries:
        if query.confirmed:
            continue
        tops = [run.rankings.get(query.query, [])[:depth] for run in usable]
        pooled = [k for rank in zip_longest(*tops) for k in rank if k is not None]
        labels = [{"venue": shown[k][0], "title": shown[k][1]} for k in dict.fromkeys(pooled)]
        queries.append({"query": query.query, "confirmed": False, "relevant": labels})
    return yaml.safe_dump({"queries": queries}, allow_unicode=True, sort_keys=False, width=100)
