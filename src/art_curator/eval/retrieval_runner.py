"""Everything `cli eval-retrieval` does around the scorer: embed the pool and the queries with each
candidate, in memory, and rank. Nothing is written but `llm_calls`: the vectors `cli embed` keeps
are left alone, so the check runs whatever `EMBED_MODEL` is.

A call the endpoint refuses (no grant, an agreement not accepted) stops that model's run: it is
reported, and the next model still runs.
"""

from collections.abc import Sequence

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from art_curator.db.models import LlmCall
from art_curator.eval.retrieval import ModelRun, RetrievalGold, ranked_keys
from art_curator.ingest import embedding
from art_curator.ingest.embedding import Item
from art_curator.llm.client import LlmClient
from art_curator.obs import telemetry


async def run_model(
    llm: LlmClient,
    sessions: async_sessionmaker[AsyncSession],
    model: str,
    gold: RetrievalGold,
    pool: Sequence[Item],
) -> ModelRun:
    """Every query's ranking of `pool` under `model`, in a trace of its own."""
    with llm.tracer.start_as_current_span("eval-retrieval") as span:
        trace_id, _ = telemetry.span_ids(span)
        rankings = {}
        documents = await embedding.embed_items(llm, model, pool)
        stopped = documents.stopped
        if not stopped:
            vectors = [(done.item, done.vector) for done in documents.embedded]
            for query in gold.queries:
                try:
                    vector = await llm.embed(
                        purpose="embed", model=model, text=query.query, input_type="query"
                    )
                except (BotoCoreError, ClientError) as exc:
                    stopped = f"{type(exc).__name__}: {exc}"
                    break
                rankings[query.query] = ranked_keys(vector, vectors)

    async with sessions() as session:
        calls, cost = (
            await session.execute(
                select(func.count(), func.coalesce(func.sum(LlmCall.cost_usd), 0)).where(
                    LlmCall.trace_id == trace_id
                )
            )
        ).one()
    return ModelRun(model, rankings if not stopped else {}, calls, cost, stopped)
