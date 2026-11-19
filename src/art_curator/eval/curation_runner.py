"""Everything `cli eval-curation` does around the scorer (PLAN.md § 8, P3), scenario by scenario:

1. A throwaway user (`users.client = "eval"`) with the scenario's profile remembered for it, so the
   context block lists those facts and `remember` has someone to write about.
2. The conversation, in memory, through `Curator.turn` with the curator's own tools over the live
   database: what a person on `cli chat` would get the same day. Each turn is its own trace, as
   in `cli chat`.
3. After each turn: what the tools returned (read off the turn's messages), the items the reply
   names, their constraints checked in SQL, and the turn's model calls from `llm_calls`.
4. With `judge_model`, one call over the whole conversation (`llm/curation_judge.py`), in a trace
   of its own.
5. The user is deleted, and its profile with it (`ON DELETE CASCADE`).

**The eval leaves nothing behind but `llm_calls`.** A call the endpoint refuses stops the run
there: what ran is reported, and the user is deleted all the same.
"""

import time
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import date

from anthropic import APIError
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from art_curator.agent import profile
from art_curator.agent.loop import Conversation, Curator
from art_curator.agent.prompt import conversation_context
from art_curator.agent.store import ConversationStore
from art_curator.agent.tools import CuratorTools
from art_curator.db.models import LlmCall, User
from art_curator.eval import curation
from art_curator.eval.curation import CallUsage, Constraints, Named, OffConstraint, ScenarioRun
from art_curator.llm import curation_judge
from art_curator.llm.client import LlmClient
from art_curator.obs import telemetry
from art_curator.queries import events as q

EVAL_CLIENT = "eval"  # what `users.client` says for the eval's throwaway users


async def unresolved_places(session: AsyncSession, names: Sequence[str]) -> list[str]:
    """Why each of the area constraints' places cannot be checked: a name that matches nothing,
    or several places far apart. Asked before any model is called."""
    problems = []
    for name in names:
        match = await q.resolve_place(session, name)
        if match.place is None:
            problems.append(str(q.UnresolvedPlace(name, match.candidates)))
    return problems


async def run_scenario(
    llm: LlmClient,
    sessions: async_sessionmaker[AsyncSession],
    scenario: curation.Scenario,
    *,
    today: date,
    model: str,
    effort: str,
    judge_model: str | None,
) -> ScenarioRun:
    """Run `scenario` as a new user whose calendar starts on `today`, check every exchange, and
    judge the conversation if `judge_model` is given."""
    user_id = await ConversationStore(sessions).user(
        EVAL_CLIENT, f"{scenario.id}-{uuid.uuid4().hex[:8]}"
    )
    try:
        run = await _converse(llm, sessions, scenario, user_id, today, model, effort)
        if judge_model is None or run.stopped or not run.exchanges:
            return run
        return await _judge(llm, sessions, judge_model, run)
    finally:
        async with sessions() as session, session.begin():
            await session.execute(delete(User).where(User.id == user_id))


async def _converse(
    llm: LlmClient,
    sessions: async_sessionmaker[AsyncSession],
    scenario: curation.Scenario,
    user_id: int,
    today: date,
    model: str,
    effort: str,
) -> ScenarioRun:
    async with sessions() as session, session.begin():
        for fact in scenario.profile:
            await profile.remember(session, user_id, fact.topic, fact.fact)
        context = conversation_context(today, await profile.snapshot(session, user_id))
    conversation = Conversation(context)
    curator = Curator(llm, CuratorTools(sessions, user_id), model=model, effort=effort)

    exchanges = []
    stopped = ""
    for turn, constraints in zip(scenario.turns, scenario.constraints(today), strict=True):
        before = len(conversation.messages)
        started = time.perf_counter()
        try:
            done = await curator.turn(conversation, turn.user)
        except APIError as exc:
            stopped = f"{type(exc).__name__}: {exc}"
            break
        seconds = time.perf_counter() - started

        uses = curation.tool_uses(conversation.messages[before:])
        seen = curation.seen_items(curation.tool_uses(conversation.messages))
        named = curation.named_items(done.reply, seen)
        async with sessions() as session, session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            off = await off_constraint(session, named, constraints)
            calls = await usage(session, done.trace_id)
        exchanges.append(
            curation.Exchange(
                user=turn.user,
                constraints=constraints,
                expect=turn.expect,
                reply=done.reply,
                stop_reason=done.stop_reason,
                iteration_count=done.iteration_count,
                uses=tuple(uses),
                calls=tuple(calls),
                seconds=seconds,
                named=tuple(named),
                off=tuple(off),
                failures=tuple(curation.check_failures(turn.expect, uses)),
            )
        )

    async with sessions() as session:
        remembered = [f"{f.topic}: {f.fact}" for f in await profile.snapshot(session, user_id)]
    return ScenarioRun(scenario, context, tuple(exchanges), tuple(remembered), stopped=stopped)


async def _judge(
    llm: LlmClient, sessions: async_sessionmaker[AsyncSession], model: str, run: ScenarioRun
) -> ScenarioRun:
    with llm.tracer.start_as_current_span("eval-curation.judge") as span:
        trace_id, _ = telemetry.span_ids(span)
        try:
            verdicts = await curation_judge.judge_conversation(
                llm,
                model,
                persona=run.scenario.persona,
                context=run.context,
                exchanges=[curation_judge.Exchange(e.user, e.uses, e.reply) for e in run.exchanges],
            )
        except curation_judge.CurationJudgeError as exc:
            judged = replace(run, judge_error=str(exc))
        except (BotoCoreError, ClientError) as exc:
            judged = replace(run, stopped=f"{type(exc).__name__}: {exc}")
        else:
            judged = replace(run, verdicts=tuple(verdicts))
    async with sessions() as session:
        return replace(judged, judge_calls=tuple(await usage(session, trace_id)))


async def off_constraint(
    session: AsyncSession, named: Sequence[Named], constraints: Constraints
) -> list[OffConstraint]:
    """Each named item that breaks one of `constraints`, once per constraint it breaks. Decided
    by `search_events` itself (CLAUDE.md § 8): an item meets a constraint if the search for that
    constraint returns any item under its title. That is the SQL the tool runs, so this checks
    that the curator kept to what was asked, not the SQL, which `tests/test_queries.py` does."""
    if not named:
        return []
    off = []
    for constraint, arguments in constraints.checks():
        meets = {row.id for row in (await q.search_events(session, **arguments)).events}
        off += [
            OffConstraint(item.title, constraint) for item in named if not meets & set(item.ids)
        ]
    return off


async def usage(session: AsyncSession, trace_id: str | None) -> list[CallUsage]:
    """The model calls recorded under `trace_id`, in the order they were made."""
    rows = await session.execute(
        select(
            LlmCall.input_tokens,
            LlmCall.cache_read_input_tokens,
            LlmCall.cache_creation_input_tokens,
            LlmCall.output_tokens,
            LlmCall.cost_usd,
            LlmCall.latency_ms,
            LlmCall.stop_reason,
        )
        .where(LlmCall.trace_id == trace_id)
        .order_by(LlmCall.id)
    )
    return [CallUsage(*row) for row in rows]
