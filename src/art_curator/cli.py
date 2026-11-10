"""Command-line entry point: `python -m art_curator.cli <command>`."""

import asyncio
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.config import get_settings
from art_curator.db.models import LlmCall
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.eval import extraction as evaluation
from art_curator.eval import runner as eval_runner
from art_curator.ingest import crawl as crawling
from art_curator.ingest import extraction, graf, seeds
from art_curator.ingest.http import RobotsPolicy, open_client
from art_curator.ingest.sync import (
    GrafSnapshot,
    SyncPlan,
    SyncResult,
    plan_sync,
    report,
    write_graf,
)
from art_curator.llm.client import get_llm_client
from art_curator.llm.pricing import Usage, cost_usd
from art_curator.obs import telemetry

app = typer.Typer(no_args_is_help=True)

SMOKE_PROMPT = "Reply with the single word: ok"


@app.command()
def config() -> None:
    """Print the effective configuration."""
    settings = get_settings().model_dump()
    settings["database_url"] = make_url(settings["database_url"]).render_as_string(
        hide_password=True
    )
    for key, value in settings.items():
        typer.echo(f"{key}={value if value is not None else ''}")


@app.command()
def smoke(
    model: Annotated[str | None, typer.Option(help="Model ID. Defaults to EXTRACT_MODEL.")] = None,
    mantle: Annotated[
        bool, typer.Option(help="Call via Mantle (Messages API) instead of Converse.")
    ] = False,
) -> None:
    """Make one real, recorded model call and check its llm_calls row. Costs a fraction of a cent.

    Passes when the call succeeds and its row carries tokens, a trace id, and a cost that
    matches pricing.yaml for the recorded tokens.
    """
    model = model or get_settings().extract_model
    row, error = asyncio.run(_smoke(model, mantle=mantle))

    if row is None:
        typer.echo(f"FAIL: no llm_calls row for the call ({error or 'no error raised'})")
        raise typer.Exit(1)

    typer.echo(
        f"{row.provider} {row.model}\n"
        f"  tokens    in={row.input_tokens} out={row.output_tokens} "
        f"cache_write={row.cache_creation_input_tokens} cache_read={row.cache_read_input_tokens}\n"
        f"  cost      ${row.cost_usd}\n"
        f"  latency   {row.latency_ms} ms\n"
        f"  trace     {row.trace_id} span {row.span_id}\n"
        f"  request   {row.request_id}"
    )
    if error is not None:
        typer.echo(f"FAIL: call raised {error} (recorded as error_type={row.error_type})")
        raise typer.Exit(1)

    # No cache TTL on the row, so this assumes 5-minute writes — true for a one-shot call.
    expected = cost_usd(
        row.model,
        Usage(
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cache_write_5m_tokens=row.cache_creation_input_tokens,
            cache_read_tokens=row.cache_read_input_tokens,
        ),
    )
    problems = [
        msg
        for ok, msg in [
            (row.output_tokens > 0, "no output tokens recorded"),
            (row.trace_id is not None, "no trace id on the row"),
            (row.cost_usd == expected, f"cost {row.cost_usd} != {expected} from pricing.yaml"),
        ]
        if not ok
    ]
    if problems:
        typer.echo("FAIL: " + "; ".join(problems))
        raise typer.Exit(1)
    typer.echo("OK")


async def _smoke(model: str, *, mantle: bool) -> tuple[LlmCall | None, str | None]:
    client = get_llm_client()
    error: str | None = None
    try:
        with client.tracer.start_as_current_span("smoke") as span:
            trace_id, _ = telemetry.span_ids(span)
            try:
                if mantle:
                    await client.create_message(
                        purpose="smoke",
                        model=model,
                        max_tokens=16,
                        messages=[{"role": "user", "content": SMOKE_PROMPT}],
                    )
                else:
                    await client.converse_message(
                        purpose="smoke",
                        model=model,
                        messages=[{"role": "user", "content": [{"text": SMOKE_PROMPT}]}],
                        inferenceConfig={"maxTokens": 16},
                    )
            except Exception as exc:  # the client has already recorded it
                error = f"{type(exc).__name__}: {exc}"

        async with get_sessionmaker()() as session:
            row = await session.scalar(select(LlmCall).where(LlmCall.trace_id == trace_id))
        return row, error
    finally:
        telemetry.setup_telemetry().force_flush()
        await get_engine().dispose()


@app.command("sync-graf")
def sync_graf(
    dry_run: Annotated[
        bool, typer.Option(help="Run every write, print the report, then roll back.")
    ] = False,
    deep_occurrences: Annotated[
        bool,
        typer.Option(help="Also ask /events/{id}/occurrences for each post: one request per post."),
    ] = False,
    min_venues: Annotated[
        int,
        typer.Option(
            help="Fail without writing if GRAF serves fewer venue terms. A blocked or truncated "
            "fetch would otherwise unjoin every venue it missed."
        ),
    ] = 500,
    record: Annotated[
        Path | None,
        typer.Option(metavar="DIR", help="Also write the scrubbed capture to DIR as fixtures."),
    ] = None,
    allow_unmatched_pilots: Annotated[
        bool, typer.Option(help="Exit 0 even if a PLAN.md § 7 pilot matches no venue term.")
    ] = False,
) -> None:
    """Pull GRAF facts into Postgres: venue terms, their profile URLs, live events.

    An unmatched pilot exits 1 *after* committing: the facts are still worth having, and the exit
    code means a scheduled run cannot swallow it.
    """
    settings = get_settings()
    capture = asyncio.run(_fetch_graf(settings.graf_base_url, deep_occurrences=deep_occurrences))
    snapshot = GrafSnapshot.parse(capture)
    plan = plan_sync(snapshot)

    if len(plan.venues) < min_venues:
        typer.echo(
            f"FAIL: GRAF served {len(plan.venues)} venue terms, expected at least {min_venues}. "
            "Nothing written."
        )
        raise typer.Exit(1)
    if record is not None:
        for path in graf.record(capture, record):
            typer.echo(f"recorded {path}")

    result = asyncio.run(_write_graf(snapshot, plan, dry_run=dry_run))

    typer.echo("GRAF sync (dry run, rolled back)" if dry_run else "GRAF sync")
    for line in report(snapshot, plan, result):
        typer.echo(f"  {line}")
    for name in plan.unmatched_pilots:
        typer.echo(f"PILOT UNMATCHED: {name}")
    if plan.unmatched_pilots and not allow_unmatched_pilots:
        raise typer.Exit(1)


async def _fetch_graf(base_url: str, *, deep_occurrences: bool) -> graf.Capture:
    async with open_client() as client:
        return await graf.fetch(client, base_url, deep_occurrences=deep_occurrences)


async def _write_graf(snapshot: GrafSnapshot, plan: SyncPlan, *, dry_run: bool) -> SyncResult:
    try:
        async with get_sessionmaker()() as session:
            result = await write_graf(session, snapshot, plan.venues)
            await (session.rollback() if dry_run else session.commit())
            return result
    finally:
        await get_engine().dispose()


@app.command()
def discover(
    venue: Annotated[
        str | None,
        typer.Option(help="Only this venue, by name. Case and accents do not matter."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Run every fetch and model call, print the report, then roll back the seeds. "
            "The model calls are still made, paid for and recorded."
        ),
    ] = False,
) -> None:
    """Find where each crawlable pilot's site lists its exhibitions. Calls a model: about two
    cents per site.

    Sites the model is sure about are accepted. The rest are printed as ambiguous, for a person
    to settle in `ingest/seeds.yaml`; a venue pinned there is not sent to the model.
    """
    try:
        overrides = seeds.load_overrides()
    except ValidationError as exc:
        typer.echo(f"FAIL: ingest/seeds.yaml is invalid.\n{exc}")
        raise typer.Exit(1) from exc

    run = asyncio.run(_discover(overrides, venue, dry_run=dry_run))
    if run is None:
        typer.echo("FAIL: no crawlable pilot venues. Run sync-graf first.")
        raise typer.Exit(1)
    if not run.results:
        typer.echo(f"FAIL: no crawlable pilot venue is named {venue!r}. Known: {run.known}")
        raise typer.Exit(1)

    typer.echo("Discover (dry run, rolled back)" if dry_run else "Discover")
    for line in seeds.report(run.results, run.unmatched_overrides):
        typer.echo(f"  {line}")
    typer.echo(f"  {'written':<11} {run.written} seeds")
    typer.echo(f"  {'cost':<11} ${run.cost_usd}   {run.calls} model calls")


@dataclass(frozen=True)
class _DiscoverRun:
    results: list[seeds.SiteResult]
    known: str  # every crawlable pilot's name, for the `--venue` error
    unmatched_overrides: list[str]
    written: int = 0
    calls: int = 0
    cost_usd: Decimal = Decimal(0)


async def _discover(
    overrides: seeds.SeedOverrides, venue: str | None, *, dry_run: bool
) -> _DiscoverRun | None:
    model = get_settings().extract_model
    llm = get_llm_client()
    try:
        # Three short transactions rather than one held open across minutes of fetching.
        async with get_sessionmaker()() as session:
            sites = await seeds.load_sites(session, overrides)
        if not sites:
            return None
        names = [site.name for site in sites]
        unmatched = overrides.unmatched(names)
        selected = seeds.select_sites(sites, venue)
        if not selected:
            return _DiscoverRun([], ", ".join(sorted(names)), unmatched)

        with llm.tracer.start_as_current_span("discover") as span:
            trace_id, _ = telemetry.span_ids(span)
            async with open_client() as client:
                robots = RobotsPolicy(client)
                results = [
                    await seeds.discover_site(client, robots, llm, model, site) for site in selected
                ]

        async with get_sessionmaker()() as session:
            written = await seeds.write_seeds(session, results, model)
            await (session.rollback() if dry_run else session.commit())
            # Spend is recorded in its own transactions, so a dry run's rollback leaves it.
            calls, cost = (
                await session.execute(
                    select(func.count(), func.coalesce(func.sum(LlmCall.cost_usd), 0)).where(
                        LlmCall.trace_id == trace_id
                    )
                )
            ).one()
        return _DiscoverRun(results, "", unmatched, written, calls, cost)
    finally:
        telemetry.setup_telemetry().force_flush()
        await get_engine().dispose()


@app.command()
def crawl(
    pilot: Annotated[
        bool, typer.Option(help="Crawl the crawlable pilot venues. The only scope there is so far.")
    ] = False,
    venue: Annotated[
        str | None,
        typer.Option(help="Only this venue, by name. Case and accents do not matter."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Fetch every page, print the report, then roll back the pages. The purge of "
            "expired text is still committed: the TTL is not optional."
        ),
    ] = False,
) -> None:
    """Fetch each pilot site's listing pages into venue_pages. No model is called.

    Pages come from `ingest/seeds.yaml` pins, else from the seeds `discover` accepted. Text older
    than 7 days is purged first, every run. A broken seed exits 1 *after* committing, so a
    scheduled run cannot swallow it.
    """
    if not pilot:
        typer.echo("FAIL: only the pilot venues can be crawled so far. Pass --pilot.")
        raise typer.Exit(1)
    try:
        overrides = seeds.load_overrides()
    except ValidationError as exc:
        typer.echo(f"FAIL: ingest/seeds.yaml is invalid.\n{exc}")
        raise typer.Exit(1) from exc

    run = asyncio.run(_crawl(overrides, venue, dry_run=dry_run))
    if run is None:
        typer.echo("FAIL: no crawlable pilot venues. Run sync-graf first.")
        raise typer.Exit(1)
    if not run.crawls:
        typer.echo(f"FAIL: no crawlable pilot venue is named {venue!r}. Known: {run.known}")
        raise typer.Exit(1)

    typer.echo("Crawl (dry run, rolled back)" if dry_run else "Crawl")
    typer.echo(f"  {'purged':<11} {_purged(run.purged)}")
    for line in crawling.report(run.crawls):
        typer.echo(f"  {line}")
    typer.echo(f"  {'written':<11} {run.written} pages")
    if crawling.broken_seeds(run.crawls):
        raise typer.Exit(1)


@dataclass(frozen=True)
class _CrawlRun:
    crawls: list[crawling.SiteCrawl]
    known: str  # every crawlable pilot's name, for the `--venue` error
    purged: int
    written: int = 0


async def _crawl(
    overrides: seeds.SeedOverrides, venue: str | None, *, dry_run: bool
) -> _CrawlRun | None:
    try:
        # Short transactions rather than one held open across minutes of fetching.
        async with get_sessionmaker()() as session:
            purged = await crawling.purge_pages(session)
            await session.commit()
            targets = await crawling.load_targets(session, overrides)
        if not targets:
            return None
        selected = crawling.select_targets(targets, venue)
        if not selected:
            return _CrawlRun([], ", ".join(sorted(t.site.name for t in targets)), purged)

        async with open_client() as client:
            robots = RobotsPolicy(client)
            crawls = [await crawling.crawl_site(client, robots, target) for target in selected]

        async with get_sessionmaker()() as session:
            written = await crawling.write_pages(session, crawls)
            await (session.rollback() if dry_run else session.commit())
        return _CrawlRun(crawls, "", purged, written)
    finally:
        await get_engine().dispose()


@app.command()
def extract(
    venue: Annotated[
        str | None,
        typer.Option(help="Only this venue, by name. Case and accents do not matter."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Send at most this many pages to the model; the rest wait."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Run every model call, print the report, then roll back what was written. The "
            "calls are still made, paid for and recorded, and the purge is still committed."
        ),
    ] = False,
) -> None:
    """Read exhibitions and events off the crawled pages. Calls a model: a few cents a page
    (PLAN.md § 6 has the measured cost).

    Only a page whose text, prompt version or model changed since its last extraction is sent.
    Text older than 7 days is purged first, as in `crawl`. A page that failed, a listing that
    found items last time and none now, or a call the endpoint refused, exits 1 *after*
    committing what was extracted.
    """
    run = asyncio.run(_extract(venue, limit, dry_run=dry_run))
    if run is None:
        typer.echo("FAIL: no crawled pages. Run crawl --pilot first.")
        raise typer.Exit(1)
    if run.known:
        typer.echo(f"FAIL: no crawled page belongs to a venue named {venue!r}. Known: {run.known}")
        raise typer.Exit(1)

    typer.echo("Extract (dry run, rolled back)" if dry_run else "Extract")
    typer.echo(f"  {'purged':<11} {_purged(run.purged)}")
    for line in extraction.report(run.plan, run.results, run.stopped):
        typer.echo(f"  {line}")
    typer.echo(
        f"  {'written':<11} {run.written.exhibitions} exhibitions   {run.written.events} events"
        f"   {run.written.pages} pages marked"
    )
    per_page = f"   ${run.cost_usd / len(run.results):.6f} a page" if run.results else ""
    typer.echo(f"  {'cost':<11} ${run.cost_usd}   {run.calls} model calls{per_page}")
    if run.stopped or any(r.extraction is None or r.broken for r in run.results):
        raise typer.Exit(1)


@dataclass(frozen=True)
class _ExtractRun:
    purged: int
    plan: extraction.PagePlan = extraction.PagePlan()
    results: list[extraction.PageResult] = field(default_factory=list)
    known: str = ""  # set only when `--venue` matched nothing: every venue with a crawled page
    written: extraction.Written = extraction.Written()
    calls: int = 0
    cost_usd: Decimal = Decimal(0)
    stopped: str = ""  # the endpoint refused a call: the pages after it were not sent


async def _extract(venue: str | None, limit: int | None, *, dry_run: bool) -> _ExtractRun | None:
    model = get_settings().extract_model
    llm = get_llm_client()
    try:
        # Short transactions rather than one held open across minutes of model calls.
        async with get_sessionmaker()() as session:
            purged = await crawling.purge_pages(session)
            await session.commit()
            everything = await extraction.load_pages(session, model)
        if not everything.venue_names:
            return None
        plan = extraction.select_pages(everything, venue)
        if not plan.venue_names:
            return _ExtractRun(purged, known=", ".join(everything.venue_names))

        with llm.tracer.start_as_current_span("extract") as span:
            trace_id, _ = telemetry.span_ids(span)
            extracted = await extraction.extract_pages(llm, model, plan.to_extract[:limit])

        async with get_sessionmaker()() as session:
            written = await extraction.write_results(session, extracted.results, model)
            await (session.rollback() if dry_run else session.commit())
            # Spend is recorded in its own transactions, so a dry run's rollback leaves it.
            calls, cost = (
                await session.execute(
                    select(func.count(), func.coalesce(func.sum(LlmCall.cost_usd), 0)).where(
                        LlmCall.trace_id == trace_id
                    )
                )
            ).one()
        return _ExtractRun(
            purged, plan, extracted.results, "", written, calls, cost, extracted.stopped
        )
    finally:
        telemetry.setup_telemetry().force_flush()
        await get_engine().dispose()


@app.command("eval-extraction")
def eval_extraction(
    venue: Annotated[
        str | None,
        typer.Option(help="Only this venue, by name. Case and accents do not matter."),
    ] = None,
    draft: Annotated[
        Path | None,
        typer.Option(
            metavar="FILE",
            help="Instead of scoring, extract every crawled page the gold set does not label yet "
            "and write what was found to FILE as unconfirmed labels, for a person to check "
            "against the live page.",
        ),
    ] = None,
    judge: Annotated[
        bool,
        typer.Option(
            "--judge",
            help="Also have JUDGE_MODEL check every summary extraction wrote against its page: "
            "supported or not. About five cents a show's own page on Opus.",
        ),
    ] = False,
) -> None:
    """Score extraction against the hand-checked gold set, and discovery against its labelled
    listings. Calls a model: about two cents a labelled page (PLAN.md § 6).

    A page is scored only if its text is still the one it was labelled against, stored or fetched
    again; otherwise it is reported stale. Text older than 7 days is purged first, as in `crawl`;
    beyond that nothing is written but the cost of the calls. The scores are a baseline, not a
    gate: the run exits 1 only if the gold set is invalid or the endpoint refused a call.
    """
    if draft is not None and judge:
        typer.echo("FAIL: --judge scores a run, and --draft does not score: use one or the other.")
        raise typer.Exit(1)
    try:
        everything = evaluation.load_gold()
    except ValidationError as exc:
        typer.echo(f"FAIL: eval/gold/extraction.yaml is invalid.\n{exc}")
        raise typer.Exit(1) from exc
    gold = everything.for_venue(venue)

    if draft is not None:
        _eval_draft(gold, venue, draft)
        return

    if not gold.pages and not gold.listings:
        known = ", ".join(everything.venues) or "none"
        typer.echo(f"FAIL: the gold set labels nothing for {venue or 'any venue'}. Known: {known}")
        raise typer.Exit(1)
    run = asyncio.run(_eval(gold, judge))

    typer.echo("Eval extraction")
    for line in evaluation.report(
        gold,
        run.today,
        run.scores,
        run.skipped,
        run.discovery,
        judged=run.judged,
        stopped=run.stopped,
    ):
        typer.echo(f"  {line}")
    cost = f"  {'cost':<11} ${run.cost_usd}   {run.calls} model calls"
    if run.judged is not None:
        cost += f"   judge ${run.judge_cost_usd} in {run.judge_calls}"
    typer.echo(cost)
    if run.stopped:
        raise typer.Exit(1)


@dataclass(frozen=True)
class _EvalRun:
    today: date
    scores: list[evaluation.PageScore]
    skipped: list[evaluation.Skipped]
    discovery: evaluation.DiscoveryScore
    judged: list[evaluation.JudgedPage] | None = None  # None when the judge was not asked
    calls: int = 0
    cost_usd: Decimal = Decimal(0)
    judge_calls: int = 0
    judge_cost_usd: Decimal = Decimal(0)
    stopped: str = ""


async def _eval(gold: evaluation.GoldSet, judge: bool) -> _EvalRun:
    settings = get_settings()
    llm = get_llm_client()
    try:
        async with get_sessionmaker()() as session:
            await crawling.purge_pages(session)
            await session.commit()
            today = await eval_runner.today(session)
            stored = await eval_runner.load_stored(session)
            verdicts = await eval_runner.load_verdicts(session)

        with llm.tracer.start_as_current_span("eval-extraction") as span:
            trace_id, _ = telemetry.span_ids(span)
            async with open_client() as client:
                ready, skipped = await eval_runner.find_texts(
                    client, RobotsPolicy(client), gold.pages, stored, today
                )
            extracted = await eval_runner.extract_all(
                llm, settings.extract_model, [text for _, text in ready]
            )
            stopped = extracted.stopped
            judged = None
            # A run the endpoint stopped is run again anyway: judging half of it is money spent
            # twice.
            if judge and not stopped:
                judging = await eval_runner.judge_all(llm, settings.judge_model, extracted.results)
                judged, stopped = judging.pages, judging.stopped

        # `zip` stops where the run did: the pages after a refused call are not scored.
        scores = [
            evaluation.score_page(page, result.extraction, result.error)
            for (page, _), result in zip(ready, extracted.results, strict=False)
        ]
        discovery = evaluation.score_discovery(gold.listings, verdicts)
        async with get_sessionmaker()() as session:
            calls, cost = await _spend(session, trace_id)
            judge_calls, judge_cost = await _spend(session, trace_id, purpose="judge")
        return _EvalRun(
            today,
            scores,
            skipped,
            discovery,
            judged=judged,
            calls=calls,
            cost_usd=cost,
            judge_calls=judge_calls,
            judge_cost_usd=judge_cost,
            stopped=stopped,
        )
    finally:
        telemetry.setup_telemetry().force_flush()
        await get_engine().dispose()


def _eval_draft(gold: evaluation.GoldSet, venue: str | None, path: Path) -> None:
    if path.exists():
        typer.echo(f"FAIL: {path} exists. Draft to a new file: a half-checked draft is work.")
        raise typer.Exit(1)
    model = get_settings().extract_model
    run = asyncio.run(_draft(gold, venue, model))
    if not run.results and not run.stopped:
        typer.echo("FAIL: no crawled page with text is left to label. Run crawl --pilot first.")
        raise typer.Exit(1)

    path.write_text(eval_runner.draft_yaml(run.results, model, run.today), encoding="utf-8")
    drafted = [r for r in run.results if r.extraction is not None]
    typer.echo("Eval extraction (draft)")
    typer.echo(f"  {'drafted':<11} {len(drafted)} pages to {path}, every one unconfirmed")
    for result in run.results:
        if result.extraction is None:
            typer.echo(f"  {'failed':<11} {result.page.url}   {result.error}")
    if run.stopped:
        typer.echo(f"  {'stopped':<11} the endpoint refused a call: {run.stopped}")
    typer.echo(f"  {'cost':<11} ${run.cost_usd}   {run.calls} model calls")
    if run.stopped:
        raise typer.Exit(1)


@dataclass(frozen=True)
class _DraftRun:
    today: date
    results: list[eval_runner.Result]
    calls: int = 0
    cost_usd: Decimal = Decimal(0)
    stopped: str = ""


async def _draft(gold: evaluation.GoldSet, venue: str | None, model: str) -> _DraftRun:
    llm = get_llm_client()
    try:
        async with get_sessionmaker()() as session:
            await crawling.purge_pages(session)
            await session.commit()
            today = await eval_runner.today(session)
            pages = eval_runner.draft_pages(await eval_runner.load_stored(session), gold, venue)
        if not pages:
            return _DraftRun(today, [])

        with llm.tracer.start_as_current_span("eval-extraction") as span:
            trace_id, _ = telemetry.span_ids(span)
            extracted = await eval_runner.extract_all(llm, model, pages)
        async with get_sessionmaker()() as session:
            calls, cost = await _spend(session, trace_id)
        return _DraftRun(today, extracted.results, calls, cost, extracted.stopped)
    finally:
        telemetry.setup_telemetry().force_flush()
        await get_engine().dispose()


async def _spend(
    session: AsyncSession, trace_id: str | None, purpose: str | None = None
) -> tuple[int, Decimal]:
    """Model calls made under `trace_id`, for `purpose` if given, and what they cost."""
    query = select(func.count(), func.coalesce(func.sum(LlmCall.cost_usd), 0)).where(
        LlmCall.trace_id == trace_id
    )
    if purpose is not None:
        query = query.where(LlmCall.purpose == purpose)
    calls, cost = (await session.execute(query)).one()
    return calls, cost


@app.command("purge-pages")
def purge_pages() -> None:
    """Drop venue-page text older than 7 days. `crawl` does this first on every run; this is the
    same purge on its own, for a scheduler or for a day nothing is crawled."""
    typer.echo(f"Purged {_purged(asyncio.run(_purge_pages()))}")


def _purged(count: int) -> str:
    return f"{count} pages of text older than {crawling.PAGE_TTL_DAYS} days"


async def _purge_pages() -> int:
    try:
        async with get_sessionmaker()() as session:
            purged = await crawling.purge_pages(session)
            await session.commit()
            return purged
    finally:
        await get_engine().dispose()


@app.callback()
def main() -> None:
    """The Art Curator."""


if __name__ == "__main__":
    app()
