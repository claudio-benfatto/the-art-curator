"""Command-line entry point: `python -m art_curator.cli <command>`."""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.engine import make_url

from art_curator.config import get_settings
from art_curator.db.models import LlmCall
from art_curator.db.session import get_engine, get_sessionmaker
from art_curator.ingest import crawl as crawling
from art_curator.ingest import graf, seeds
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
