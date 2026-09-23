"""Command-line entry point: `python -m art_curator.cli <command>`."""

import asyncio
from typing import Annotated

import typer
from sqlalchemy import select
from sqlalchemy.engine import make_url

from art_curator.config import get_settings
from art_curator.db.models import LlmCall
from art_curator.db.session import get_engine, get_sessionmaker
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


@app.callback()
def main() -> None:
    """The Art Curator."""


if __name__ == "__main__":
    app()
