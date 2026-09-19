"""Command-line entry point: `python -m art_curator.cli <command>`."""

import typer

from art_curator.config import get_settings

app = typer.Typer(no_args_is_help=True)


@app.command()
def config() -> None:
    """Print the effective configuration."""
    for key, value in get_settings().model_dump().items():
        typer.echo(f"{key}={value if value is not None else ''}")


@app.callback()
def main() -> None:
    """The Art Curator."""


if __name__ == "__main__":
    app()
