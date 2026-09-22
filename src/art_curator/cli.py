"""Command-line entry point: `python -m art_curator.cli <command>`."""

import typer
from sqlalchemy.engine import make_url

from art_curator.config import get_settings

app = typer.Typer(no_args_is_help=True)


@app.command()
def config() -> None:
    """Print the effective configuration."""
    settings = get_settings().model_dump()
    settings["database_url"] = make_url(settings["database_url"]).render_as_string(
        hide_password=True
    )
    for key, value in settings.items():
        typer.echo(f"{key}={value if value is not None else ''}")


@app.callback()
def main() -> None:
    """The Art Curator."""


if __name__ == "__main__":
    app()
