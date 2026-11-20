"""Architecture invariants from CLAUDE.md, checked by scanning source text.

§2: model clients are constructed and called only in `llm/client.py`.
§3: `api/` and `agent/` never reference Telegram, and `queries/` knows nothing of who asks: no
agent, API, client, MCP or CLI imports, so the dev MCP server and the agent tools share it as is.
The agent knows no transport (no web framework, API or client imports); the API knows no client;
and a client reaches the curator only over HTTP, never by importing the backend.
"""

import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "art_curator"
MODEL_CLIENT = PKG / "llm" / "client.py"

MODEL_ACCESS = re.compile(
    r"AnthropicBedrock|AnthropicBedrockMantle|\bAnthropic\s*\("
    r"|bedrock-runtime|bedrock-mantle"
    r"|\.messages\.(create|stream)\b|\.converse(_stream)?\s*\(|\.invoke_model\w*\s*\("
)
TELEGRAM = re.compile(r"telegram", re.IGNORECASE)


def _imports(*, packages: tuple[str, ...] = (), ours: tuple[str, ...]) -> re.Pattern[str]:
    """An import of any of `packages`, or of any of `ours` (subpackages of art_curator), in either
    `import` form."""
    modules = [*packages, *(rf"art_curator\.{name}" for name in ours)]
    return re.compile(
        r"^\s*(from|import)\s+(" + "|".join(modules) + r")\b"
        r"|^\s*from\s+art_curator\s+import\b.*\b(" + "|".join(ours) + r")\b"
    )


TRANSPORT = _imports(
    packages=("fastapi", "starlette", "uvicorn", "mcp", "typer", "telegram"),
    ours=("api", "clients", "cli", "mcp_server"),
)
CLIENT = _imports(packages=("mcp", "typer", "telegram"), ours=("clients", "cli", "mcp_server"))
BACKEND = _imports(ours=("agent", "api", "queries", "db", "llm", "ingest", "eval", "config"))
CALLER = _imports(
    packages=("mcp", "fastapi", "starlette", "typer", "telegram"),
    ours=("agent", "api", "clients", "mcp_server", "cli"),
)


def _sources(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py")) if root.exists() else []


def _violations(files: list[Path], pattern: re.Pattern[str]) -> list[str]:
    return [
        f"{path.relative_to(PKG.parent)}:{lineno}: {line.strip()}"
        for path in files
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]


def test_model_access_only_in_llm_client():
    files = [p for p in _sources(PKG) if p != MODEL_CLIENT]
    assert not _violations(files, MODEL_ACCESS), "model access outside llm/client.py"


@pytest.mark.parametrize("layer", ["api", "agent"])
def test_backend_does_not_know_telegram(layer):
    files = _sources(PKG / layer)
    assert files, f"{layer}/ moved: update this test"
    assert not _violations(files, TELEGRAM), f"{layer}/ references Telegram"


def test_the_agent_knows_no_transport():
    assert not _violations(_sources(PKG / "agent"), TRANSPORT), "agent/ imports a transport"


def test_the_api_knows_no_client():
    assert not _violations(_sources(PKG / "api"), CLIENT), "api/ imports a client or the CLI"


def test_clients_reach_the_curator_only_over_http():
    files = _sources(PKG / "clients")
    assert files, "clients/ moved: update this test"
    assert not _violations(files, BACKEND), "clients/ imports the backend"


def test_queries_know_nothing_of_their_callers():
    files = _sources(PKG / "queries")
    assert files, "queries/ moved: update this test"
    assert not _violations(files, CALLER), "queries/ imports an agent or transport concern"


@pytest.mark.parametrize(
    "line",
    [
        "from mcp.server.mcpserver import MCPServer",
        "import typer",
        "from art_curator.agent.loop import run",
        "from art_curator import mcp_server",
        "from art_curator import config, cli",
    ],
)
def test_caller_pattern_catches(line):
    assert CALLER.search(line)


@pytest.mark.parametrize(
    ("pattern", "line"),
    [
        (TRANSPORT, "from fastapi import FastAPI"),
        (TRANSPORT, "from art_curator.api.main import app"),
        (TRANSPORT, "from art_curator import clients"),
        (CLIENT, "from art_curator.clients.api import CuratorApi"),
        (CLIENT, "from art_curator import cli"),
        (BACKEND, "from art_curator.agent.loop import Curator"),
        (BACKEND, "from art_curator import config"),
        (BACKEND, "import art_curator.db.models"),
    ],
)
def test_import_patterns_catch(pattern, line):
    assert pattern.search(line)


@pytest.mark.parametrize(
    ("pattern", "line"),
    [
        (TRANSPORT, "from art_curator.agent.tools import CuratorTools"),
        (CLIENT, "from art_curator.agent.loop import Curator"),
        (BACKEND, "import httpx2"),
        (BACKEND, "from art_curator.clients import api"),
    ],
)
def test_import_patterns_let_through(pattern, line):
    assert not pattern.search(line)


@pytest.mark.parametrize(
    "line",
    [
        "chat = AnthropicBedrockMantle(aws_region=r)",
        "client = boto3.client('bedrock-runtime')",
        "resp = await chat.messages.create(model=m)",
        "brt.converse(modelId=m, messages=msgs)",
        "brt.invoke_model_with_response_stream(body=b)",
    ],
)
def test_model_access_pattern_catches(line):
    assert MODEL_ACCESS.search(line)
