"""Architecture invariants from CLAUDE.md, checked by scanning source text.

§2: model clients are constructed and called only in `llm/client.py`.
§3: `api/` and `agent/` never reference Telegram, and `queries/` knows nothing of who asks: no
agent, API, client, MCP or CLI imports, so the dev MCP server and the agent tools share it as is.
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
_CALLERS = r"(agent|api|clients|mcp_server|cli)"
CALLER = re.compile(
    r"^\s*(from|import)\s+(mcp|fastapi|starlette|typer|telegram|art_curator\." + _CALLERS + r")\b"
    r"|^\s*from\s+art_curator\s+import\b.*\b" + _CALLERS + r"\b"
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
    assert not _violations(_sources(PKG / layer), TELEGRAM), f"{layer}/ references Telegram"


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
