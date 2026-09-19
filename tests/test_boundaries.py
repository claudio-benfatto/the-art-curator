"""Architecture invariants from CLAUDE.md, checked by scanning source text.

§2: model clients are constructed and called only in `llm/client.py`.
§3: `api/` and `agent/` never reference Telegram.
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
