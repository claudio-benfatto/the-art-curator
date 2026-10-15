"""The forced-tool pattern over Converse, shared by discovery and extraction.

Bedrock has no structured outputs on this endpoint (CLAUDE.md § 5). The closest equivalent is one
tool the model must call, whose input is validated with pydantic and, when invalid, handed back
as a failed tool result for one more try. These are the pieces of that exchange; how many tries,
and what to do when they run out, is the caller's business.

Nothing here calls a model.
"""

from typing import Any

from pydantic import BaseModel, ValidationError


def flat_schema(output: type[BaseModel]) -> dict[str, Any]:
    """`output`'s JSON schema as one object: every `$ref` replaced by what it points at."""
    schema = output.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, list):
            return [inline(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            rest = {key: value for key, value in node.items() if key != "$ref"}
            return inline({**defs[node["$ref"].rpartition("/")[2]], **rest})
        return {key: inline(value) for key, value in node.items()}

    return inline(schema)


def forced_tool(name: str, description: str, output: type[BaseModel]) -> dict[str, Any]:
    """A `toolConfig` with one tool, which the model must call."""
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": flat_schema(output)},
                }
            }
        ],
        "toolChoice": {"tool": {"name": name}},
    }


def tool_use(response: dict[str, Any]) -> dict[str, Any] | None:
    """The `toolUse` block of a Converse response, or None if the model answered in text."""
    content = response["output"]["message"]["content"]
    return next((block["toolUse"] for block in content if "toolUse" in block), None)


def validation_feedback(exc: ValidationError) -> str:
    """Location and message only. pydantic's default rendering echoes the input back, and a
    rejected string could be a quote from the page (CLAUDE.md § 1)."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'input'}: {error['msg']}"
        for error in exc.errors()
    )


def with_failed_result(
    messages: list[dict[str, Any]], response: dict[str, Any], use: dict[str, Any], error: str
) -> list[dict[str, Any]]:
    """`messages` plus the model's reply and `error` as that tool call's failed result: the
    conversation to send for the retry."""
    failed = {"toolUseId": use["toolUseId"], "content": [{"text": error}], "status": "error"}
    return [
        *messages,
        response["output"]["message"],
        {"role": "user", "content": [{"toolResult": failed}]},
    ]
