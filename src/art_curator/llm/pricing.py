"""Per-call cost from token usage and `pricing.yaml`. Pure: no I/O beyond reading the price table.

Prices are USD per million tokens. Arithmetic is in `Decimal` and the result is quantized to
the precision of `llm_calls.cost_usd` (6 dp), so a stored cost is exactly what this returns.
"""

from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict

PER_TOKEN = Decimal(1_000_000)
COST_QUANTUM = Decimal("0.000001")  # llm_calls.cost_usd is Numeric(12, 6)


class UnknownModelError(KeyError):
    """The model has no entry in pricing.yaml. Add one; never guess a price."""


class MissingPriceError(ValueError):
    """The usage reports tokens of a kind the model has no price for."""


class ModelPrice(BaseModel):
    """USD per million tokens. Cache prices are optional: non-Claude models may not cache."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Decimal
    output: Decimal
    cache_write_5m: Decimal | None = None
    cache_write_1h: Decimal | None = None
    cache_read: Decimal | None = None


class PriceTable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    models: dict[str, ModelPrice]


@dataclass(frozen=True)
class Usage:
    """Token counts for one call. `input_tokens` excludes cache reads and writes, matching the
    Messages API `usage` object; cache writes are split by TTL because they're priced apart."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def cache_write_tokens(self) -> int:
        return self.cache_write_5m_tokens + self.cache_write_1h_tokens


@lru_cache
def load_prices() -> PriceTable:
    raw = files("art_curator.llm").joinpath("pricing.yaml").read_text()
    return PriceTable.model_validate(yaml.safe_load(raw))


def price_for(model: str, table: PriceTable | None = None) -> ModelPrice:
    table = table or load_prices()
    try:
        return table.models[model]
    except KeyError:
        raise UnknownModelError(model) from None


def cost_usd(model: str, usage: Usage, table: PriceTable | None = None) -> Decimal:
    price = price_for(model, table)
    parts = [
        (usage.input_tokens, price.input, "input"),
        (usage.output_tokens, price.output, "output"),
        (usage.cache_write_5m_tokens, price.cache_write_5m, "cache_write_5m"),
        (usage.cache_write_1h_tokens, price.cache_write_1h, "cache_write_1h"),
        (usage.cache_read_tokens, price.cache_read, "cache_read"),
    ]
    total = Decimal(0)
    for tokens, per_mtok, kind in parts:
        if not tokens:
            continue
        if per_mtok is None:
            raise MissingPriceError(f"{model} has no {kind} price but reported {tokens} tokens")
        total += tokens * per_mtok
    return (total / PER_TOKEN).quantize(COST_QUANTUM)
