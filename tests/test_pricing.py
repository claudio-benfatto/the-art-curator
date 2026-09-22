"""Cost arithmetic against hand-computed values. Prices are USD per million tokens."""

from decimal import Decimal

import pytest

from art_curator.config import Settings
from art_curator.llm.pricing import (
    MissingPriceError,
    ModelPrice,
    PriceTable,
    UnknownModelError,
    Usage,
    cost_usd,
    load_prices,
)

OPUS = "anthropic.claude-opus-5"
HAIKU = "anthropic.claude-haiku-4-5"


@pytest.mark.parametrize(
    ("model", "usage", "expected"),
    [
        # Chat turn, warm cache: 2500×0.50 + 400×5 + 600×25 = 18 250 µ$
        (OPUS, Usage(input_tokens=400, output_tokens=600, cache_read_tokens=2500), "0.018250"),
        # Chat turn, cold cache (5m write): 2500×6.25 + 400×5 + 600×25 = 32 625 µ$
        (OPUS, Usage(input_tokens=400, output_tokens=600, cache_write_5m_tokens=2500), "0.032625"),
        # 1h write is 2× input: 2500×10 = 25 000 µ$
        (OPUS, Usage(cache_write_1h_tokens=2500), "0.025000"),
        # Extraction page (PLAN § 6): 3300×1 + 600×5 = 6 300 µ$
        (HAIKU, Usage(input_tokens=3300, output_tokens=600), "0.006300"),
        (HAIKU, Usage(), "0.000000"),
    ],
)
def test_cost_matches_hand_computed(model, usage, expected):
    assert cost_usd(model, usage) == Decimal(expected)


def test_cost_is_quantized_to_the_column_precision():
    # 3 cache-read tokens on Opus = 1.5 µ$; stored as Numeric(12, 6), so rounded to 6 dp.
    cost = cost_usd(OPUS, Usage(cache_read_tokens=3))
    assert cost.as_tuple().exponent == -6


def test_unknown_model_raises():
    with pytest.raises(UnknownModelError):
        cost_usd("anthropic.claude-nonexistent", Usage(input_tokens=1))


def test_cache_tokens_without_a_cache_price_raise():
    table = PriceTable(models={"embed": ModelPrice(input=Decimal("0.02"), output=Decimal(0))})
    assert cost_usd("embed", Usage(input_tokens=1000), table) == Decimal("0.000020")
    with pytest.raises(MissingPriceError):
        cost_usd("embed", Usage(cache_read_tokens=1), table)


def test_price_table_rejects_unknown_fields():
    with pytest.raises(ValueError):
        PriceTable.model_validate({"models": {"m": {"input": 1, "output": 1, "cache": 1}}})


def test_configured_default_models_are_priced():
    table = load_prices()
    settings = Settings(_env_file=None)
    for model in (settings.chat_model, settings.extract_model):
        assert model in table.models, f"{model} missing from pricing.yaml"


def test_claude_cache_prices_follow_the_multipliers():
    # 5m write 1.25×, 1h write 2×, read 0.1× input — catches a typo in pricing.yaml.
    for name, price in load_prices().models.items():
        if not name.startswith("anthropic.claude-"):
            continue
        assert price.cache_write_5m == price.input * Decimal("1.25"), name
        assert price.cache_write_1h == price.input * 2, name
        assert price.cache_read == price.input * Decimal("0.1"), name
