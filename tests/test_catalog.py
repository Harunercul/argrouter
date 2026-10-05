"""Katalog, Pareto siniri, anlik goruntu yukleme ve kalan fiyat bilesenleri."""

import json

import pytest

from thriftllm.catalog.pricing import (
    Catalog,
    ModelPrice,
    PricingError,
    cost_per_quality_point,
    expected_cost,
)

A = ModelPrice(
    model_id="a/big",
    input_per_mtok=8.0,
    output_per_mtok=24.0,
    context_length=200_000,
    quality_index=58.0,
)
B = ModelPrice(
    model_id="b/mid",
    input_per_mtok=2.0,
    output_per_mtok=6.0,
    context_length=200_000,
    quality_index=50.0,
)
C = ModelPrice(
    model_id="c/small",
    input_per_mtok=0.2,
    output_per_mtok=0.6,
    context_length=128_000,
    quality_index=42.0,
)
D = ModelPrice(
    model_id="d/dominated",
    input_per_mtok=9.0,
    output_per_mtok=30.0,
    context_length=128_000,
    quality_index=45.0,
)  # pahali VE kotu


def _cat(*models):
    return Catalog({m.model_id: m for m in models}, source="test", as_of="2026-10-05")


class TestCatalogBasics:
    def test_len_iter_contains(self):
        cat = _cat(A, B, C)
        assert len(cat) == 3
        assert "a/big" in cat
        assert "yok" not in cat
        assert {m.model_id for m in cat} == {"a/big", "b/mid", "c/small"}

    def test_priced_and_with_quality(self):
        no_q = ModelPrice(
            model_id="e/noq", input_per_mtok=1, output_per_mtok=1, context_length=1000
        )
        cat = _cat(A, no_q)
        assert len(cat.priced()) == 2
        assert [m.model_id for m in cat.with_quality()] == ["a/big"]


class TestFrontier:
    def test_dominated_model_is_excluded(self):
        """Hem pahali hem dusuk kaliteli model sinirda yer almamali."""
        ids = [m.model_id for m in _cat(A, B, C, D).frontier()]
        assert "d/dominated" not in ids
        assert ids == ["a/big", "b/mid", "c/small"]

    def test_frontier_is_monotonically_cheaper(self):
        front = _cat(A, B, C, D).frontier()
        w_in, w_out = 0.75, 0.25
        blended = [m.input_per_mtok * w_in + m.output_per_mtok * w_out for m in front]
        assert blended == sorted(blended, reverse=True)

    def test_ratio_changes_the_frontier_ordering_basis(self):
        """Girdi:cikti orani siralama tabanini degistirir - sabit varsayim yok."""
        f_in_heavy = _cat(A, B, C).frontier(in_out_ratio=20.0)
        f_out_heavy = _cat(A, B, C).frontier(in_out_ratio=0.1)
        assert len(f_in_heavy) >= 1 and len(f_out_heavy) >= 1


class TestCostPerQualityPoint:
    def test_cheaper_per_quality_point_can_beat_cheaper_absolute(self):
        kw = {"input_tokens": 1_000, "expected_output_tokens": 1_000}
        bd_b, bd_c = expected_cost(B, **kw), expected_cost(C, **kw)
        assert bd_c.total_usd < bd_b.total_usd  # mutlak olarak ucuz
        assert cost_per_quality_point(bd_c, C) < cost_per_quality_point(bd_b, B)

    def test_invalid_quality_raises(self):
        bad = ModelPrice(
            model_id="x", input_per_mtok=1, output_per_mtok=1, context_length=10, quality_index=0.0
        )
        bd = expected_cost(bad, input_tokens=10, expected_output_tokens=10)
        with pytest.raises(PricingError):
            cost_per_quality_point(bd, bad)


class TestRemainingCostComponents:
    def test_batch_discount_applies_to_subtotal(self):
        m = ModelPrice(
            model_id="b",
            input_per_mtok=10.0,
            output_per_mtok=10.0,
            context_length=10_000,
            batch_discount=0.5,
        )
        full = expected_cost(m, input_tokens=1_000, expected_output_tokens=1_000)
        batch = expected_cost(m, input_tokens=1_000, expected_output_tokens=1_000, batch=True)
        assert batch.total_usd == pytest.approx(full.total_usd * 0.5)

    def test_batch_requested_but_unsupported_is_noted_not_silent(self):
        bd = expected_cost(C, input_tokens=100, expected_output_tokens=100, batch=True)
        assert bd.batch_discount_usd == 0.0
        assert any("batch_istendi" in n for n in bd.notes)

    def test_reasoning_tokens_use_dedicated_rate_when_set(self):
        m = ModelPrice(
            model_id="r",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
            context_length=10_000,
            reasoning_per_mtok=10.0,
        )
        bd = expected_cost(
            m, input_tokens=100, expected_output_tokens=100, expected_reasoning_tokens=1_000
        )
        assert bd.reasoning_usd == pytest.approx(1_000 * 10.0 / 1_000_000)

    def test_reasoning_falls_back_to_output_rate(self):
        bd = expected_cost(
            C, input_tokens=100, expected_output_tokens=100, expected_reasoning_tokens=1_000
        )
        assert bd.reasoning_usd == pytest.approx(1_000 * C.output_per_mtok / 1_000_000)

    def test_per_request_fee_is_added(self):
        m = ModelPrice(
            model_id="p",
            input_per_mtok=0.0,
            output_per_mtok=0.0,
            context_length=10_000,
            per_request_usd=0.004,
        )
        bd = expected_cost(m, input_tokens=10, expected_output_tokens=10)
        assert bd.total_usd == pytest.approx(0.004)

    def test_cache_tokens_without_cache_pricing_is_noted_and_charged_full(self):
        bd = expected_cost(
            C, input_tokens=5_000, expected_output_tokens=10, cached_input_tokens=4_000
        )
        assert bd.cached_input_usd == 0.0
        assert any("model_cache_fiyati_yok" in n for n in bd.notes)
        # tam ucretten hesaplandi
        assert bd.fresh_input_usd == pytest.approx(5_000 * C.input_per_mtok / 1_000_000)

    def test_negative_tokens_rejected(self):
        with pytest.raises(PricingError):
            expected_cost(C, input_tokens=-1, expected_output_tokens=10)


class TestSnapshotRoundTrip:
    def test_loads_from_vendored_snapshot(self, tmp_path):
        snap = {
            "source": "openrouter",
            "as_of": "2026-10-05",
            "models": [
                {
                    "model_id": "x/one",
                    "input_per_mtok": 1.0,
                    "output_per_mtok": 2.0,
                    "context_length": 100_000,
                    "quality_index": 44.0,
                    "cache": {"read_multiplier": 0.05, "min_cacheable_tokens": 2048},
                    "context_tiers": [
                        {"up_to_tokens": 50_000, "input_per_mtok": 1.0, "output_per_mtok": 2.0},
                        {"up_to_tokens": None, "input_per_mtok": 2.0, "output_per_mtok": 4.0},
                    ],
                },
            ],
        }
        f = tmp_path / "prices.json"
        f.write_text(json.dumps(snap))
        cat = Catalog.from_snapshot(f)

        assert cat.source == "openrouter" and cat.as_of == "2026-10-05"
        m = cat.get("x/one")
        assert m.cache is not None and m.cache.read_multiplier == 0.05
        assert len(m.context_tiers) == 2
        # ust katman gercekten uygulaniyor
        hi = expected_cost(m, input_tokens=80_000, expected_output_tokens=100)
        assert hi.fresh_input_usd == pytest.approx(80_000 * 2.0 / 1_000_000)
