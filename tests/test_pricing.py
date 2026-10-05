"""Projenin tezini kanitlayan testler.

En onemlisi `test_naive_ranking_is_wrong`: rakiplerin kullandigi
`input_rate + output_rate` siralamasinin gercek is yuklerinde yanlis
model sectigini gosterir. Bu test projenin varlik sebebidir.
"""
import pytest

from thriftllm.catalog.pricing import (
    CacheRates, Catalog, ContextTier, ModelPrice, PricingError,
    UnpricedModel, cost_per_quality_point, expected_cost,
)

# Gercekci iki model: biri girdide ucuz/ciktida pahali, digeri tersi.
RAG_FRIENDLY = ModelPrice(                 # ucuz girdi, pahali cikti
    model_id="cheap-in/pricey-out",
    input_per_mtok=0.10, output_per_mtok=8.00,
    context_length=200_000, quality_index=40.0,
)
GEN_FRIENDLY = ModelPrice(                 # pahali girdi, ucuz cikti
    model_id="pricey-in/cheap-out",
    input_per_mtok=3.00, output_per_mtok=1.00,
    context_length=200_000, quality_index=40.0,
)


def _naive_rank(models):
    """Rakiplerin yaptigi: iki ucreti topla, sirala. 1:1 oran varsayar."""
    return sorted(models, key=lambda m: m.input_per_mtok + m.output_per_mtok)


class TestNaiveRankingIsWrong:
    """litellm lowest_cost.py:255 ve plano model_metrics.rs:33-36 bunu yapiyor."""

    def test_naive_picks_same_model_for_both_workloads(self):
        # Naif siralama is yukunden bagimsiz - her zaman ayni cevabi verir.
        assert _naive_rank([RAG_FRIENDLY, GEN_FRIENDLY])[0] is GEN_FRIENDLY
        # 3.00+1.00=4.00  <  0.10+8.00=8.10

    def test_but_rag_workload_wants_the_other_one(self):
        """20k girdi / 500 cikti - klasik RAG. Naif siralama YANLIS model secer."""
        kw = dict(input_tokens=20_000, expected_output_tokens=500)
        rag = expected_cost(RAG_FRIENDLY, **kw).total_usd
        gen = expected_cost(GEN_FRIENDLY, **kw).total_usd

        assert rag < gen, "RAG yukunde ucuz-girdi modeli kazanmali"
        # Naif siralamanin sectigi GEN_FRIENDLY burada daha pahali:
        assert _naive_rank([RAG_FRIENDLY, GEN_FRIENDLY])[0] is GEN_FRIENDLY
        assert gen / rag > 2.0, "fark onemsiz degil, 2 kattan fazla"

    def test_and_reasoning_workload_flips_it_back(self):
        """2k girdi / 8k cikti - reasoning. Simdi diger model dogru."""
        kw = dict(input_tokens=2_000, expected_output_tokens=8_000)
        rag = expected_cost(RAG_FRIENDLY, **kw).total_usd
        gen = expected_cost(GEN_FRIENDLY, **kw).total_usd
        assert gen < rag, "reasoning yukunde ucuz-cikti modeli kazanmali"

    def test_the_point(self):
        """Dogru model is yukune bagli. Sabit siralama ikisinde de dogru olamaz."""
        rag_kw = dict(input_tokens=20_000, expected_output_tokens=500)
        gen_kw = dict(input_tokens=2_000, expected_output_tokens=8_000)
        winner_rag = min([RAG_FRIENDLY, GEN_FRIENDLY],
                         key=lambda m: expected_cost(m, **rag_kw).total_usd)
        winner_gen = min([RAG_FRIENDLY, GEN_FRIENDLY],
                         key=lambda m: expected_cost(m, **gen_kw).total_usd)
        assert winner_rag is not winner_gen


class TestUnpricedIsAnAlarm:
    """Uc bagimsiz projede ayni sessiz hata var: fiyatsiz model 0.0 alip kazaniyor."""

    def test_missing_model_raises_not_returns_zero(self):
        cat = Catalog({"a": RAG_FRIENDLY}, source="test", as_of="2026-10-05")
        with pytest.raises(UnpricedModel) as e:
            cat.get("yok-boyle-bir-model")
        assert "yonlendirmeye dahil edilemez" in str(e.value)

    def test_missing_quality_raises_not_ranks_infinitely_good(self):
        no_q = ModelPrice(model_id="x", input_per_mtok=1, output_per_mtok=1,
                          context_length=1000)
        bd = expected_cost(no_q, input_tokens=100, expected_output_tokens=100)
        with pytest.raises(UnpricedModel):
            cost_per_quality_point(bd, no_q)


class TestCacheIsPricedAtDecisionTime:
    """vLLM SR ve Switchyard cache'i sadece sonradan hesapliyor."""

    def test_cache_read_discount_applies(self):
        m = ModelPrice(model_id="c", input_per_mtok=10.0, output_per_mtok=10.0,
                       context_length=100_000,
                       cache=CacheRates(read_multiplier=0.1, min_cacheable_tokens=1024))
        cold = expected_cost(m, input_tokens=10_000, expected_output_tokens=100)
        warm = expected_cost(m, input_tokens=10_000, expected_output_tokens=100,
                             cached_input_tokens=9_000)
        assert warm.total_usd < cold.total_usd
        assert warm.cached_input_usd > 0

    def test_below_minimum_silently_does_not_cache_and_we_say_so(self):
        """Minimum altinda cache sessizce calismaz - saglayicilar hata vermez."""
        m = ModelPrice(model_id="c", input_per_mtok=10.0, output_per_mtok=10.0,
                       context_length=100_000,
                       cache=CacheRates(min_cacheable_tokens=1024))
        bd = expected_cost(m, input_tokens=2_000, expected_output_tokens=50,
                           cached_input_tokens=500)   # 1024'un altinda
        assert bd.cached_input_usd == 0.0
        assert any("minimumun_altinda" in n for n in bd.notes)

    def test_cache_write_without_cache_pricing_raises(self):
        m = ModelPrice(model_id="n", input_per_mtok=1, output_per_mtok=1,
                       context_length=1000)
        with pytest.raises(PricingError):
            expected_cost(m, input_tokens=100, expected_output_tokens=10,
                          cache_write_tokens=100)


class TestContextTiers:
    """plano'nun parser'i `tiers` alanini tamamen dusuruyor."""

    def test_long_context_uses_the_higher_tier(self):
        m = ModelPrice(
            model_id="t", input_per_mtok=1.0, output_per_mtok=2.0,
            context_length=1_000_000,
            context_tiers=(
                ContextTier(up_to_tokens=200_000, input_per_mtok=1.0, output_per_mtok=2.0),
                ContextTier(up_to_tokens=None, input_per_mtok=2.0, output_per_mtok=4.0),
            ),
        )
        short = expected_cost(m, input_tokens=100_000, expected_output_tokens=1_000)
        long_ = expected_cost(m, input_tokens=400_000, expected_output_tokens=1_000)
        # 4x token ama 8x maliyet: katman degisimi hesaba katildi
        assert long_.fresh_input_usd / short.fresh_input_usd == pytest.approx(8.0)


class TestBreakdownIsAuditable:
    def test_every_component_is_separately_visible(self):
        m = ModelPrice(model_id="a", input_per_mtok=1.0, output_per_mtok=2.0,
                       context_length=10_000, per_request_usd=0.001,
                       cache=CacheRates())
        bd = expected_cost(m, input_tokens=5_000, expected_output_tokens=500,
                           cached_input_tokens=4_000, expected_reasoning_tokens=200)
        d = bd.as_dict()
        for key in ("fresh_input_usd", "cached_input_usd", "output_usd",
                    "reasoning_usd", "per_request_usd", "total_usd"):
            assert key in d
        parts = (bd.fresh_input_usd + bd.cached_input_usd + bd.cache_write_usd
                 + bd.output_usd + bd.reasoning_usd + bd.per_request_usd
                 - bd.batch_discount_usd)
        assert parts == pytest.approx(bd.total_usd)
