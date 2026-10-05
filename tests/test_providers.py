"""Saglayici secimi. Gercek bir faturada gozlenen hatanin testi dahil."""

import pytest

from thriftllm.providers import Endpoint, NoEligibleProvider, Policy, select


def ep(tag, inp, out, quant="fp8", up=99.5, **kw):
    return Endpoint(
        model_id="m/x",
        provider_name=tag.split("/")[0],
        tag=tag,
        quantization=quant,
        input_per_mtok=inp,
        output_per_mtok=out,
        context_length=kw.pop("ctx", 128_000),
        uptime_30m=up,
        **kw,
    )


# 2026-10-05 glm-5.3-flash faturasindan: servis eden ucuz-girdi/pahali-cikti idi.
RELACE = ep("relace/fp8", 0.035, 0.50)
STREAMLAK = ep("streamlake/fp8", 0.087, 0.29)
DEEPINFRA = ep("deepinfra/fp4", 0.075, 0.25, quant="fp4")
NOVITA = ep("novita/fp8", 0.084, 0.28, up=85.4)


class TestTokenMixDecides:
    def test_output_heavy_request_avoids_cheap_input_provider(self):
        """31 girdi / 197 cikti: gercekte servis edilen Relace burada yanlis secim."""
        sel = select([RELACE, STREAMLAK], input_tokens=31, expected_output_tokens=197)
        assert sel.best.endpoint is STREAMLAK
        relace_cost = next(c for c in sel.ranked if c.endpoint is RELACE).cost.total_usd
        assert sel.best.cost.total_usd < relace_cost * 0.7  # >%30 ucuz

    def test_input_heavy_request_flips_to_cheap_input_provider(self):
        sel = select([RELACE, STREAMLAK], input_tokens=20_000, expected_output_tokens=100)
        assert sel.best.endpoint is RELACE


class TestPolicyIsAHardFilter:
    def test_fp4_rejected_under_default_policy_with_reason(self):
        sel = select([DEEPINFRA, STREAMLAK], input_tokens=31, expected_output_tokens=197)
        assert sel.best.endpoint is STREAMLAK
        assert any(e is DEEPINFRA and "quantization" in why for e, why in sel.rejected)

    def test_fp4_allowed_when_policy_relaxed(self):
        sel = select(
            [DEEPINFRA, STREAMLAK],
            input_tokens=31,
            expected_output_tokens=197,
            policy=Policy(min_quantization="fp4"),
        )
        assert sel.best.endpoint is DEEPINFRA

    def test_low_uptime_rejected(self):
        sel = select([NOVITA, STREAMLAK], input_tokens=31, expected_output_tokens=197)
        assert all(c.endpoint is not NOVITA for c in sel.ranked)
        assert any(e is NOVITA and "uptime" in why for e, why in sel.rejected)

    def test_unknown_quantization_rejected_by_default(self):
        unk = ep("mystery/x", 0.001, 0.001, quant="unknown")
        sel = select([unk, STREAMLAK], input_tokens=10, expected_output_tokens=10)
        assert sel.best.endpoint is STREAMLAK

    def test_unpriced_provider_never_wins(self):
        free = ep("free/fp8", 0.0, 0.0)
        sel = select([free, STREAMLAK], input_tokens=10, expected_output_tokens=10)
        assert sel.best.endpoint is STREAMLAK
        assert any(e is free and "fiyat" in why for e, why in sel.rejected)

    def test_context_and_max_completion_limits(self):
        small = ep("small/fp8", 0.01, 0.01, ctx=4_000)
        capped = ep("capped/fp8", 0.01, 0.01, max_completion_tokens=100)
        sel = select([small, capped, STREAMLAK], input_tokens=3_900, expected_output_tokens=500)
        assert sel.best.endpoint is STREAMLAK
        reasons = {e.tag: w for e, w in sel.rejected}
        assert "baglam" in reasons["small/fp8"]
        assert "max_cikti" in reasons["capped/fp8"]

    def test_required_parameter(self):
        tools = ep("tools/fp8", 0.2, 0.6, supported_parameters=("tools",))
        sel = select(
            [STREAMLAK, tools],
            input_tokens=10,
            expected_output_tokens=10,
            policy=Policy(required_parameters=("tools",)),
        )
        assert sel.best.endpoint is tools

    def test_nothing_eligible_raises_instead_of_falling_back(self):
        with pytest.raises(NoEligibleProvider) as e:
            select([DEEPINFRA, NOVITA], input_tokens=10, expected_output_tokens=10)
        assert "deepinfra/fp4" in str(e.value)

    def test_empty_input_raises(self):
        with pytest.raises(NoEligibleProvider):
            select([], input_tokens=1, expected_output_tokens=1)


class TestOpenRouterParams:
    def test_order_is_cost_sorted_eligible_only_no_escape(self):
        sel = select(
            [RELACE, STREAMLAK, DEEPINFRA, NOVITA], input_tokens=31, expected_output_tokens=197
        )
        p = sel.openrouter_provider(fallbacks=3)
        assert p["allow_fallbacks"] is False
        assert p["order"] == ["streamlake/fp8", "relace/fp8"]  # fp4 ve dusuk uptime yok
        assert "deepinfra/fp4" not in p["order"]

    def test_default_order_keeps_every_eligible_endpoint(self):
        sel = select(
            [RELACE, STREAMLAK, DEEPINFRA, NOVITA], input_tokens=31, expected_output_tokens=197
        )
        assert sel.openrouter_provider()["order"] == [c.endpoint.tag for c in sel.ranked]
        assert sel.openrouter_provider(fallbacks=1)["order"] == ["streamlake/fp8"]

    def test_explain_lists_rejections(self):
        sel = select([RELACE, STREAMLAK, DEEPINFRA], input_tokens=31, expected_output_tokens=197)
        text = "\n".join(sel.explain())
        assert "✗ deepinfra/fp4" in text and "✓" in text


class TestFromOpenRouter:
    def test_parses_real_endpoint_shape(self):
        raw = {
            "model_id": "z-ai/glm-5.3-flash",
            "provider_name": "StreamLake",
            "tag": "streamlake/fp8",
            "quantization": "fp8",
            "context_length": 1024000,
            "pricing": {
                "prompt": "0.000000087",
                "completion": "0.00000029",
                "input_cache_read": "0.0000000174",
            },
            "max_completion_tokens": 128000,
            "status": 0,
            "uptime_last_30m": 99.39,
            "uptime_last_1d": 99.2,
            "supported_parameters": ["max_tokens", "tools"],
        }
        e = Endpoint.from_openrouter(raw)
        assert e.input_per_mtok == pytest.approx(0.087)
        assert e.output_per_mtok == pytest.approx(0.29)
        assert e.as_price().cache is not None
        assert e.as_price().cache.read_multiplier == pytest.approx(0.2)
        assert "tools" in e.supported_parameters

    def test_cached_tokens_reduce_cost_when_provider_caches(self):
        e = Endpoint.from_openrouter(
            {
                "model_id": "m",
                "tag": "a/fp8",
                "quantization": "fp8",
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000001",
                    "input_cache_read": "0.0000001",
                },
                "context_length": 100000,
                "uptime_last_30m": 100,
            }
        )
        cold = select([e], input_tokens=10_000, expected_output_tokens=10)
        warm = select(
            [e], input_tokens=10_000, expected_output_tokens=10, cached_input_tokens=9_000
        )
        assert warm.best.cost.total_usd < cold.best.cost.total_usd
