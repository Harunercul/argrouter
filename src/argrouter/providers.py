"""Ayni model icin saglayici secimi.

Ayni model ID'si OpenRouter'da 10-30 farkli saglayicidan sunulabiliyor ve
fiyatlar girdide/ciktida 3-14 kat degisiyor. Varsayilan secim girdi ve cikti
ucretlerini istegin token karisimina gore tartmiyor; cikti agirlikli bir
istekte "ucuz girdi / pahali cikti" saglayicisi secilebiliyor.

Bu modul, istegin BEKLENEN maliyetine gore siralar ve politikayi (quantization,
uptime, baglam, parametre destegi) sert filtre olarak uygular. Elenen her
saglayicinin nedeni kaydedilir.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from argrouter.catalog.pricing import CacheRates, CostBreakdown, ModelPrice, expected_cost

# Yuksek = daha az kayipli. Bilinmeyen en altta: varsayilan politika elemesi icin.
QUANT_RANK: dict[str, int] = {
    "fp32": 5,
    "bf16": 4,
    "fp16": 4,
    "fp8": 3,
    "int8": 3,
    "fp6": 2,
    "fp4": 1,
    "nvfp4": 1,
    "int4": 1,
    "unknown": 0,
}


class NoEligibleProvider(Exception):
    """Politikaya uyan saglayici yok. Sessizce politika disina cikilmaz."""


@dataclass(frozen=True)
class Endpoint:
    model_id: str
    provider_name: str
    tag: str  # OpenRouter provider.order icin kullanilan kimlik
    quantization: str
    input_per_mtok: float
    output_per_mtok: float
    context_length: int
    max_completion_tokens: int | None = None
    cache_read_per_mtok: float | None = None
    uptime_30m: float | None = None
    uptime_1d: float | None = None
    status: int = 0
    supported_parameters: tuple[str, ...] = ()

    @property
    def quant_rank(self) -> int:
        return QUANT_RANK.get(self.quantization.lower(), 0)

    def as_price(self) -> ModelPrice:
        cache = None
        if self.cache_read_per_mtok and self.input_per_mtok > 0:
            cache = CacheRates(read_multiplier=self.cache_read_per_mtok / self.input_per_mtok)
        return ModelPrice(
            model_id=f"{self.model_id}@{self.tag}",
            input_per_mtok=self.input_per_mtok,
            output_per_mtok=self.output_per_mtok,
            context_length=self.context_length,
            cache=cache,
        )

    @classmethod
    def from_openrouter(cls, raw: dict[str, Any]) -> Endpoint:
        p = raw.get("pricing") or {}

        def per_m(k: str) -> float:
            try:
                return float(p.get(k) or 0) * 1_000_000
            except (TypeError, ValueError):
                return 0.0

        cache = per_m("input_cache_read")
        return cls(
            model_id=raw["model_id"],
            provider_name=str(raw.get("provider_name") or "?"),
            tag=str(raw.get("tag") or ""),
            quantization=str(raw.get("quantization") or "unknown"),
            input_per_mtok=per_m("prompt"),
            output_per_mtok=per_m("completion"),
            context_length=int(raw.get("context_length") or 0),
            max_completion_tokens=raw.get("max_completion_tokens"),
            cache_read_per_mtok=cache or None,
            uptime_30m=raw.get("uptime_last_30m"),
            uptime_1d=raw.get("uptime_last_1d"),
            status=int(raw.get("status") or 0),
            supported_parameters=tuple(raw.get("supported_parameters") or ()),
        )


@dataclass(frozen=True)
class Policy:
    """Sert filtreler. Maliyet bunlarin ICINDE optimize edilir, onlarla takas edilmez."""

    min_quantization: str = "fp8"
    allow_unknown_quantization: bool = False
    min_uptime: float = 99.0
    required_parameters: tuple[str, ...] = ()


@dataclass
class Candidate:
    endpoint: Endpoint
    cost: CostBreakdown


@dataclass
class Selection:
    model_id: str
    ranked: list[Candidate]
    rejected: list[tuple[Endpoint, str]] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Candidate:
        return self.ranked[0]

    def openrouter_provider(self, fallbacks: int | None = None) -> dict[str, Any]:
        """OpenRouter `provider` parametresi.

        Uygun saglayicilar maliyet sirasiyla; `fallbacks` verilmezse hepsi.
        allow_fallbacks=False: biri duserse (429, kesinti) listede sonrakine
        gecer, ama politika disindaki bir saglayiciya ASLA gitmez. Listeyi
        kisa tutmak en ucuz saglayici sikistiginda istegi basarisiz kilar.
        """
        ranked = self.ranked if fallbacks is None else self.ranked[: max(1, fallbacks)]
        return {
            "order": [c.endpoint.tag for c in ranked],
            "allow_fallbacks": False,
        }

    def explain(self) -> list[str]:
        lines = [f"{self.model_id}: {len(self.ranked)} uygun, {len(self.rejected)} elendi"]
        for c in self.ranked[:5]:
            e = c.endpoint
            lines.append(f"  ✓ ${c.cost.total_usd:.8f}  {e.tag:<28} {e.quantization}")
        for e, why in self.rejected:
            lines.append(f"  ✗ {e.tag:<28} {why}")
        return lines


def _reject_reason(e: Endpoint, policy: Policy, need_ctx: int, need_out: float) -> str | None:
    if e.status != 0:
        return f"status={e.status}"
    if e.input_per_mtok <= 0 and e.output_per_mtok <= 0:
        return "fiyat_yok"  # fiyatsiz saglayici asla 'en ucuz' sayilmaz
    q = e.quantization.lower()
    if q in ("unknown", "") and not policy.allow_unknown_quantization:
        return "quantization_bilinmiyor"
    if q not in ("unknown", "") and e.quant_rank < QUANT_RANK.get(policy.min_quantization, 0):
        return f"quantization<{policy.min_quantization} ({e.quantization})"
    up = e.uptime_30m if e.uptime_30m is not None else e.uptime_1d
    if up is None:
        return "uptime_bilinmiyor"
    if up < policy.min_uptime:
        return f"uptime {up:.1f}<{policy.min_uptime}"
    if e.context_length and need_ctx > e.context_length:
        return f"baglam {need_ctx}>{e.context_length}"
    if e.max_completion_tokens and need_out > e.max_completion_tokens:
        return f"max_cikti {int(need_out)}>{e.max_completion_tokens}"
    missing = [p for p in policy.required_parameters if p not in e.supported_parameters]
    if missing:
        return f"desteklenmiyor: {','.join(missing)}"
    return None


def select(
    endpoints: Iterable[Endpoint],
    *,
    input_tokens: int,
    expected_output_tokens: float,
    cached_input_tokens: int = 0,
    policy: Policy | None = None,
) -> Selection:
    policy = policy or Policy()
    eps: Sequence[Endpoint] = list(endpoints)
    if not eps:
        raise NoEligibleProvider("hic endpoint verilmedi")
    model_id = eps[0].model_id
    need_ctx = input_tokens + int(expected_output_tokens)

    ranked: list[Candidate] = []
    rejected: list[tuple[Endpoint, str]] = []
    for e in eps:
        why = _reject_reason(e, policy, need_ctx, expected_output_tokens)
        if why:
            rejected.append((e, why))
            continue
        cost = expected_cost(
            e.as_price(),
            input_tokens=input_tokens,
            expected_output_tokens=expected_output_tokens,
            cached_input_tokens=cached_input_tokens if e.cache_read_per_mtok else 0,
        )
        ranked.append(Candidate(e, cost))

    # Esitlikte: daha az kayipli quantization, sonra daha yuksek uptime.
    ranked.sort(
        key=lambda c: (
            c.cost.total_usd,
            -c.endpoint.quant_rank,
            -(c.endpoint.uptime_30m or 0.0),
        )
    )
    sel = Selection(
        model_id,
        ranked,
        rejected,
        inputs={
            "input_tokens": input_tokens,
            "expected_output_tokens": expected_output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "policy": policy.__dict__,
        },
    )
    if not ranked:
        raise NoEligibleProvider(
            f"{model_id}: politikaya uyan saglayici yok ({len(rejected)} elendi). "
            + "; ".join(f"{e.tag}: {w}" for e, w in rejected[:6])
        )
    return sel
