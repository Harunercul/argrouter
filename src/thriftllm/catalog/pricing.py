"""Model fiyat katalogu.

Tasarim kararlari ve her birinin gerekcesi (rakip incelemesinden):

1. ICE AKTARIMDA AG CAGRISI YOK. litellm `__init__.py` icinde bloklayan bir
   fetch yapiyor; import suresini aga baglamak kabul edilemez.

2. FIYATI BILINMEYEN MODEL BIR SAYI DEGIL, ALARM DURUMUDUR.
   Uc bagimsiz projede ayni sessiz hata var: fiyatsiz modele 0.0 verilip
   "en ucuz" secilmesi. Burada `UnpricedModel` yukseltilir.

3. GIRDI VE CIKTI FIYATI TOPLANMAZ.
   `input_rate + output_rate` siralamasi 1:1 oran varsayar. 20k girdi/500 cikti
   bir RAG cagrisinda bu siralama yanlistir. Beklenen maliyet, beklenen
   token sayilariyla agirliklandirilir.

4. CACHE VE KATMANLI FIYAT KARAR ANINDA HESABA KATILIR, sonradan degil.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


class UnpricedModel(Exception):
    """Fiyati bilinmeyen model. Asla sessizce 0 veya varsayilan sayi kullanma."""


class PricingError(Exception):
    """Katalog tutarsiz veya eksik."""


@dataclass(frozen=True)
class ContextTier:
    """Baglam uzunluguna gore degisen fiyat (orn. 200k ustu farkli ucret)."""

    up_to_tokens: int | None  # None = ust sinir yok
    input_per_mtok: float
    output_per_mtok: float


@dataclass(frozen=True)
class CacheRates:
    """Prompt cache carpanlari. Modele gore degisir - sabit 0.1x varsayimi yanlis."""

    read_multiplier: float = 0.1  # Opus 5.5'te 0.05, Fable 5.1'de 0.025
    write_multiplier: float = 1.25  # 5 dk TTL
    write_multiplier_1h: float | None = 2.0
    min_cacheable_tokens: int = 1024  # altinda sessizce cache'lenmez


@dataclass(frozen=True)
class ModelPrice:
    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    context_length: int
    cache: CacheRates | None = None
    context_tiers: tuple[ContextTier, ...] = ()
    reasoning_per_mtok: float | None = None  # None => output ucretinden
    per_request_usd: float = 0.0
    batch_discount: float | None = None  # orn. 0.5 = %50 indirim
    quality_index: float | None = None  # OpenRouter artificial_analysis
    supports_logprobs: bool = False
    supports_tools: bool = False

    def _rates_for(self, input_tokens: int) -> tuple[float, float]:
        """Baglam katmanina gore gecerli girdi/cikti ucreti."""
        for tier in self.context_tiers:
            if tier.up_to_tokens is None or input_tokens <= tier.up_to_tokens:
                return tier.input_per_mtok, tier.output_per_mtok
        return self.input_per_mtok, self.output_per_mtok


@dataclass
class CostBreakdown:
    """Her bilesen ayri - "nereden geldi" sorusu cevaplanabilir olmali."""

    model_id: str
    fresh_input_usd: float = 0.0
    cached_input_usd: float = 0.0
    cache_write_usd: float = 0.0
    output_usd: float = 0.0
    reasoning_usd: float = 0.0
    per_request_usd: float = 0.0
    batch_discount_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return (
            self.fresh_input_usd
            + self.cached_input_usd
            + self.cache_write_usd
            + self.output_usd
            + self.reasoning_usd
            + self.per_request_usd
            - self.batch_discount_usd
        )

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = dict(self.__dict__)
        d["total_usd"] = self.total_usd
        return d


def expected_cost(
    price: ModelPrice,
    *,
    input_tokens: int,
    expected_output_tokens: float,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    expected_reasoning_tokens: float = 0.0,
    batch: bool = False,
) -> CostBreakdown:
    """Bir istegin beklenen maliyeti.

    `expected_output_tokens` float'tir ve kasitlidir: bu bir TAHMINDIR,
    YAML'a yazilan sabit degil. Faturanin %60-90'i cikti tarafindadir ve
    bu alandaki projelerin hicbiri bunu tahmin etmiyor.
    """
    if input_tokens < 0 or expected_output_tokens < 0:
        raise PricingError("token sayilari negatif olamaz")

    in_rate, out_rate = price._rates_for(input_tokens)
    bd = CostBreakdown(model_id=price.model_id)

    fresh_input = max(0, input_tokens - cached_input_tokens)
    bd.fresh_input_usd = fresh_input * in_rate / 1_000_000

    if cached_input_tokens:
        if price.cache is None:
            bd.notes.append("cache_tokens_verildi_ama_model_cache_fiyati_yok")
            bd.fresh_input_usd += cached_input_tokens * in_rate / 1_000_000
        elif cached_input_tokens < price.cache.min_cacheable_tokens:
            # Minimum altinda SESSIZCE cache'lenmez - bu tuzagi acikca isaretle
            bd.notes.append(
                f"cache_minimumun_altinda({cached_input_tokens}<"
                f"{price.cache.min_cacheable_tokens})_tam_ucret"
            )
            bd.fresh_input_usd += cached_input_tokens * in_rate / 1_000_000
        else:
            bd.cached_input_usd = (
                cached_input_tokens * in_rate * price.cache.read_multiplier / 1_000_000
            )

    if cache_write_tokens:
        if price.cache is None:
            raise PricingError(f"{price.model_id}: cache yazimi istendi, cache fiyati yok")
        bd.cache_write_usd = cache_write_tokens * in_rate * price.cache.write_multiplier / 1_000_000

    bd.output_usd = expected_output_tokens * out_rate / 1_000_000

    if expected_reasoning_tokens:
        rate = price.reasoning_per_mtok if price.reasoning_per_mtok is not None else out_rate
        bd.reasoning_usd = expected_reasoning_tokens * rate / 1_000_000

    bd.per_request_usd = price.per_request_usd

    if batch:
        if price.batch_discount is None:
            bd.notes.append("batch_istendi_ama_model_batch_desteklemiyor")
        else:
            subtotal = (
                bd.fresh_input_usd
                + bd.cached_input_usd
                + bd.cache_write_usd
                + bd.output_usd
                + bd.reasoning_usd
            )
            bd.batch_discount_usd = subtotal * price.batch_discount

    return bd


def cost_per_quality_point(bd: CostBreakdown, price: ModelPrice) -> float:
    """Siralama birimi ham dolar degil, $/kalite puani.

    Kalite skoru yoksa hata verilir - sessizce "sonsuz iyi" sayilmaz.
    """
    if price.quality_index is None:
        raise UnpricedModel(f"{price.model_id}: kalite skoru yok, siralanamaz")
    if price.quality_index <= 0:
        raise PricingError(f"{price.model_id}: gecersiz kalite skoru {price.quality_index}")
    return bd.total_usd / price.quality_index


class Catalog:
    """Fiyat katalogu. Yerel anlik goruntuden yuklenir, ice aktarimda ag yok."""

    def __init__(self, prices: dict[str, ModelPrice], *, source: str, as_of: str):
        self._prices = prices
        self.source = source
        self.as_of = as_of

    def __len__(self) -> int:
        return len(self._prices)

    def __iter__(self) -> Iterator[ModelPrice]:
        return iter(self._prices.values())

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._prices

    def get(self, model_id: str) -> ModelPrice:
        """Fiyat yoksa YUKSELT. Varsayilan sayi dondurmek sessiz hataya yol acar."""
        try:
            return self._prices[model_id]
        except KeyError:
            raise UnpricedModel(
                f"'{model_id}' katalogda yok (kaynak={self.source}, tarih={self.as_of}). "
                "Fiyatsiz model yonlendirmeye dahil edilemez."
            ) from None

    def priced(self) -> list[ModelPrice]:
        return list(self._prices.values())

    def with_quality(self) -> list[ModelPrice]:
        return [p for p in self._prices.values() if p.quality_index is not None]

    def frontier(self, *, in_out_ratio: float = 3.0) -> list[ModelPrice]:
        """Maliyet/kalite Pareto siniri: her kalite seviyesinde en ucuz model.

        `in_out_ratio` harmanlanmis fiyat icin varsayilan girdi:cikti orani.
        Gercek yonlendirmede bu tahmin edilir; burada sadece siralama icin.
        """
        w_in = in_out_ratio / (in_out_ratio + 1.0)
        w_out = 1.0 - w_in
        scored: list[tuple[float, float, ModelPrice]] = [
            (p.quality_index, p.input_per_mtok * w_in + p.output_per_mtok * w_out, p)
            for p in self.with_quality()
            if p.quality_index is not None  # with_quality() garanti ediyor; mypy icin acik
        ]
        scored.sort(key=lambda r: (-r[0], r[1]))
        out: list[ModelPrice] = []
        cheapest = math.inf
        for _q, blended, p in scored:
            if blended < cheapest:
                out.append(p)
                cheapest = blended
        return out

    @classmethod
    def from_snapshot(cls, path: str | Path) -> Catalog:
        """Vendorlanmis anlik goruntuden yukle. Uretim yolu budur."""
        data = json.loads(Path(path).read_text())
        prices: dict[str, ModelPrice] = {}
        for row in data["models"]:
            cache = CacheRates(**row["cache"]) if row.get("cache") else None
            tiers = tuple(ContextTier(**t) for t in row.get("context_tiers", []))
            prices[row["model_id"]] = ModelPrice(
                **{k: v for k, v in row.items() if k not in ("cache", "context_tiers")},
                cache=cache,
                context_tiers=tiers,
            )
        return cls(prices, source=data["source"], as_of=data["as_of"])
