"""Cikti uzunlugu tahmini.

Cikti tokenlari faturanin cogunu olusturuyor ama bu alandaki projeler ya sabit
bir sayi kullaniyor (YAML'a yazilan `expected_output_tokens`) ya da hic
hesaba katmiyor. Buradaki yaklasim kasitli olarak basit: istek turune (route)
gore gozlenen cikti uzunluklarini tut, medyani dondur. Ayni route'un ciktilari
birbirine benzer; bu, dagilimi ogrenmenin en ucuz yolu.

Soguk baslangicta tahmin `source="prior"` ile isaretlenir - karar kaydinda
tahminin gozleme mi yoksa varsayima mi dayandigi gorunur.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Forecast:
    tokens: float
    source: str  # "observed" | "prior"
    samples: int
    p90: float | None = None


class OutputForecaster:
    def __init__(
        self, *, prior_tokens: float = 400.0, window: int = 200, min_samples: int = 3
    ) -> None:
        if prior_tokens < 0 or window < 1 or min_samples < 1:
            raise ValueError("gecersiz parametre")
        self.prior_tokens = prior_tokens
        self.window = window
        self.min_samples = min_samples
        self._obs: dict[str, deque[int]] = {}

    def observe(self, route: str, output_tokens: int) -> None:
        if output_tokens < 0:
            raise ValueError("cikti token sayisi negatif olamaz")
        self._obs.setdefault(route, deque(maxlen=self.window)).append(output_tokens)

    def predict(self, route: str, *, max_tokens: int | None = None) -> Forecast:
        xs = self._obs.get(route)
        if not xs or len(xs) < self.min_samples:
            t = self.prior_tokens if max_tokens is None else min(self.prior_tokens, max_tokens)
            return Forecast(tokens=float(t), source="prior", samples=len(xs or ()))
        ordered = sorted(xs)
        med = float(statistics.median(ordered))
        p90 = float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))])
        if max_tokens is not None:
            med, p90 = min(med, max_tokens), min(p90, max_tokens)
        return Forecast(tokens=med, source="observed", samples=len(ordered), p90=p90)
