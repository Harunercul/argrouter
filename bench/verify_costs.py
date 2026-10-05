"""Fiyat formulumuzu saglayicinin gercek faturasina karsi dogrular.

Tahmini degil GERCEK token sayilarini kullanir: boylece olctugumuz sey
cikti-uzunlugu tahmini degil, fiyat tablosu + formulun dogrulugudur.
Gercek maliyet OpenRouter'in `usage.cost` alanindan gelir (faturalanan tutar).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

from thriftllm.catalog.pricing import Catalog, expected_cost

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().split() if "=" in line)
KEY = ENV["OPENROUTER_API_KEY"]

CANDIDATES = [
    "openai/gpt-oss-20b", "z-ai/glm-5.3-flash", "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash", "openai/gpt-5-mini", "deepseek/deepseek-v4-flash-0731",
    "mistralai/mistral-small-3.2-24b-instruct", "qwen/qwen3-235b-a22b-2507",
]
PROMPT = ("In exactly three short sentences, explain why prompt caching reduces "
          "the cost of repeated LLM requests.")


def call(model: str) -> dict:
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEY}"},
        json={"model": model, "max_tokens": 300,
              "messages": [{"role": "user", "content": PROMPT}],
              "usage": {"include": True}},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    cat = Catalog.from_snapshot(ROOT / "src/thriftllm/catalog/prices.json")
    models = [m for m in CANDIDATES if m in cat][:5]
    print(f"katalog {cat.as_of} | test edilecek: {len(models)} model\n")
    print(f"{'model':40s} {'in':>5} {'out':>5} {'bizim $':>11} {'fatura $':>11} {'fark':>8}")
    rows, total = [], 0.0
    for mid in models:
        try:
            resp = call(mid)
        except Exception as e:
            print(f"{mid:40s}  HATA: {str(e)[:60]}")
            continue
        u = resp.get("usage") or {}
        pin, pout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        billed = u.get("cost")
        served = resp.get("model", "?")
        ours = expected_cost(cat.get(mid), input_tokens=pin,
                             expected_output_tokens=pout,
                             cached_input_tokens=cached).total_usd
        diff = (ours - billed) / billed * 100 if billed else float("nan")
        total += billed or 0
        print(f"{mid:40s} {pin:>5} {pout:>5} {ours:>11.8f} {billed or 0:>11.8f} {diff:>+7.1f}%")
        rows.append({"model": mid, "served": served, "provider": resp.get("provider"),
                     "prompt_tokens": pin, "completion_tokens": pout,
                     "cached_tokens": cached, "ours_usd": ours, "billed_usd": billed,
                     "usage": u})
    out = ROOT / "bench" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify_costs.json").write_text(json.dumps(rows, indent=2))
    print(f"\ntoplam harcama: ${total:.6f}  | ham kayit: bench/runs/verify_costs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
