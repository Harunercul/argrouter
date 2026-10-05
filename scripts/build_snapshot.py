#!/usr/bin/env python3
"""OpenRouter katalogundan vendorlanabilir fiyat anlik goruntusu uretir.

Neden anlik goruntu:
  - Ice aktarimda ag cagrisi olmamali (litellm'in hatasi)
  - Fiyatlar degisince eski olcumler yeniden hesaplanabilmeli
  - SHA ile sabitlenip surum kontrolune girmeli

Calistir:  python scripts/build_snapshot.py -o src/thriftllm/catalog/prices.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import httpx

SOURCE_URL = "https://openrouter.ai/api/v1/models"   # anahtar gerekmez


def _f(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _per_mtok(rate_per_token) -> float:
    """OpenRouter token basina veriyor; biz milyon token basina tutuyoruz."""
    return _f(rate_per_token) * 1_000_000


def _cache_block(p: dict, base_in: float) -> dict | None:
    """Cache carpanlarini mutlak ucretlerden hesapla.

    Carpan modele gore degisiyor (0.025x - 0.1x arasi gorduk), sabit varsayim yanlis.
    """
    read = _per_mtok(p.get("input_cache_read"))
    if read <= 0 or base_in <= 0:
        return None
    out = {"read_multiplier": round(read / base_in, 6)}

    write = _per_mtok(p.get("input_cache_write"))
    if write > 0:
        out["write_multiplier"] = round(write / base_in, 6)

    write_1h = _per_mtok(p.get("input_cache_write_1h"))
    if write_1h > 0:
        out["write_multiplier_1h"] = round(write_1h / base_in, 6)

    # OpenRouter minimum cache esigini vermiyor; saglayici belgelerine gore
    # 512-4096 arasi degisiyor. Muhafazakar varsayilan, yerel dosyayla ezilebilir.
    out["min_cacheable_tokens"] = 1024
    return out


def _context_tiers(p: dict, base_in: float, base_out: float) -> list[dict]:
    """`overrides` -> baglam katmanlari.

    OpenRouter ALT sinir veriyor (min_prompt_tokens), bizim semamiz UST sinir
    kullaniyor, ceviriyoruz.
    """
    ov = p.get("overrides") or []
    if not ov:
        return []
    cuts = sorted(ov, key=lambda o: int(o.get("min_prompt_tokens") or 0))
    tiers: list[dict] = []
    prev_in, prev_out = base_in, base_out
    for o in cuts:
        edge = int(o.get("min_prompt_tokens") or 0)
        if edge <= 0:
            continue
        tiers.append({"up_to_tokens": edge,
                      "input_per_mtok": prev_in, "output_per_mtok": prev_out})
        prev_in = _per_mtok(o.get("prompt")) or prev_in
        prev_out = _per_mtok(o.get("completion")) or prev_out
    if tiers:
        tiers.append({"up_to_tokens": None,
                      "input_per_mtok": prev_in, "output_per_mtok": prev_out})
    return tiers


def build(raw: dict) -> dict:
    models: list[dict] = []
    skipped_unpriced = 0

    for m in raw["data"]:
        p = m.get("pricing") or {}
        base_in, base_out = _per_mtok(p.get("prompt")), _per_mtok(p.get("completion"))

        # Fiyatsiz model katalogda yer almaz. Sessizce 0 yazmak, bu alanin
        # en yaygin sessiz hatasi - fiyatsiz model her zaman "en ucuz" cikiyor.
        if base_in <= 0 and base_out <= 0:
            skipped_unpriced += 1
            continue

        sp = m.get("supported_parameters") or []
        bench = (m.get("benchmarks") or {}).get("artificial_analysis") or {}

        row: dict = {
            "model_id": m["id"],
            "input_per_mtok": round(base_in, 6),
            "output_per_mtok": round(base_out, 6),
            "context_length": int(m.get("context_length") or 0),
            "supports_logprobs": "logprobs" in sp,
            "supports_tools": "tools" in sp,
        }
        if (q := bench.get("intelligence_index")) is not None:
            row["quality_index"] = float(q)
        if (ir := _per_mtok(p.get("internal_reasoning"))) > 0:
            row["reasoning_per_mtok"] = round(ir, 6)
        if (ws := _f(p.get("request"))) > 0:
            row["per_request_usd"] = ws
        if (cache := _cache_block(p, base_in)):
            row["cache"] = cache
        if (tiers := _context_tiers(p, base_in, base_out)):
            row["context_tiers"] = tiers

        models.append(row)

    models.sort(key=lambda r: r["model_id"])
    payload = {
        "source": "openrouter",
        "source_url": SOURCE_URL,
        "as_of": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        "model_count": len(models),
        "skipped_unpriced": skipped_unpriced,
        "models": models,
    }
    body = json.dumps(payload["models"], sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="src/thriftllm/catalog/prices.json")
    args = ap.parse_args()

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0),
                      follow_redirects=True) as c:
        raw = c.get(SOURCE_URL, headers={"Accept": "application/json"}).raise_for_status().json()

    snap = build(raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=2, sort_keys=False) + "\n")

    with_q = sum(1 for m in snap["models"] if "quality_index" in m)
    with_c = sum(1 for m in snap["models"] if "cache" in m)
    with_t = sum(1 for m in snap["models"] if "context_tiers" in m)
    with_r = sum(1 for m in snap["models"] if "reasoning_per_mtok" in m)
    print(f"yazildi: {out}  ({out.stat().st_size:,} bayt)")
    print(f"  tarih        : {snap['as_of']}")
    print(f"  sha256       : {snap['sha256'][:16]}...")
    print(f"  fiyatli model: {snap['model_count']}  (atlanan fiyatsiz: {snap['skipped_unpriced']})")
    print(f"  kalite skoru : {with_q}")
    print(f"  cache fiyati : {with_c}")
    print(f"  baglam katmani: {with_t}")
    print(f"  reasoning ucreti: {with_r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
