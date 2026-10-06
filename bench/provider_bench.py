"""Ayni model, farkli saglayici secim yontemleri — gercek faturayla.

Kollar:
  or_default  OpenRouter varsayilan (cogu kullanicinin aldigi)
  or_price    OpenRouter sort=price + ayni quantization filtresi (piyasanin hazir cozumu)
  naive_sum   input+output fiyat toplamina gore sirala, ayni politika
              (litellm lowest_cost / plano cheapest siralama kurali; yeniden uygulama)
  thrift      beklenen maliyet + ogrenen cikti tahmini, ayni politika

Tum maliyetler OpenRouter'in faturaladigi `usage.cost`. Tahmin degil.
Calistir:  python bench/provider_bench.py --reps 4
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import statistics
import time
from pathlib import Path

import httpx

from argrouter import Endpoint, OutputForecaster, Policy, select

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().split() if "=" in line)
KEY = ENV["OPENROUTER_API_KEY"]
API = "https://openrouter.ai/api/v1"

MODELS = [
    "openai/gpt-oss-120b",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4-flash-0731",
    "qwen/qwen3-235b-a22b-2507",
]
ARMS = ["or_default", "or_price", "naive_sum", "thrift"]
POLICY = Policy(min_quantization="fp8", min_uptime=99.0)
OR_QUANTS = ["fp8", "int8", "bf16", "fp16", "fp32"]  # POLICY ile esdeger

LICENSE = (ROOT / "LICENSE").read_text()
PREREG = (ROOT / "PRE-REGISTRATION.md").read_text()[:3500]
WORKLOADS = {
    "rag": {
        "max_tokens": 200,
        "prompt": LICENSE + "\n\nQuestion: which section of this license grants patent rights, "
        "and to whom? Answer in one sentence.",
    },
    "gen": {
        "max_tokens": 1200,
        "prompt": "Write a roughly 600-word technical explanation of how prompt caching works in "
        "LLM APIs: what is cached, when it expires, how it is billed, and one worked example.",
    },
    "balanced": {
        "max_tokens": 500,
        "prompt": PREREG + "\n\nSummarise the methodology above in about 200 words.",
    },
}


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


async def endpoints(c: httpx.AsyncClient, model: str) -> list[dict]:
    r = await c.get(f"{API}/models/{model}/endpoints")
    r.raise_for_status()
    return r.json()["data"]["endpoints"]


def provider_param(arm, eps, wl, fc):
    """Her kol icin OpenRouter `provider` parametresi + karar notu."""
    ep_objs = [Endpoint.from_openrouter(e) for e in eps]
    in_tok = est_tokens(WORKLOADS[wl]["prompt"])
    if arm == "or_default":
        return None, {}
    if arm == "or_price":
        return {"sort": "price", "quantizations": OR_QUANTS}, {}
    if arm == "naive_sum":
        sel = select(ep_objs, input_tokens=in_tok, expected_output_tokens=1, policy=POLICY)
        ranked = sorted(
            sel.ranked, key=lambda c: c.endpoint.input_per_mtok + c.endpoint.output_per_mtok
        )
        return (
            {"order": [c.endpoint.tag for c in ranked], "allow_fallbacks": False},
            {"pick": ranked[0].endpoint.tag},
        )
    # thrift
    f = fc.predict(wl, max_tokens=WORKLOADS[wl]["max_tokens"])
    sel = select(ep_objs, input_tokens=in_tok, expected_output_tokens=f.tokens, policy=POLICY)
    return (
        sel.openrouter_provider(),
        {
            "pick": sel.best.endpoint.tag,
            "forecast": f.tokens,
            "forecast_src": f.source,
            "predicted_usd": sel.best.cost.total_usd,
        },
    )


async def one_call(c, sem, model, wl, arm, prov, note, rep):
    body = {
        "model": model,
        "max_tokens": WORKLOADS[wl]["max_tokens"],
        "temperature": 0,
        "messages": [{"role": "user", "content": WORKLOADS[wl]["prompt"]}],
        "usage": {"include": True},
    }
    if prov:
        body["provider"] = prov
    row = {"rep": rep, "model": model, "workload": wl, "arm": arm, "provider_param": prov, **note}
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await c.post(f"{API}/chat/completions", json=body)
            row["latency_ms"] = round((time.perf_counter() - t0) * 1000)
            d = r.json()
            if r.status_code != 200 or "error" in d:
                row["error"] = str(d.get("error") or r.status_code)[:300]
                return row
            u = d.get("usage") or {}
            row.update(
                {
                    "served": d.get("provider"),
                    "prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                    "reasoning_tokens": (u.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    ),
                    "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "cost_usd": u.get("cost"),
                    "finish": (d.get("choices") or [{}])[0].get("finish_reason"),
                }
            )
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            row["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return row


async def main(reps: int) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "bench" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"provider_bench_{stamp}.jsonl"
    err_path = out_dir / f"provider_bench_{stamp}.errors.jsonl"

    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(timeout=180, headers=headers) as c:
        eps = {m: await endpoints(c, m) for m in MODELS}
        (out_dir / f"provider_bench_{stamp}.endpoints.json").write_text(json.dumps(eps))
        forecasters = {m: OutputForecaster(prior_tokens=400, min_samples=2) for m in MODELS}
        sem = asyncio.Semaphore(6)
        n_ok = n_err = 0
        for rep in range(reps):
            tasks = []
            for m in MODELS:
                for wl in WORKLOADS:
                    for arm in ARMS:
                        prov, note = provider_param(arm, eps[m], wl, forecasters[m])
                        tasks.append(one_call(c, sem, m, wl, arm, prov, note, rep))
            results = await asyncio.gather(*tasks)
            with rows_path.open("a") as fr, err_path.open("a") as fe:
                for row in results:
                    if "error" in row:
                        n_err += 1
                        fe.write(json.dumps(row) + "\n")
                    else:
                        n_ok += 1
                        fr.write(json.dumps(row) + "\n")
                        # thrift sadece KENDI trafiginden ogrenir
                        if row["arm"] == "thrift" and row.get("completion_tokens") is not None:
                            forecasters[row["model"]].observe(
                                row["workload"], row["completion_tokens"]
                            )
            print(f"  tur {rep + 1}/{reps}: {n_ok} basarili, {n_err} hata", flush=True)
    return rows_path


def summarize(rows_path: Path) -> str:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    eps = json.loads(
        rows_path.with_name(rows_path.name.replace(".jsonl", ".endpoints.json")).read_text()
    )
    price = {}  # (model, provider_name) -> (in, out) per token; en ucuz olani al
    for m, lst in eps.items():
        for e in lst:
            p = e.get("pricing") or {}
            k = (m, e.get("provider_name"))
            v = (float(p.get("prompt") or 0), float(p.get("completion") or 0))
            if k not in price or sum(v) < sum(price[k]):
                price[k] = v

    lines = [
        f"# Provider benchmark — {rows_path.stem}",
        "",
        f"{len(rows)} successful calls. All costs are OpenRouter-billed `usage.cost`.",
        "",
    ]

    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    # 1) Gozlenen fatura
    lines += [
        "## Billed cost per call (observed)",
        "",
        "| model | workload | " + " | ".join(ARMS) + " |",
        "|---|---|" + "---:|" * len(ARMS),
    ]
    tot = dict.fromkeys(ARMS, 0.0)
    for m in MODELS:
        for wl in WORKLOADS:
            cells = []
            for a in ARMS:
                xs = [
                    r["cost_usd"]
                    for r in rows
                    if r["model"] == m
                    and r["workload"] == wl
                    and r["arm"] == a
                    and r.get("cost_usd") is not None
                ]
                tot[a] += sum(xs)
                cells.append(f"${mean(xs) * 1e6:,.1f}µ" if xs else "—")
            lines.append(f"| {m.split('/')[-1]} | {wl} | " + " | ".join(cells) + " |")
    lines += ["", "**Total billed:** " + " · ".join(f"{a} ${tot[a]:.5f}" for a in ARMS), ""]

    # 2) Karisim-normalize: ayni token karisimi, her kolun sectigi saglayicinin fiyati
    lines += [
        "## Price effect at a fixed token mix",
        "",
        "Each (model, workload) is priced at the median prompt/completion tokens observed "
        "across all arms, using the list price of the provider each arm was actually "
        "served by. This removes output-length noise and isolates the provider choice.",
        "",
        "| model | workload | tokens in/out | " + " | ".join(ARMS) + " | thrift vs best rival |",
        "|---|---|---|" + "---:|" * len(ARMS) + "---:|",
    ]
    for m in MODELS:
        for wl in WORKLOADS:
            sub = [r for r in rows if r["model"] == m and r["workload"] == wl]
            if not sub:
                continue
            pin = statistics.median(r["prompt_tokens"] for r in sub)
            pout = statistics.median(r["completion_tokens"] for r in sub)
            vals = {}
            for a in ARMS:
                cs = [
                    pin * price[(m, r["served"])][0] + pout * price[(m, r["served"])][1]
                    for r in sub
                    if r["arm"] == a and (m, r.get("served")) in price
                ]
                vals[a] = mean(cs) if cs else float("nan")
            rival = min(v for k, v in vals.items() if k != "thrift")
            delta = (
                (vals["thrift"] - rival) / rival * 100
                if rival == rival and rival > 0
                else float("nan")
            )
            lines.append(
                f"| {m.split('/')[-1]} | {wl} | {pin:.0f}/{pout:.0f} | "
                + " | ".join(f"${vals[a] * 1e6:,.1f}µ" for a in ARMS)
                + f" | {delta:+.1f}% |"
            )

    err_file = rows_path.with_name(rows_path.name.replace(".jsonl", ".errors.jsonl"))
    errs = [json.loads(x) for x in err_file.read_text().splitlines()] if err_file.exists() else []
    lines += [
        "",
        "## Failed requests",
        "",
        " · ".join(f"{a} {sum(1 for e in errs if e['arm'] == a)}" for a in ARMS),
    ]

    # 3) Hangi saglayicilar
    lines += ["", "## Served providers", ""]
    for a in ARMS:
        cnt = {}
        for r in rows:
            if r["arm"] == a:
                cnt[r["served"]] = cnt.get(r["served"], 0) + 1
        lines.append(
            f"- **{a}**: "
            + ", ".join(f"{k} ×{v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1]))
        )

    # 4) Tahmin
    th = [r for r in rows if r["arm"] == "thrift"]
    if th:
        lines += ["", "## thrift forecaster", ""]
        for wl in WORKLOADS:
            xs = [r for r in th if r["workload"] == wl]
            if xs:
                srcs = sorted({r.get("forecast_src") for r in xs})
                lines.append(
                    f"- {wl}: forecast {mean([r['forecast'] for r in xs]):.0f} tok "
                    f"({'/'.join(srcs)}), actual median "
                    f"{statistics.median(r['completion_tokens'] for r in xs):.0f} tok"
                )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--summarize", type=Path, help="mevcut bir kosuyu ozetle")
    a = ap.parse_args()
    path = a.summarize or asyncio.run(main(a.reps))
    md = summarize(path)
    path.with_suffix(".md").write_text(md)
    print(md)
