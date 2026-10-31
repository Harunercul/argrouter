"""v4 response matrix: question x model, prompts from prompts_v4 (clean room, no RouterArena templates).

Plain call: model + our messages, provider defaults (no temperature/max_tokens); `usage.include` only adds
the bill to the response. Resumes from route_data/v4/matrix/<model>.jsonl.

    python bench/v4/run_v4.py --items route_data/v4/pilot.jsonl --models google/gemma-4-31b-it --concurrency 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from prompts_v4 import build_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ENV = dict(x.split("=", 1) for x in (ROOT / ".env").read_text().split() if "=" in x)
API = "https://openrouter.ai/api/v1/chat/completions"
OUT = ROOT / "route_data" / "v4" / "matrix"


def out_path(model: str) -> Path:
    return OUT / (model.replace("/", "__").replace(":", "_") + ".jsonl")


def done_keys(model: str) -> set[str]:
    p = out_path(model)
    if not p.exists():
        return set()
    keys = set()
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if not r.get("error"):
            keys.add(r["gidx"])
    return keys


def split_variant(model: str) -> tuple[str, dict]:
    """ "model@low" -> (model id, reasoning body); without @ the provider default is used."""
    if "@" not in model:
        return model, {}
    mid, effort = model.split("@", 1)
    return mid, {"reasoning": {"effort": effort}}  # "none" = dusunme kapali


async def call(client: httpx.AsyncClient, model: str, messages: list[dict]) -> dict:
    mid, extra = split_variant(model)
    body = {
        "model": mid,
        "messages": messages,
        "usage": {"include": True},
        **extra,
    }
    last = ""
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            r = await client.post(API, json=body)
            dt = round((time.perf_counter() - t0) * 1000)
            d = r.json()
            if r.status_code == 200 and "error" not in d and d.get("choices"):
                u = d.get("usage") or {}
                msg = d["choices"][0].get("message") or {}
                return {
                    "response": msg.get("content") or "",
                    "finish": d["choices"][0].get("finish_reason"),
                    "provider": d.get("provider"),
                    "input_tokens": u.get("prompt_tokens"),
                    "output_tokens": u.get("completion_tokens"),
                    "reasoning_tokens": (u.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    ),
                    "cost_usd": u.get("cost"),
                    "latency_ms": dt,
                }
            last = str(d.get("error") or r.status_code)[:300]
        except Exception as e:  # ag hatasi, zaman asimi
            last = f"{type(e).__name__}: {e}"[:300]
        await asyncio.sleep(2**attempt + random.random())
    return {"error": last}


async def main(
    items_path: Path, models: list[str], concurrency: int, limit: int | None
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [json.loads(x) for x in items_path.read_text().splitlines() if x.strip()]
    if limit:
        items = items[:limit]
    headers = {"Authorization": f"Bearer {ENV['OPENROUTER_API_KEY']}"}
    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(420.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        jobs = []
        for m in models:
            done = done_keys(m)
            jobs += [(m, it) for it in items if it["id"] not in done]
        random.Random(0).shuffle(jobs)  # modelleri karistir: tek saglayiciya yuklenme
        print(
            f"{len(jobs)} cagri ({len(models)} model x {len(items)} soru, yapilanlar atlandi)",
            flush=True,
        )
        n = {"ok": 0, "err": 0, "usd": 0.0}
        files = {m: out_path(m).open("a") for m in models}

        async def one(m: str, it: dict) -> None:
            async with sem:
                res = await call(client, m, build_messages(it))
            row = {
                "model": m,
                "gidx": it["id"],
                "dataset": it["benchmark"],
                **res,
            }
            files[m].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[m].flush()
            n["err" if "error" in res else "ok"] += 1
            n["usd"] += res.get("cost_usd") or 0
            if (n["ok"] + n["err"]) % 100 == 0:
                print(f"  {n['ok']} ok, {n['err']} hata, ${n['usd']:.3f}", flush=True)

        await asyncio.gather(*(one(m, it) for m, it in jobs))
        for f in files.values():
            f.close()
        print(
            f"bitti: {n['ok']} ok, {n['err']} hata, toplam ${n['usd']:.4f}", flush=True
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out-dir", type=Path, help="matrix folder (default route_data/v4/matrix)")
    a = ap.parse_args()
    if a.out_dir:
        OUT = a.out_dir
    asyncio.run(main(a.items, a.models, a.concurrency, a.limit))
