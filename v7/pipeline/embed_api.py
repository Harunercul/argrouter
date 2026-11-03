"""Embed the v7 calibration texts (bare, A, B renderings of all.jsonl + qanta_s1.jsonl) with an OpenRouter
embedding model; same text as fit_router_v4.embed_all (last MAX_CHARS characters). Cached per model in
route_data/v7/emb/<model>.npz.

    python bench/v7/embed_api.py qwen/qwen3-embedding-8b openai/text-embedding-3-large
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fit_router_v7 as f  # noqa: E402
from fit_router_v4 import MAX_CHARS, SYSTEM, bare, render, render_b  # noqa: E402

API = "https://openrouter.ai/api/v1/embeddings"
KEY = dict(x.split("=", 1) for x in (f.ROOT / ".env").read_text().split() if "=" in x)["OPENROUTER_API_KEY"].strip('"')
OUT = f.ROOT / "route_data" / "v7" / "emb"
BATCH, CONC = 32, 8


def texts_v7():
    items = f.jsonl(f.ITEMS) + f.jsonl(f.S1)
    return {"bare": [bare(it) for it in items], "A": [render(it) for it in items],
            "B": [SYSTEM + "\n" + render_b(it) for it in items]}


async def embed(model: str, texts: list[str]) -> tuple[np.ndarray, float]:
    out: list = [None] * len(texts)
    cost = 0.0
    sem = asyncio.Semaphore(CONC)

    async def one(client, i):
        nonlocal cost
        batch = [t[-MAX_CHARS:] or " " for t in texts[i:i + BATCH]]
        async with sem:
            for attempt in range(6):
                try:
                    r = await client.post(API, json={"model": model, "input": batch},
                                          headers={"Authorization": f"Bearer {KEY}"}, timeout=120)
                    d = r.json()
                    if "data" not in d:
                        raise RuntimeError(str(d)[:200])
                    for row in d["data"]:
                        out[i + row["index"]] = row["embedding"]
                    cost += (d.get("usage") or {}).get("cost") or 0
                    return
                except Exception as e:  # retry with backoff
                    if attempt == 5:
                        raise
                    print(f"  retry {model} batch {i}: {e}", flush=True)
                    await asyncio.sleep(2 ** attempt)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, i) for i in range(0, len(texts), BATCH)))
    return np.array(out, dtype=np.float32), cost


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    T = texts_v7()
    for model in sys.argv[1:]:
        p = OUT / (model.replace("/", "__") + ".npz")
        if p.exists():
            print(f"{model}: cached")
            continue
        E, total = {}, 0.0
        for k, v in T.items():
            E[k], c = asyncio.run(embed(model, v))
            total += c
        np.savez(p, n=len(T["A"]), **E)
        print(f"{model}: dim {E['A'].shape[1]}, ${total:.3f}", flush=True)
