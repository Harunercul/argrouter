"""Last choices for the frozen router under the RouterArena-calibrated objective (nested_cal.py): which single
embedding (one embedding call per RouterArena query) and lam, for the consensus pool of nested_cal.log.

    python bench/v7/final_cal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare_emb as ce  # noqa: E402
import fit_router_v7 as f  # noqa: E402
import select_cluster as sc  # noqa: E402
from nested_cal import KIND, SCALE  # noqa: E402

POOL = ["google/gemini-4-argon", "openai/gpt-6.1-terra@low", "z-ai/glm-5.3-flash", "google/gemini-3.7-flash@low",
        "openai/gpt-6.1-terra", "anthropic/claude-haiku-5.5"]
PRICE = {"qwen__qwen3-embedding-8b": 0.01, "openai__text-embedding-3-large": 0.13, "voyageai__voyage-4-large": 0.12,
         "google__gemini-embedding-2": 0.2, "baai__bge-m3": 0.01}  # $ / M input tokens

if __name__ == "__main__":
    sc.COST_SCALE = SCALE
    items, models, S, C, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
    cands = [j for j, m in enumerate(models) if m not in skip]
    w = ce.weights(KIND, items, S, ref, cands)
    S, C = np.nan_to_num(S), np.nan_to_num(C)
    pool = [models.index(m) for m in POOL]
    for e, price in PRICE.items():
        res = sc.score(pool, sc.oof_cluster(ce.load_emb(e), S, C, w, 30, 5.0), S, C, w)
        lam = max(res, key=lambda x: res[x][0])
        # router cost: ~1 token per 4 characters of the embedded text (last 4000 characters)
        tok = np.mean([min(len(t), 4000) / 4 for t in f.embed_all(items)[0]["A"]])
        rc = tok * price / 1e6 * 1000
        print(f"{e:32} " + " ".join(f"{x}:{res[x][0]:.2f}" for x in sc.LAMS) + f"  best {lam}: {res[lam][0]:.2f} "
              f"(${res[lam][2]:.3f}/1K)  embedding ${rc:.4f}/1K -> {f.arena(res[lam][1], res[lam][2] + rc):.2f}",
              flush=True)
