"""Which objective weighting tracks RouterArena? Re-score the two routers that have a disclosed RouterArena result
(v4 73.56, v6 76.18; both the RouterV4 class: LR on bge-small, their own pool and lam, their own training weights)
on our set with out-of-fold predictions (LOWO, 5 folds, seeds 0 and 1; trained on the v4 items only, like the
originals), under every weighting. A weighting whose v6 - v4 gap is near +2.62 and whose levels are near the RA
scores is the better stand-in for RouterArena. Uses only those two published totals, no per-query RA data.

    python bench/v7/calib_ra.py [weighting ...]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare_emb as ce  # noqa: E402
import fit_router_v7 as f  # noqa: E402

ROUTERS = {  # name: (pool, lam, training weights, RA score)
    "v4": (["google/gemma-4-31b-it", "openai/gpt-6-luna", "google/gemini-3.1-flash-lite@none",
            "anthropic/claude-haiku-5.5", "deepseek/deepseek-v4-flash-0731", "google/gemini-3-flash-preview@none"],
           0.08, "equal", 73.56),
    "v6": (["google/gemini-3.7-flash@low", "deepseek/deepseek-v4.1-flash"], 0.04, "v5hard", 76.18),
}
KINDS = tuple(sys.argv[1:]) or ("hard", "ramix", "ramix_hard", "rasub", "rasub_hard")

if __name__ == "__main__":
    items, models, S, C, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
    cands = [j for j, m in enumerate(models) if m not in skip]
    S0 = S
    S, C = np.nan_to_num(S), np.nan_to_num(C)
    n, ar = len(items), np.arange(len(items))
    v4items = np.array([not it["id"].endswith("/s1") for it in items])
    bc = collections.Counter(it["benchmark"] for it in items)
    w_eq = np.array([1.0 / bc[it["benchmark"]] for it in items]); w_eq /= w_eq.sum()
    TRAIN = {"equal": w_eq, "v5hard": w_eq * (f.HARD_FLOOR + 1 - S[ref].mean(0))}
    W = {k: ce.weights(k, items, S0, ref, cands) for k in KINDS}
    E = ce.load_emb("bge-small")
    texts = f.embed_all(items)[0]
    L = {k: f.length_feature(v) for k, v in texts.items()}
    res = collections.defaultdict(list)
    for name, (pl, lam, tw, ra) in ROUTERS.items():
        pool = [models.index(m) for m in pl]
        for seed in (0, 1):
            order = np.random.default_rng(seed).permutation(n)
            for held in f.RENDER:
                tr_r = [r for r in f.RENDER if r != held]
                P, K = np.zeros((len(pool), n)), np.zeros((len(pool), n))
                for k in range(5):
                    te = order[k::5]
                    tr = np.setdiff1d(order, te)
                    tr = tr[v4items[tr]]
                    P[:, te], K[:, te] = ce.fit_predict("lr", [E[r][tr] for r in tr_r], [L[r][tr] for r in tr_r],
                                                        S[pool][:, tr], C[pool][:, tr], TRAIN[tw][tr], E[held][te],
                                                        L[held][te])
                pick = np.array(pool)[(P - lam * K).argmax(0)]
                for kind in KINDS:
                    w = W[kind]
                    res[(name, kind)].append((f.arena(w @ S[pick, ar], w @ C[pick, ar] * 1000), w @ S[pick, ar],
                                              w @ C[pick, ar] * 1000))
        print(f"done {name}", flush=True)
    print(f"\n{'weighting':12} {'v4':>7} {'v6':>7} {'v6-v4':>7}   (RouterArena: v4 73.56, v6 76.18, gap +2.62)")
    for kind in KINDS:
        a4, a6 = np.mean(res[("v4", kind)], 0), np.mean(res[("v6", kind)], 0)
        print(f"{kind:12} {a4[0]:7.2f} {a6[0]:7.2f} {a6[0] - a4[0]:+7.2f}   v4 acc {a4[1] * 100:.1f}% ${a4[2]:.3f}"
              f"  v6 acc {a6[1] * 100:.1f}% ${a6[2]:.3f}")
