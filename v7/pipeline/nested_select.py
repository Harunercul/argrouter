"""R&D: honest estimate of what predictor-aware pool selection gains over the v6 pair. Items are split into two
halves (stratified by benchmark, seed 123); greedy selection + lam are chosen on one half's out-of-fold scores and
the chosen router is scored on the other half, both ways round. The out-of-fold predictions themselves are the
cluster-router LOWO ones (mean over the given embeddings).

    python bench/v7/nested_select.py --emb qwen__qwen3-embedding-8b openai__text-embedding-3-large
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare_emb as ce  # noqa: E402
import fit_router_v7 as f  # noqa: E402
from select_cluster import oof_cluster, score  # noqa: E402

V6 = ["google/gemini-3.7-flash@low", "deepseek/deepseek-v4.1-flash"]


def greedy(runs, S, C, w, cands, max_pool=6):
    pool, best = [], -np.inf
    while len(pool) < max_pool:
        trial = {j: max(v[0] for v in score(pool + [j], runs, S, C, w).values()) for j in cands if j not in pool}
        j = max(trial, key=trial.get)
        if trial[j] - best < f.MIN_GAIN:
            break
        pool.append(j)
        best = trial[j]
    sc = score(pool, runs, S, C, w)
    return pool, max(sc, key=lambda x: sc[x][0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", default=["qwen__qwen3-embedding-8b"])
    a = ap.parse_args()
    items, models, S, C, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
    cands = [j for j, m in enumerate(models) if m not in skip]
    S0 = S
    S, C = np.nan_to_num(S), np.nan_to_num(C)
    rng = np.random.default_rng(123)
    half = np.zeros(len(items), int)
    for b in {it["benchmark"] for it in items}:
        idx = rng.permutation([i for i, it in enumerate(items) if it["benchmark"] == b])
        half[idx[: len(idx) // 2]] = 1
    v6 = [models.index(m) for m in V6]
    for wk in ("hard", "ramix", "ramix_hard"):
        w = ce.weights(wk, items, S0, ref, cands)
        per = [oof_cluster(ce.load_emb(e), S, C, w, 30, 5.0) for e in a.emb]
        runs = [(np.mean([p[i][0] for p in per], 0), np.mean([p[i][1] for p in per], 0)) for i in range(len(per[0]))]
        for sel_half in (0, 1):
            ws = w * (half == sel_half); ws /= ws.sum()
            we = w * (half != sel_half); we /= we.sum()
            pool, lam = greedy(runs, S, C, ws, cands)
            sc_sel, sc_v6 = score(pool, runs, S, C, we), score(v6, runs, S, C, we)
            lam_v6 = max(score(v6, runs, S, C, ws).items(), key=lambda x: x[1][0])[0]
            print(f"{wk:10} select on half {sel_half}: pool {[models[j].split('/')[-1] for j in pool]} lam {lam} | "
                  f"other half: selected {sc_sel[lam][0]:.2f} vs v6 pair {sc_v6[lam_v6][0]:.2f} "
                  f"(gain {sc_sel[lam][0] - sc_v6[lam_v6][0]:+.2f})", flush=True)
