"""R&D: nested_select.py with one selection objective for all three weightings: a pool and one lam are chosen to
maximise the mean of the hard, ramix and ramix_hard scores (each with its own cluster-router predictions) on one
half; the result is scored on the other half under each weighting.

    python bench/v7/nested_combo.py --emb qwen__qwen3-embedding-8b openai__text-embedding-3-large
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
from nested_select import V6  # noqa: E402
from select_cluster import LAMS, oof_cluster, score  # noqa: E402

WK = ("hard", "ramix", "ramix_hard")


def combo(pool, R, S, C, W):
    sc = [score(pool, R[k], S, C, W[k]) for k in WK]
    return {lam: np.mean([s[lam][0] for s in sc]) for lam in LAMS}


def greedy(R, S, C, W, cands, max_pool=6):
    pool, best = [], -np.inf
    while len(pool) < max_pool:
        trial = {j: max(combo(pool + [j], R, S, C, W).values()) for j in cands if j not in pool}
        j = max(trial, key=trial.get)
        if trial[j] - best < f.MIN_GAIN:
            break
        pool.append(j)
        best = trial[j]
    c = combo(pool, R, S, C, W)
    return pool, max(c, key=c.get)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", default=["qwen__qwen3-embedding-8b"])
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[123])
    a = ap.parse_args()
    items, models, S, C, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
    cands = [j for j, m in enumerate(models) if m not in skip]
    S0 = S
    S, C = np.nan_to_num(S), np.nan_to_num(C)
    v6 = [models.index(m) for m in V6]
    Wfull, R = {}, {}
    for k in WK:
        Wfull[k] = ce.weights(k, items, S0, ref, cands)
        per = [oof_cluster(ce.load_emb(e), S, C, Wfull[k], 30, 5.0) for e in a.emb]
        R[k] = [(np.mean([p[i][0] for p in per], 0), np.mean([p[i][1] for p in per], 0)) for i in range(len(per[0]))]
    for split_seed in a.split_seeds:
      rng = np.random.default_rng(split_seed)
      half = np.zeros(len(items), int)
      for b in sorted({it["benchmark"] for it in items}):
        idx = rng.permutation([i for i, it in enumerate(items) if it["benchmark"] == b])
        half[idx[: len(idx) // 2]] = 1
      print(f"-- split seed {split_seed}", flush=True)
      for sel_half in (0, 1) if len(a.split_seeds) > 1 else (0, 1, None):
          Ws = {k: Wfull[k] * (half == sel_half if sel_half is not None else 1) for k in WK}
          We = {k: Wfull[k] * (half != sel_half if sel_half is not None else 1) for k in WK}
          Ws = {k: v / v.sum() for k, v in Ws.items()}
          We = {k: v / v.sum() for k, v in We.items()}
          pool, lam = greedy(R, S, C, Ws, cands)
          c6 = combo(v6, R, S, C, Ws)
          lam6 = max(c6, key=c6.get)
          tag = f"select on half {sel_half}" if sel_half is not None else "ALL items (final pick)"
          print(f"{tag}: pool {[models[j] for j in pool]} lam {lam}", flush=True)
          for k in WK:
              s1, s6 = score(pool, R[k], S, C, We[k])[lam], score(v6, R[k], S, C, We[k])[lam6]
              print(f"   {k:10} {'other half' if sel_half is not None else 'in-sample '}: {s1[0]:.2f} "
                    f"(acc {s1[1] * 100:.1f}% ${s1[2]:.3f}/1K) vs v6 pair {s6[0]:.2f}  gain {s1[0] - s6[0]:+.2f}",
                    flush=True)
