"""R&D: predictor-aware pool selection. The v7 pool was chosen with the LR predictor; the cluster router predicts
better, so redo the greedy forward selection with cluster-router out-of-fold predictions for every candidate.

Out-of-fold predictions: leave-one-wrapper-out x 5 folds x seeds (0, 1) (the k-means clustering does not depend on
the model, so one clustering per fold serves all candidates). Selection objective = mean over the 6 runs of the
score at the best lam; reported under each weighting so a pool that only wins under one of them is visible.

    python bench/v7/select_cluster.py --emb qwen__qwen3-embedding-8b --weights hard ramix ramix_hard
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

LAMS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
COST_SCALE = 1.0  # RouterArena cost / our cost; calib_ra.py measured about 0.6


def oof_cluster(E, S, C, w, k, pc, seeds=(0, 1)):
    from sklearn.cluster import KMeans
    n = S.shape[1]
    runs = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(n)
        for held in f.RENDER:
            tr_r = [r for r in f.RENDER if r != held]
            P, K = np.zeros(S.shape), np.zeros(S.shape)
            for fo in range(5):
                te = order[fo::5]
                tr = np.setdiff1d(order, te)
                X = np.vstack([E[r][tr] for r in tr_r])
                ww = np.tile(w[tr], 2)
                ww = ww / ww.mean()
                km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)
                lab, labte = km.labels_, km.predict(E[held][te])
                sw = np.bincount(lab, ww, k) + pc
                for j in range(S.shape[0]):
                    y, c = np.tile(S[j, tr], 2), np.tile(C[j, tr] * 1000, 2)
                    gy, gc = np.average(y, weights=ww), np.average(c, weights=ww)
                    P[j, te] = ((np.bincount(lab, ww * y, k) + pc * gy) / sw)[labte]
                    K[j, te] = ((np.bincount(lab, ww * c, k) + pc * gc) / sw)[labte]
            runs.append((P, K))
    return runs


def score(pool, runs, S, C, w):
    ar = np.arange(S.shape[1])
    out = {}
    for lam in LAMS:
        v = []
        for P, K in runs:
            pick = np.array(pool)[(P[pool] - lam * K[pool]).argmax(0)]
            cost = w @ C[pick, ar] * 1000 * COST_SCALE
            v.append((f.arena(w @ S[pick, ar], cost), w @ S[pick, ar], cost))
        out[lam] = np.mean(v, 0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="qwen__qwen3-embedding-8b")
    ap.add_argument("--weights", nargs="+", default=["hard", "ramix", "ramix_hard"])
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--C", type=float, default=5.0)
    ap.add_argument("--max-pool", type=int, default=6)
    a = ap.parse_args()
    items, models, S, C, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
    cands = [j for j, m in enumerate(models) if m not in skip]
    print("candidates:", [models[j] for j in cands])
    E = ce.load_emb(a.emb)
    S0, C0 = S, C
    S, C = np.nan_to_num(S), np.nan_to_num(C)
    for wk in a.weights:
        w = ce.weights(wk, items, S0, ref, cands)
        runs = oof_cluster(E, S, C, w, a.k, a.C)
        pool, best = [], -np.inf
        print(f"\n== weights {wk}  ({a.emb}, cluster k={a.k} pseudo {a.C})", flush=True)
        while len(pool) < a.max_pool:
            trial = {j: max(v[0] for v in score(pool + [j], runs, S, C, w).values()) for j in cands if j not in pool}
            j = max(trial, key=trial.get)
            if trial[j] - best < f.MIN_GAIN:
                print(f"  stop: best addition {models[j]} {trial[j]:.2f} (+{trial[j] - best:.2f})")
                break
            pool.append(j)
            best = trial[j]
            print(f"  + {models[j]:40} {best:.2f}   runners-up: " + ", ".join(
                f"{models[q].split('/')[-1]} {trial[q]:.2f}" for q in sorted(trial, key=trial.get)[-4:-1][::-1]),
                flush=True)
        for lam, (sc, acc, cost) in score(pool, runs, S, C, w).items():
            print(f"  lam {lam:<5} {sc:6.2f}  acc {acc * 100:.1f}%  ${cost:.3f}/1K")
        for name, pl in (("v6 pair", ["google/gemini-3.7-flash@low", "deepseek/deepseek-v4.1-flash"]),
                         ("v7 pool", ce.POOL)):
            sc = score([models.index(m) for m in pl], runs, S, C, w)
            print(f"  {name} under this predictor: " + " ".join(f"{lam}:{v[0]:.2f}" for lam, v in sc.items()))
