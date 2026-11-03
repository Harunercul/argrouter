"""R&D: benchmark-aware predictors for the fixed v7 pool, same LOWO protocol as compare_emb.py.

The per-benchmark-best rule (route every item of a benchmark to its best model) scores ~82.9 in-sample, above every
learned router so far, so these learners first recognise which benchmark a query looks like (kNN on the embedding,
trained on our own items' benchmark tags) and then route from smoothed per-benchmark / per-sub-cluster statistics:

  bench     P = sum_b p(b|x) * smoothed mean correctness of benchmark b (cost likewise)
  bsub      hard benchmark guess, then k-means sub-clusters inside it (one per ~--m training rows), shrunk to the
            benchmark mean with pseudo-count --C
  clens     cluster router averaged over k in (20, 30, 50) x 3 seeds
  blend     0.5 * bsub + 0.5 * lr (cost from bsub)

    python bench/v7/compare_pred.py --emb qwen__qwen3-embedding-8b --learner bench bsub --C 5 --m 60
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare_emb as ce  # noqa: E402
import fit_router_v7 as f  # noqa: E402


def bench_proba(Xtr, btr, Xte, nb, k=10):
    sim = Xte @ Xtr.T
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    p = np.zeros((len(Xte), nb))
    for c in range(k):
        np.add.at(p, (np.arange(len(Xte)), btr[idx[:, c]]), 1.0 / k)
    return p


def smoothed(lab, ww, v, nl, prior, C):
    return (np.bincount(lab, ww * v, nl) + C * prior) / (np.bincount(lab, ww, nl) + C)


def predict(learner, Xtr, Ltr, btr, S, Cs, w, Xte, Lte, nb, C, m):
    from sklearn.cluster import KMeans
    rep = len(Xtr)
    X = np.vstack(Xtr)
    b = np.tile(btr, rep)
    ww = np.tile(w, rep)
    ww = ww / ww.mean()
    if learner == "clens":
        out = [ce.fit_predict("cluster", Xtr, Ltr, S, Cs, w, Xte, Lte, C, k) for k in (20, 30, 50)]
        return np.mean([o[0] for o in out], 0), np.mean([o[1] for o in out], 0)
    if learner == "blend":
        P1, K1 = predict("bsub", Xtr, Ltr, btr, S, Cs, w, Xte, Lte, nb, C, m)
        P2, _ = ce.fit_predict("lr", Xtr, Ltr, S, Cs, w, Xte, Lte)
        return 0.5 * P1 + 0.5 * P2, K1
    pb = bench_proba(X, b, Xte, nb)
    P, K = np.zeros((S.shape[0], len(Xte))), np.zeros((S.shape[0], len(Xte)))
    if learner == "bench":
        for j in range(S.shape[0]):
            y, c = np.tile(S[j], rep), np.tile(Cs[j] * 1000, rep)
            gy, gc = np.average(y, weights=ww), np.average(c, weights=ww)
            P[j] = pb @ smoothed(b, ww, y, nb, gy, C)
            K[j] = pb @ smoothed(b, ww, c, nb, gc, C)
        return P, K
    # bsub: sub-clusters inside the guessed benchmark
    bte = pb.argmax(1)
    for bb in np.unique(bte):
        tr, te = np.where(b == bb)[0], np.where(bte == bb)[0]
        k = max(1, len(tr) // m)
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(X[tr]) if k > 1 else None
        lab = km.labels_ if km else np.zeros(len(tr), int)
        labte = km.predict(Xte[te]) if km else np.zeros(len(te), int)
        for j in range(S.shape[0]):
            y, c = np.tile(S[j], rep)[tr], np.tile(Cs[j] * 1000, rep)[tr]
            wb = ww[tr]
            gy = np.average(y, weights=wb) if wb.sum() else y.mean()
            gc = np.average(c, weights=wb) if wb.sum() else c.mean()
            P[j, te] = smoothed(lab, wb, y, k, gy, C)[labte]
            K[j, te] = smoothed(lab, wb, c, k, gc, C)[labte]
    return P, K


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", default=["qwen__qwen3-embedding-8b"])
    ap.add_argument("--learner", nargs="+", default=["bench", "bsub"])
    ap.add_argument("--C", type=float, default=5.0)
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--weights", default="hard", choices=["hard", "ramix", "ramix_hard", "rasub", "rasub_hard"])
    ap.add_argument("--pool", nargs="+", default=ce.POOL)
    a = ap.parse_args()
    items, models, S, Cs, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
    cands = [j for j, mm in enumerate(models) if mm not in skip]
    bc = collections.Counter(it["benchmark"] for it in items)
    w = ce.weights(a.weights, items, S, ref, cands)
    S, Cs = np.nan_to_num(S), np.nan_to_num(Cs)
    pool = [models.index(mm) for mm in a.pool]
    Sp, Cp = S[pool], Cs[pool]
    names = sorted(bc)
    B = np.array([names.index(it["benchmark"]) for it in items])
    texts = f.embed_all(items)[0]
    L = {k: f.length_feature(v) for k, v in texts.items()}
    n, ar = S.shape[1], np.arange(S.shape[1])
    order = np.random.default_rng(0).permutation(n)
    # reference: per-benchmark best model, chosen on the training folds only
    for emb in a.emb:
        E = ce.load_emb(emb)
        for learner in a.learner:
            res, bacc = collections.defaultdict(list), []
            for held in f.RENDER:
                tr_r = [r for r in f.RENDER if r != held]
                P, K = np.zeros((len(pool), n)), np.zeros((len(pool), n))
                for k in range(5):
                    te = order[k::5]
                    tr = np.setdiff1d(order, te)
                    P[:, te], K[:, te] = predict(learner, [E[r][tr] for r in tr_r], [L[r][tr] for r in tr_r], B[tr],
                                                 Sp[:, tr], Cp[:, tr], w[tr], E[held][te], L[held][te], len(names),
                                                 a.C, a.m)
                    if learner in ("bench", "bsub"):
                        bacc.append((bench_proba(np.vstack([E[r][tr] for r in tr_r]), np.tile(B[tr], 2), E[held][te],
                                                 len(names)).argmax(1) == B[te]).mean())
                for lam in f.LAMS:
                    pick = (P - lam * K).argmax(0)
                    acc, cost = w @ Sp[pick, ar], w @ Cp[pick, ar] * 1000
                    res[lam].append((f.arena(acc, cost), acc, cost))
            means = {lam: np.mean(v, 0) for lam, v in res.items()}
            best = max(means, key=lambda x: means[x][0])
            print(f"{emb:34} {learner:5} C={a.C} m={a.m} w={a.weights} pool={len(pool)}: best lam {best}: {means[best][0]:.2f} "
                  f"acc {means[best][1] * 100:.1f}% ${means[best][2]:.3f}/1K   all: " +
                  " ".join(f"{lam}:{mm[0]:.2f}" for lam, mm in means.items()) +
                  (f"   bench-id acc {np.mean(bacc) * 100:.1f}%" if bacc else ""), flush=True)
