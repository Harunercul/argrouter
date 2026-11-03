"""R&D on our own calibration set: which embedding / learner predicts per-query correctness best for a fixed pool.

Protocol = the v5/v7 leave-one-wrapper-out CV (train on two renderings, test on the third, 5 folds), seed 0, under
the v7 objective weights; reports the mean arena score per lam and the best. No RouterArena data.

    python bench/v7/compare_emb.py --emb bge-small qwen__qwen3-embedding-8b --learner lr knn
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fit_router_v7 as f  # noqa: E402

POOL = ["google/gemini-3.7-flash@low", "deepseek/deepseek-v4.1-flash-1017", "z-ai/glm-5.3-flash",
        "openai/gpt-6.1-terra", "anthropic/claude-haiku-5.5"]


def weights(kind, items, S, ref, cands):
    """hard = v7 objective (equal benchmark share x difficulty, noise 0); ramix = RA benchmark shares
    (route_data/v7/ra_mix.json), noise 0; ramix_hard = RA shares x difficulty, noise 0; rasub(_hard) = RA benchmark
    and subset shares (x difficulty), noise 0."""
    bc = collections.Counter(it["benchmark"] for it in items)
    if kind.startswith("rasub"):
        # RA benchmark shares, and inside a benchmark RA's subset shares (route_data/v7/ra_mix.json subset_count);
        # our items of subsets RA does not have get 0
        mix = json.loads((f.ROOT / "route_data/v7/ra_mix.json").read_text())
        w = np.zeros(len(items))
        for b in bc:
            idx = [i for i, it in enumerate(items) if it["benchmark"] == b]
            groups = mix["subset_count"].get(b)
            if not groups:
                w[idx] = mix["share"][b] / len(idx)
                continue
            for g, cnt in groups.items():
                gi = [i for i in idx if items[i].get("subset") in g.split("|")]
                if gi:
                    w[gi] = cnt / len(gi)
            w[idx] *= mix["share"][b] / w[idx].sum()
    elif kind.startswith("ramix"):
        share = json.loads((f.ROOT / "route_data/v7/ra_mix.json").read_text())["share"]
        w = np.array([share[it["benchmark"]] / bc[it["benchmark"]] for it in items])
    else:
        w = np.array([1.0 / bc[it["benchmark"]] for it in items])
    if kind in ("hard", "ramix_hard", "rasub_hard"):
        w = w * (f.HARD_FLOOR + 1 - S[ref].mean(0))
    elif kind.startswith("hard@"):  # hard@<floor>: equal benchmark shares x (floor + reference error)
        w = w * (float(kind[5:]) + 1 - S[ref].mean(0))
    w[np.nan_to_num(S[cands]).max(0) == 0] = 0
    return w / w.sum()


def load_emb(name):
    p = f.fit_router_v4.CACHE if name == "bge-small" else f.ROOT / "route_data/v7/emb" / f"{name}.npz"
    z = np.load(p)
    E = {k: z[k].astype(np.float64) for k in f.RENDER}
    for k in E:
        E[k] /= np.linalg.norm(E[k], axis=1, keepdims=True) + 1e-9
    return E


def fit_predict(learner, Xtr, Ltr, S, C, w, Xte, Lte, C_lr=1.0, k_nn=50):
    """-> P, K arrays (pool x test) for one fold."""
    from sklearn.linear_model import LogisticRegression, Ridge
    rep = len(Xtr)
    X, XL = np.vstack(Xtr), np.hstack([np.vstack(Xtr), np.vstack(Ltr)])
    ww = np.tile(w, rep)
    ww = ww / ww.mean()
    XLte = np.hstack([Xte, Lte])
    if learner.startswith("cluster"):
        # cluster router: k-means on training texts; per cluster, smoothed weighted mean correctness and cost
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k_nn, n_init=3, random_state=0).fit(X)
        lab, labte = km.labels_, km.predict(Xte)
        P, K = [], []
        for j in range(S.shape[0]):
            y, c = np.tile(S[j], rep), np.tile(C[j] * 1000, rep)
            g_y, g_c = np.average(y, weights=ww), np.average(c, weights=ww)
            sw = np.bincount(lab, ww, k_nn)
            py = (np.bincount(lab, ww * y, k_nn) + C_lr * g_y) / (sw + C_lr)
            pc = (np.bincount(lab, ww * c, k_nn) + C_lr * g_c) / (sw + C_lr)
            P.append(py[labte])
            K.append(pc[labte])
        return np.array(P), np.array(K)
    P, K = [], []
    for j in range(S.shape[0]):
        y = np.tile(S[j], rep)
        if learner == "lr":
            m = LogisticRegression(C=C_lr, max_iter=3000).fit(X, y, sample_weight=ww)
            P.append(m.predict_proba(Xte)[:, 1])
        elif learner == "knn":
            sim = Xte @ X.T
            idx = np.argpartition(-sim, k_nn, axis=1)[:, :k_nn]
            s = np.take_along_axis(sim, idx, 1)
            wt = np.exp(s * 20) * ww[idx]
            P.append((wt * y[idx]).sum(1) / wt.sum(1))
        yc = np.tile(np.log(C[j] * 1000 + 1e-4), rep)
        rg = Ridge(alpha=1.0).fit(XL, yc, sample_weight=ww)
        s2 = float(np.average((yc - rg.predict(XL)) ** 2, weights=ww))
        K.append(np.exp(rg.predict(XLte) + s2 / 2))
    return np.array(P), np.array(K)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", default=["bge-small"])
    ap.add_argument("--learner", nargs="+", default=["lr"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--weights", default="hard", choices=["hard", "ramix", "ramix_hard", "rasub", "rasub_hard"])
    ap.add_argument("--pool", nargs="+", default=POOL)
    a = ap.parse_args()
    items, models, S, C, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
    cands = [j for j, m in enumerate(models) if m not in skip]
    w = weights(a.weights, items, S, ref, cands)
    S, C = np.nan_to_num(S), np.nan_to_num(C)
    pool = [models.index(m) for m in a.pool]
    Sp, Cp = S[pool], C[pool]
    texts = f.embed_all(items)[0]
    L = {k: f.length_feature(v) for k, v in texts.items()}
    n, ar = S.shape[1], np.arange(S.shape[1])
    order = np.random.default_rng(0).permutation(n)
    for emb in a.emb:
        E = load_emb(emb)
        for learner in a.learner:
            res = collections.defaultdict(list)
            for held in f.RENDER:
                tr_r = [r for r in f.RENDER if r != held]
                P, K = np.zeros((len(pool), n)), np.zeros((len(pool), n))
                for k in range(5):
                    te = order[k::5]
                    tr = np.setdiff1d(order, te)
                    P[:, te], K[:, te] = fit_predict(learner, [E[r][tr] for r in tr_r], [L[r][tr] for r in tr_r],
                                                     Sp[:, tr], Cp[:, tr], w[tr], E[held][te], L[held][te], a.C, a.k)
                for lam in f.LAMS:
                    pick = (P - lam * K).argmax(0)
                    acc, cost = w @ Sp[pick, ar], w @ Cp[pick, ar] * 1000
                    res[lam].append((f.arena(acc, cost), acc, cost))
            means = {lam: np.mean(v, 0) for lam, v in res.items()}
            best = max(means, key=lambda x: means[x][0])
            print(f"{emb:34} {learner:4} C={a.C} k={a.k} w={a.weights} pool={len(pool)}: best lam {best}: {means[best][0]:.2f} "
                  f"acc {means[best][1] * 100:.1f}% ${means[best][2]:.3f}/1K   all: " +
                  " ".join(f"{lam}:{m[0]:.2f}" for lam, m in means.items()), flush=True)
