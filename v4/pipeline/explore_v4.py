"""Policy exploration on OUR v4 set only (5-fold CV, equal benchmark weights). Compares:
single models, the best fixed cascade, a learned router (embedding -> P(correct) per model, argmax P - lam*cost),
and a learned cascade (two cheap drafts; on disagreement the learned router picks the escalation target).
Nothing here reads RouterArena data.

    python bench/v4/explore_v4.py --prices route_data/v4/prices_standard.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_full_v4 import ROOT, arena, load  # noqa: E402

EMB = ROOT / "route_data" / "v4" / "emb_bge_small.npy"


def embed(items):
    if EMB.exists():
        return np.load(EMB)
    from fastembed import TextEmbedding

    texts = [((it.get("context") or "")[-1500:] + "\n" + it["question"] + "\n" + "\n".join(it.get("options") or []))
             .strip()[:4000] for it in items]
    E = np.array(list(TextEmbedding("BAAI/bge-small-en-v1.5").embed(texts)))
    np.save(EMB, E)
    return E


def fold_probs(E, S, w, C=None, folds=5, seed=0):
    """Out-of-fold P(correct) for every model and row; with C also the out-of-fold expected cost ($/1K: the
    model's mean cost on the training folds), since a row's own cost is not known when routing it."""
    from sklearn.linear_model import LogisticRegression

    n = E.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    P = np.zeros_like(S)
    K = np.zeros_like(S)
    for k in range(folds):
        te = order[k::folds]
        tr = np.setdiff1d(order, te)
        if C is not None:
            K[:, te] = (C[:, tr] @ (w[tr] / w[tr].sum()) * 1000)[:, None]
        for j in range(S.shape[0]):
            y = S[j, tr]
            if y.min() == y.max():
                P[j, te] = y[0]
                continue
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(E[tr], y, sample_weight=w[tr] * len(tr))
            P[j, te] = lr.predict_proba(E[te])[:, 1]
    return (P, K) if C is not None else P


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=Path)
    a = ap.parse_args()
    prices = json.loads(a.prices.read_text()) if a.prices else None
    items, models, S, C, A = load(None, prices)
    keep = ~np.isnan(S).any(0)
    ki = np.where(keep)[0]
    items = [items[i] for i in ki]
    S, C = S[:, keep], np.nan_to_num(C[:, keep])
    A = [[ans[i] for i in ki] for ans in A]
    bc = collections.Counter(it["benchmark"] for it in items)
    w = np.array([1.0 / bc[it["benchmark"]] for it in items])
    w = w / w.sum()
    short = [m.split("/")[-1] for m in models]
    n, M = S.shape[1], len(models)
    ar = np.arange(n)

    E = embed(items)
    P, mcost = fold_probs(E, S, w, C)
    print("out-of-fold calibration (weighted Brier per model):",
          {short[j]: round(float(w @ (P[j] - S[j]) ** 2), 3) for j in range(M)})

    def report(tag, pick, extra_cost=0.0):
        acc = w @ S[pick, ar]
        cost = w @ C[pick, ar] * 1000 + extra_cost
        return arena(acc, cost), acc, cost, tag

    rows = []
    for j in range(M):
        rows.append(report(f"single {short[j]}", np.full(n, j)))
    # learned router over all 8 models
    for lam in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        U = P - lam * mcost
        rows.append(report(f"router all8 lam={lam}", U.argmax(0)))
    # learned cascade: drafts d1,d2 always paid; agree -> d1; else router picks among the rest
    idx = {s: j for j, s in enumerate(short)}
    pairs = [("gpt-6-luna", "gemma-4-31b-it"), ("gpt-6-luna", "claude-haiku-5.5"), ("gemma-4-31b-it", "claude-haiku-5.5"),
             ("deepseek-v4-flash-0731", "gpt-6-luna")]
    for s1, s2 in pairs:
        d1, d2 = idx[s1], idx[s2]
        agree = np.array([A[d1][i] != "" and A[d1][i] == A[d2][i] for i in range(n)])
        rest = [j for j in range(M) if j not in (d1, d2)]
        for lam in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
            U = P[rest] - lam * mcost[rest]
            esc = np.array(rest)[U.argmax(0)]
            pick = np.where(agree, d1, esc)
            acc = w @ S[pick, ar]
            cost = w @ (C[d1] + C[d2] + np.where(agree, 0, C[esc, ar])) * 1000
            rows.append((arena(acc, cost), acc, cost, f"cascade {s1}+{s2} -> router lam={lam} (agree {w @ agree:.0%})"))
    # three-draft majority: luna, gemma, haiku; 2-of-3 agree -> that answer, else escalate to the router's pick
    t = [idx["gpt-6-luna"], idx["gemma-4-31b-it"], idx["claude-haiku-5.5"]]
    rest = [j for j in range(M) if j not in t]
    for lam in (0.0, 0.02, 0.05):
        U = P[rest] - lam * mcost[rest]
        esc = np.array(rest)[U.argmax(0)]
        pick = np.empty(n, dtype=int)
        maj = np.zeros(n, bool)
        for i in range(n):
            ans = [A[j][i] for j in t]
            c = collections.Counter(x for x in ans if x)
            if c and c.most_common(1)[0][1] >= 2:
                top = c.most_common(1)[0][0]
                pick[i] = t[ans.index(top)]
                maj[i] = True
            else:
                pick[i] = esc[i]
        acc = w @ S[pick, ar]
        cost = w @ (C[t].sum(0) + np.where(maj, 0, C[esc, ar])) * 1000
        rows.append((arena(acc, cost), acc, cost, f"majority luna+gemma+haiku -> router lam={lam} (maj {w @ maj:.0%})"))
    oracle = (S * 10 - C).argmax(0)
    rows.append(report("ORACLE (upper bound)", oracle))
    for sc, acc, cost, tag in sorted(rows, reverse=True)[:25]:
        print(f"{sc:6.2f}  acc {acc * 100:5.1f}%  ${cost:6.3f}/1K  {tag}")
