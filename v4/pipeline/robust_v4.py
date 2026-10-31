"""Prompt-format robustness: fit the router on variant-A labels of every row outside the 840-item robustness set,
then score its picks on those 840 items under variant A and under variant B labels (route_data/v4/robust_b),
next to every single model. Equal benchmark weights. Nothing here reads RouterArena data.

    python bench/v4/robust_v4.py --prices route_data/v4/prices_standard.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze_full_v4 as af  # noqa: E402
from explore_v4 import embed  # noqa: E402

POOL = ["google/gemma-4-31b-it", "openai/gpt-6-luna", "google/gemini-3.1-flash-lite@none",
        "anthropic/claude-haiku-5.5", "deepseek/deepseek-v4-flash-0731", "google/gemini-3-flash-preview@none"]

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression, Ridge

    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=Path)
    a = ap.parse_args()
    prices = json.loads(a.prices.read_text()) if a.prices else None
    items, models, S, C, _ = af.load(None, prices)
    keep = ~np.isnan(S).any(0)
    ki = np.where(keep)[0]
    items = [items[i] for i in ki]
    S, C = S[:, keep], np.nan_to_num(C[:, keep])
    E = embed(items)
    sel = [models.index(m) for m in POOL]
    S, C = S[sel], C[sel]
    rid = {json.loads(x)["id"] for x in (af.ROOT / "route_data/v4/robust_items.jsonl").read_text().split("\n") if x.strip()}
    te = np.array([it["id"] in rid for it in items])
    tr = ~te
    bc = collections.Counter(it["benchmark"] for it in items)
    w = np.array([1.0 / bc[it["benchmark"]] for it in items])

    af.MATRIX = af.ROOT / "route_data/v4/robust_b"
    _, mb, SB, CB, _ = af.load(None, prices)
    selb = [mb.index(m) for m in POOL]
    SB, CB = SB[selb][:, ki][:, te], np.nan_to_num(CB[selb][:, ki][:, te])
    SA, CA = S[:, te], C[:, te]
    wt = w[te] / w[te].sum()
    ok = ~np.isnan(SB).any(0)
    print(f"robust items {te.sum()}, answered under B by all: {ok.sum()}")
    SA, CA, SB, CB, wt = SA[:, ok], CA[:, ok], SB[:, ok], CB[:, ok], wt[ok] / wt[ok].sum()
    Et = E[te][ok]

    P = []
    for j in range(len(POOL)):
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(E[tr], S[j, tr], sample_weight=w[tr] / w[tr].mean())
        P.append(lr.predict_proba(Et)[:, 1])
    P = np.array(P)
    # per-row expected cost ($/1K): ridge on log cost over embedding + log prompt length, lognormal mean
    L = np.log1p([len((it.get("context") or "") + it["question"] + " ".join(it.get("options") or [])) for it in items])
    X = np.hstack([E, L[:, None] / 10])
    K = []
    for j in range(len(POOL)):
        y = np.log(C[j, tr] * 1000 + 1e-4)
        r = Ridge(alpha=1.0).fit(X[tr], y, sample_weight=w[tr] / w[tr].mean())
        s2 = np.average((y - r.predict(X[tr])) ** 2, weights=w[tr])
        K.append(np.exp(r.predict(X[te][ok]) + s2 / 2))
    K = np.array(K)
    n = SA.shape[1]
    ar = np.arange(n)
    print(f"{'':44} {'variant A':>22} {'variant B':>22}")
    for j, m in enumerate(POOL):
        ra = af.arena(wt @ SA[j], wt @ CA[j] * 1000)
        rb = af.arena(wt @ SB[j], wt @ CB[j] * 1000)
        print(f"single {m:37} {wt @ SA[j] * 100:5.1f}% ${wt @ CA[j] * 1000:5.3f} {ra:6.2f}   "
              f"{wt @ SB[j] * 100:5.1f}% ${wt @ CB[j] * 1000:5.3f} {rb:6.2f}")
    for lam in (0.08, 0.12, 0.16):
        pick = (P - lam * K).argmax(0)
        ra = af.arena(wt @ SA[pick, ar], wt @ CA[pick, ar] * 1000)
        rb = af.arena(wt @ SB[pick, ar], wt @ CB[pick, ar] * 1000)
        print(f"router lam={lam:<36} {wt @ SA[pick, ar] * 100:5.1f}% ${wt @ CA[pick, ar] * 1000:5.3f} {ra:6.2f}   "
              f"{wt @ SB[pick, ar] * 100:5.1f}% ${wt @ CB[pick, ar] * 1000:5.3f} {rb:6.2f}")
