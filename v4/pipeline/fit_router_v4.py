"""Fit and freeze the argrouter v4 router on OUR v4 calibration set only (route_data/v4src/all.jsonl, 4,264 items,
equal benchmark weights, standard list prices). Nothing here reads RouterArena data, templates or configs.

Router: per model, P(correct) = logistic regression on a bge-small embedding of the query text, and expected cost
($/1K) = exp(ridge on log cost over [embedding, log prompt length] + residual variance / 2). Route to
argmax_j P_j - lam * K_j over the pool.

Wrapper invariance: the router sees whatever text a caller sends, instruction wrapper included. It is trained on three
renderings of every item (bare content, our prompt variant A, our prompt variant B) so that no single wrapper's
wording is learned; at inference the text is embedded as received.

lam is chosen by leave-one-wrapper-out cross-validation (train on two renderings, test on the third; 5 folds x 2 seeds):
the grid value with the highest mean score. Then the final router is fitted on all rows and saved.

    python bench/v4/fit_router_v4.py --out route_data/v4/router_v4.npz
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_full_v4 import ROOT, arena, load  # noqa: E402
from prompts_v4 import render  # noqa: E402
from prompts_v4b import SYSTEM, render_b  # noqa: E402

POOL = ["google/gemma-4-31b-it", "openai/gpt-6-luna", "google/gemini-3.1-flash-lite@none",
        "anthropic/claude-haiku-5.5", "deepseek/deepseek-v4-flash-0731", "google/gemini-3-flash-preview@none"]
LAMS = (0.06, 0.08, 0.12, 0.16, 0.24)
EMB_MODEL = "BAAI/bge-small-en-v1.5"
MAX_CHARS = 4000  # last 4000 characters of the text are embedded
CACHE = ROOT / "route_data" / "v4" / "emb_router_v4.npz"


def bare(it: dict) -> str:
    return ((it.get("context") or "") + "\n" + it["question"] + "\n" + "\n".join(it.get("options") or [])).strip()


def length_feature(texts: list[str]) -> np.ndarray:
    return (np.log1p([len(t) for t in texts]) / 10)[:, None]


def embed_all(items: list[dict]):
    texts = {"bare": [bare(it) for it in items], "A": [render(it) for it in items],
             "B": [SYSTEM + "\n" + render_b(it) for it in items]}
    if CACHE.exists():
        z = np.load(CACHE)
        if z["n"] == len(items):
            return texts, {k: z[k] for k in texts}
    from fastembed import TextEmbedding

    m = TextEmbedding(EMB_MODEL)
    E = {k: np.array(list(m.embed([t[-MAX_CHARS:] for t in v]))) for k, v in texts.items()}
    np.savez(CACHE, n=len(items), **E)
    return texts, E


def fit(Xs, Ls, S, C, w):
    """Fit per-model classifiers and cost regressors on stacked renderings."""
    from sklearn.linear_model import LogisticRegression, Ridge

    X = np.vstack(Xs)
    XL = np.hstack([X, np.vstack(Ls)])
    rep = len(Xs)
    ww = np.tile(w, rep)
    ww = ww / ww.mean()
    params = []
    for j in range(S.shape[0]):
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(X, np.tile(S[j], rep), sample_weight=ww)
        y = np.tile(np.log(C[j] * 1000 + 1e-4), rep)
        rg = Ridge(alpha=1.0).fit(XL, y, sample_weight=ww)
        s2 = float(np.average((y - rg.predict(XL)) ** 2, weights=ww))
        params.append((lr.coef_[0], lr.intercept_[0], rg.coef_, rg.intercept_, s2))
    return params


def predict(params, E, L):
    P = np.array([1 / (1 + np.exp(-(E @ a + b))) for a, b, *_ in params])
    EL = np.hstack([E, L])
    K = np.array([np.exp(EL @ c + d + s2 / 2) for _, _, c, d, s2 in params])
    return P, K


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "route_data" / "v4" / "router_v4.npz")
    a = ap.parse_args()
    prices = json.loads((ROOT / "route_data" / "v4" / "prices_standard.json").read_text())
    items, models, S, C, _ = load(ROOT / "route_data" / "v4" / "lcb_graded.jsonl", prices)
    assert not np.isnan(S).any(), "every item must be graded for every model"
    sel = [models.index(m) for m in POOL]
    S, C = S[sel], np.nan_to_num(C[sel])
    bc = collections.Counter(it["benchmark"] for it in items)
    w = np.array([1.0 / bc[it["benchmark"]] for it in items])
    w = w / w.sum()
    n = len(items)
    ar = np.arange(n)
    texts, E = embed_all(items)
    L = {k: length_feature(v) for k, v in texts.items()}

    singles = {m.split("/")[-1]: arena(w @ S[j], w @ C[j] * 1000) for j, m in enumerate(POOL)}
    print("single models:", {k: round(v, 2) for k, v in singles.items()})
    scores = collections.defaultdict(list)
    for seed in (0, 1):
        order = np.random.default_rng(seed).permutation(n)
        for held in ("bare", "A", "B"):
            train_r = [r for r in ("bare", "A", "B") if r != held]
            picks = {lam: np.zeros(n, dtype=int) for lam in LAMS}
            for k in range(5):
                te = order[k::5]
                tr = np.setdiff1d(order, te)
                p = fit([E[r][tr] for r in train_r], [L[r][tr] for r in train_r], S[:, tr], C[:, tr], w[tr])
                P, K = predict(p, E[held][te], L[held][te])
                for lam in LAMS:
                    picks[lam][te] = (P - lam * K).argmax(0)
            for lam in LAMS:
                pk = picks[lam]
                scores[lam].append(arena(w @ S[pk, ar], w @ C[pk, ar] * 1000))
                print(f"seed {seed} held-out wrapper {held:4} lam {lam}: {scores[lam][-1]:.2f}  "
                      f"acc {w @ S[pk, ar] * 100:.1f}% ${w @ C[pk, ar] * 1000:.3f}/1K", flush=True)
    lam = max(LAMS, key=lambda x: np.mean(scores[x]))
    print("mean over held-out wrappers:", {k: round(float(np.mean(v)), 2) for k, v in scores.items()}, "-> lam", lam)

    params = fit([E[r] for r in ("bare", "A", "B")], [L[r] for r in ("bare", "A", "B")], S, C, w)
    np.savez(a.out,
             pool=np.array(POOL), lam=lam, emb_model=EMB_MODEL, max_chars=MAX_CHARS,
             lr_coef=np.array([p[0] for p in params]), lr_intercept=np.array([p[1] for p in params]),
             cost_coef=np.array([p[2] for p in params]), cost_intercept=np.array([p[3] for p in params]),
             cost_s2=np.array([p[4] for p in params]))
    print(f"saved {a.out}  sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
