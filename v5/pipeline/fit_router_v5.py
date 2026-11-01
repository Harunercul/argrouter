"""Fit and freeze the argrouter v5 router on OUR v4 calibration set only (route_data/v4src/all.jsonl, 4,264 items).
Labels: route_data/v4/full_matrix (8 reference models) + route_data/v5/matrix (new models), graded with the v4 grader;
cost from tokens at the --prices list prices. Nothing here reads RouterArena data, templates or configs.

Router and wrapper invariance as in bench/v4/fit_router_v4.py (fit/predict/embed_all are imported from there).
Differences: "hard" weights = equal per benchmark x (HARD_FLOOR + mean error of the 8 reference models), and the pool
is chosen by greedy forward selection over every complete, priced candidate (5-fold CV, one seed, train on all three
renderings, test on A; objective = max over lam of the arena score). lam for the final pool is then chosen by
leave-one-wrapper-out CV (5 folds x 2 seeds) and the router is fitted on all rows and saved in the v4 npz format.

    python bench/v5/fit_router_v5.py --prices route_data/v4/prices_standard.json [--out route_data/v5/router_v5.npz]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v4"))
from analyze_full_v4 import ITEMS, ROOT, arena  # noqa: E402
from fit_router_v4 import EMB_MODEL, MAX_CHARS, embed_all, fit, length_feature, predict  # noqa: E402
from grade_v4 import grade  # noqa: E402

REF = ROOT / "route_data" / "v4" / "full_matrix"
NEW = ROOT / "route_data" / "v5" / "matrix"
LCB = [ROOT / "route_data" / "v4" / "lcb_graded.jsonl", ROOT / "route_data" / "v5" / "lcb_graded.jsonl"]
LAMS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
HARD_FLOOR = 0.5  # pre-registration part 4, amendment: v4 router OOF accuracy closest to its RouterArena accuracy
MIN_GAIN, MAX_POOL = 0.10, 6
RENDER = ("bare", "A", "B")


def jsonl(p: Path):
    return [json.loads(x) for x in p.read_text().split("\n") if x.strip()]


def load_v5(prices: dict):
    """-> items, models, S, C (nan where missing; C nan for unpriced models), ref (bool per model), skip {model: why}."""
    items = jsonl(ITEMS)
    idx = {it["id"]: i for i, it in enumerate(items)}
    n = len(items)
    lcb = collections.defaultdict(dict)
    for p in LCB:
        if p.exists():
            for r in jsonl(p):
                lcb[r["model"]][r.get("id") or r["gidx"]] = float(r["score"])
    models, S, C, ref, skip = [], [], [], [], {}
    for folder in (REF, NEW):
        for f in sorted(folder.glob("*.jsonl")):
            s, c, seen = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
            name = f.stem.replace("__", "/", 1)
            price = None
            for r in jsonl(f):
                if r.get("error") or r["gidx"] not in idx:
                    continue
                name = r["model"]
                price = prices.get(name.split("@")[0])
                i = idx[r["gidx"]]
                it = items[i]
                g = lcb.get(name, {}).get(it["id"]) if it["task"] == "code" else grade(it, r.get("response") or "")
                if g is not None:
                    s[i] = g
                seen[i] = True
                if price:
                    c[i] = ((r.get("input_tokens") or 0) * price[0] + (r.get("output_tokens") or 0) * price[1]) / 1e6
            if name in models:
                print(f"  duplicate {name} in {f}, ignored")
                continue
            why = []
            if not seen.all():
                why.append(f"incomplete {seen.sum()}/{n}")
            miss = np.where(seen & np.isnan(s))[0]
            if len(miss):
                code = all(items[i]["task"] == "code" for i in miss)
                why.append(f"code not graded ({len(miss)} items)" if code else f"{len(miss)} items ungraded")
            if prices.get(name.split("@")[0]) is None:
                why.append("no price")
            if why:
                skip[name] = ", ".join(why)
            models.append(name)
            S.append(s)
            C.append(c)
            ref.append(folder == REF)
    return items, models, np.array(S), np.array(C), np.array(ref), skip


def oof(E, L, S, C, w, js, train_r, test_r, seed, folds=5):
    """Out-of-fold (P, K) per model j on rendering test_r; per-model fits are independent so they are cached by j."""
    n = S.shape[1]
    order = np.random.default_rng(seed).permutation(n)
    out = {j: (np.zeros(n), np.zeros(n)) for j in js}
    for k in range(folds):
        te = order[k::folds]
        tr = np.setdiff1d(order, te)
        for j in js:
            p = fit([E[r][tr] for r in train_r], [L[r][tr] for r in train_r], S[j:j + 1, tr], C[j:j + 1, tr], w[tr])
            P, K = predict(p, E[test_r][te], L[test_r][te])
            out[j][0][te], out[j][1][te] = P[0], K[0]
    return out


def pool_scores(pool, pk, S, C, w, w_eq=None):
    """{lam: (arena under w, acc, $/1K, arena under w_eq)} for routing argmax P - lam K over pool."""
    ar = np.arange(S.shape[1])
    P, K = np.array([pk[j][0] for j in pool]), np.array([pk[j][1] for j in pool])
    res = {}
    for lam in LAMS:
        pick = np.array(pool)[(P - lam * K).argmax(0)]
        s, c = S[pick, ar], C[pick, ar] * 1000
        res[lam] = (arena(w @ s, w @ c), w @ s, w @ c, arena(w_eq @ s, w_eq @ c) if w_eq is not None else None)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "route_data" / "v5" / "router_v5.npz")
    ap.add_argument("--dry", action="store_true", help="candidate completeness + single-model table only")
    ap.add_argument("--only", nargs="+", help="restrict candidates to these model ids")
    a = ap.parse_args()
    t0 = time.time()
    items, models, S, C, ref, skip = load_v5(json.loads(a.prices.read_text()))
    assert ref.sum() == 8 and not np.isnan(S[ref]).any(), "the 8 reference models must be fully graded"
    for m in models:
        if a.only and m not in a.only and m not in skip:
            skip[m] = "not in --only"
        print(f"  {'skip' if m in skip else 'ok  '} {m:40} {skip.get(m, '')}")
    cands = [j for j, m in enumerate(models) if m not in skip]
    bc = collections.Counter(it["benchmark"] for it in items)
    w_eq = np.array([1.0 / bc[it["benchmark"]] for it in items])
    w_eq /= w_eq.sum()
    w = w_eq * (HARD_FLOOR + 1 - S[ref].mean(0))
    w /= w.sum()
    n, ar = len(items), np.arange(len(items))
    print(f"{n} items, {len(cands)} candidates; hard weights: floor {HARD_FLOOR}, effective n {1 / (w @ w):.0f} "
          f"(equal {1 / (w_eq @ w_eq):.0f})  [{time.time() - t0:.0f}s]")

    print("\nmodel                                    hard: acc%  $/1K  arena | equal: acc%  $/1K  arena")
    for j in sorted(cands, key=lambda j: -arena(w @ S[j], w @ C[j] * 1000)):
        h, e = (w @ S[j], w @ C[j] * 1000), (w_eq @ S[j], w_eq @ C[j] * 1000)
        print(f"{models[j]:40} {h[0] * 100:9.1f} {h[1]:5.3f} {arena(*h):6.2f} | "
              f"{e[0] * 100:10.1f} {e[1]:5.3f} {arena(*e):6.2f}")
    if a.dry:
        sys.exit()
    S, C = np.nan_to_num(S), np.nan_to_num(C)
    texts, E = embed_all(items)
    L = {k: length_feature(v) for k, v in texts.items()}

    pk = oof(E, L, S, C, w, cands, RENDER, "A", seed=0)
    print(f"\nselection CV predictions for {len(cands)} candidates  [{time.time() - t0:.0f}s]")
    pool, best = [], -np.inf
    while len(pool) < MAX_POOL and len(pool) < len(cands):
        trial = {j: max(v[0] for v in pool_scores(pool + [j], pk, S, C, w).values()) for j in cands if j not in pool}
        j = max(trial, key=trial.get)
        if trial[j] - best < MIN_GAIN:
            print(f"  stop: best addition {models[j]} {trial[j]:.2f} (+{trial[j] - best:.2f} < {MIN_GAIN})")
            break
        pool.append(j)
        best = trial[j]
        print(f"  + {models[j]:40} objective {best:.2f}   runners-up: " +
              ", ".join(f"{models[k].split('/')[-1]} {trial[k]:.2f}" for k in sorted(trial, key=trial.get)[-4:-1][::-1]))
    POOL = [models[j] for j in pool]
    print("pool:", POOL)

    scores, eq = collections.defaultdict(list), collections.defaultdict(list)
    for seed in (0, 1):
        for held in RENDER:
            pk = oof(E, L, S, C, w, pool, [r for r in RENDER if r != held], held, seed)
            for lam, (sc, acc, cost, se) in pool_scores(pool, pk, S, C, w, w_eq).items():
                scores[lam].append(sc)
                eq[lam].append(se)
                print(f"seed {seed} held-out wrapper {held:4} lam {lam}: {sc:.2f}  acc {acc * 100:.1f}% "
                      f"${cost:.3f}/1K  (equal-weight {se:.2f})", flush=True)
    lam = max(LAMS, key=lambda x: np.mean(scores[x]))
    print("\nlam      hard   equal   (mean over 2 seeds x 3 held-out wrappers)")
    for x in LAMS:
        print(f"{x:<7} {np.mean(scores[x]):6.2f} {np.mean(eq[x]):7.2f}{'  <-' if x == lam else ''}")

    sel = np.array(pool)
    params = fit([E[r] for r in RENDER], [L[r] for r in RENDER], S[sel], C[sel], w)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out,
             pool=np.array(POOL), lam=lam, emb_model=EMB_MODEL, max_chars=MAX_CHARS,
             lr_coef=np.array([p[0] for p in params]), lr_intercept=np.array([p[1] for p in params]),
             cost_coef=np.array([p[2] for p in params]), cost_intercept=np.array([p[3] for p in params]),
             cost_s2=np.array([p[4] for p in params]))
    print(f"saved {a.out}  sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}  [{time.time() - t0:.0f}s]")
