"""Fit and freeze the argrouter v7 router on OUR calibration set only (pre-registration part 8).

Items: the 4,264 v4 calibration items (route_data/v4src/all.jsonl) + the 200 QANTA start-of-question items
(route_data/v7/qanta_s1.jsonl). Labels: route_data/v4/full_matrix (8 reference models), route_data/v5/matrix,
route_data/v7/matrix (new v7 candidates) and route_data/v7/s1 (every candidate on the QANTA first-sentence items),
graded with the v4 grader; code from the sandbox grades in LCB. Nothing here reads RouterArena data.

Same procedure as bench/v5/fit_router_v5.py (forward selection, leave-one-wrapper-out lam, RouterV4 format), with
part 8's changes: QANTA's benchmark share is split equally between full and first-sentence questions (both count as
benchmark "qanta"), and items that no candidate answers correctly get weight 0.

    python bench/v7/fit_router_v7.py --prices route_data/v7/prices_ra.json --only <candidates> [--out ...]
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v5"))
import fit_router_v4  # noqa: E402
from analyze_full_v4 import ITEMS, ROOT, arena  # noqa: E402
from fit_router_v4 import EMB_MODEL, MAX_CHARS, embed_all, fit, length_feature  # noqa: E402
from fit_router_v5 import HARD_FLOOR, LAMS, MAX_POOL, MIN_GAIN, RENDER, jsonl, oof, pool_scores  # noqa: E402
from grade_v4 import grade  # noqa: E402

S1 = ROOT / "route_data" / "v7" / "qanta_s1.jsonl"
REF = ROOT / "route_data" / "v4" / "full_matrix"
FOLDERS = [REF, ROOT / "route_data" / "v5" / "matrix", ROOT / "route_data" / "v7" / "matrix",
           ROOT / "route_data" / "v7" / "s1"]
LCB = [ROOT / "route_data" / p / "lcb_graded.jsonl" for p in ("v4", "v5", "v7")]
fit_router_v4.CACHE = ROOT / "route_data" / "v7" / "emb_cache_v7.npz"


def load_v7(prices: dict):
    """-> items, models, S, C (nan where missing), ref (bool per model), skip {model: why}."""
    items = jsonl(ITEMS) + jsonl(S1)
    idx = {it["id"]: i for i, it in enumerate(items)}
    n = len(items)
    lcb = collections.defaultdict(dict)
    for p in LCB:
        if p.exists():
            for r in jsonl(p):
                lcb[r["model"]][r.get("id") or r["gidx"]] = float(r["score"])
    s, c, seen, is_ref = {}, {}, {}, set()
    for folder in FOLDERS:
        for f in sorted(folder.glob("*.jsonl")):
            for r in jsonl(f):
                if r.get("error") or r["gidx"] not in idx:
                    continue
                m = r["model"]
                if m not in s:
                    s[m], c[m], seen[m] = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
                if folder == REF:
                    is_ref.add(m)
                i = idx[r["gidx"]]
                it = items[i]
                g = lcb.get(m, {}).get(it["id"]) if it["task"] == "code" else grade(it, r.get("response") or "")
                if g is not None:
                    s[m][i] = g
                seen[m][i] = True
                price = prices.get(m.split("@")[0])
                if price:
                    c[m][i] = ((r.get("input_tokens") or 0) * price[0] + (r.get("output_tokens") or 0) * price[1]) / 1e6
    models = sorted(s, key=lambda m: (m not in is_ref, m))
    skip = {}
    for m in models:
        why = []
        if not seen[m].all():
            why.append(f"incomplete {seen[m].sum()}/{n}")
        miss = np.where(seen[m] & np.isnan(s[m]))[0]
        if len(miss):
            code = all(items[i]["task"] == "code" for i in miss)
            why.append(f"code not graded ({len(miss)} items)" if code else f"{len(miss)} items ungraded")
        if prices.get(m.split("@")[0]) is None:
            why.append("no price")
        if why:
            skip[m] = ", ".join(why)
    return (items, models, np.array([s[m] for m in models]), np.array([c[m] for m in models]),
            np.array([m in is_ref for m in models]), skip)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "route_data" / "v7" / "router_v7.npz")
    ap.add_argument("--dry", action="store_true", help="candidate completeness + single-model table only")
    ap.add_argument("--only", nargs="+", help="restrict candidates to these model ids")
    a = ap.parse_args()
    t0 = time.time()
    items, models, S, C, ref, skip = load_v7(json.loads(a.prices.read_text()))
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
    noise = np.nan_to_num(S[cands]).max(0) == 0
    w[noise] = 0
    w /= w.sum()
    n, ar = len(items), np.arange(len(items))
    print(f"{n} items ({sum(i['id'].endswith('/s1') for i in items)} QANTA first-sentence), {len(cands)} candidates; "
          f"{noise.sum()} items no candidate answers (weight 0); hard weights: floor {HARD_FLOOR}, "
          f"effective n {1 / (w @ w):.0f} (equal {1 / (w_eq @ w_eq):.0f})  [{time.time() - t0:.0f}s]")

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
