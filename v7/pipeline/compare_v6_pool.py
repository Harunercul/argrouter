"""Diagnostic (not a decision): the v6 pool (gemini-3.7-flash@low + deepseek-v4.1-flash) scored with the v7
objective and the same leave-one-wrapper-out CV, so the v7 router's gain over v6 can be read on our own set."""
import collections, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fit_router_v7 as f

items, models, S, C, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
cands = [j for j, m in enumerate(models) if m not in skip]
bc = collections.Counter(it["benchmark"] for it in items)
w_eq = np.array([1.0 / bc[it["benchmark"]] for it in items]); w_eq /= w_eq.sum()
w = w_eq * (f.HARD_FLOOR + 1 - S[ref].mean(0)); w[np.nan_to_num(S[cands]).max(0) == 0] = 0; w /= w.sum()
S, C = np.nan_to_num(S), np.nan_to_num(C)
texts, E = f.embed_all(items)
L = {k: f.length_feature(v) for k, v in texts.items()}
for name, pool_names in [("v6", ["google/gemini-3.7-flash@low", "deepseek/deepseek-v4.1-flash"])]:
    pool = [models.index(m) for m in pool_names]
    sc = collections.defaultdict(list)
    for seed in (0, 1):
        for held in f.RENDER:
            pk = f.oof(E, L, S, C, w, pool, [r for r in f.RENDER if r != held], held, seed)
            for lam, (s, acc, cost, se) in f.pool_scores(pool, pk, S, C, w, w_eq).items():
                sc[lam].append((s, acc, cost))
    for lam in f.LAMS:
        a = np.array(sc[lam]).mean(0)
        print(f"{name} lam {lam}: hard {a[0]:.2f}  acc {a[1]*100:.1f}%  ${a[2]:.3f}/1K")
