"""R&D: pool selection under the RouterArena-calibrated objective (calib_ra.py): equal benchmark shares x (0.25 +
reference error), noise items 0, costs x 0.6. Under it our set reproduces both disclosed RouterArena results
(v4 73.00 vs 73.56, v6 75.95 vs 76.18), so a held-out score here is read as an expected RouterArena score.

Cluster router (k=30, pseudo 5, mean of the given embeddings' predictions). For each split seed: greedy selection +
lam on one half, scored on the other half, both ways; then the consensus pool (models picked in >= half of the
selections) and the greedy pick on all items, both with lam chosen on all items.

    python bench/v7/nested_cal.py --emb qwen__qwen3-embedding-8b openai__text-embedding-3-large --split-seeds 1 2 3
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
import select_cluster as sc  # noqa: E402
from nested_select import V6, greedy  # noqa: E402

KIND, SCALE = "hard@0.25", 0.6

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", default=["qwen__qwen3-embedding-8b"])
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    sc.COST_SCALE = SCALE
    items, models, S, C, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
    cands = [j for j, m in enumerate(models) if m not in skip]
    w = ce.weights(KIND, items, S, ref, cands)
    S, C = np.nan_to_num(S), np.nan_to_num(C)
    per = [sc.oof_cluster(ce.load_emb(e), S, C, w, 30, 5.0) for e in a.emb]
    R = [(np.mean([p[i][0] for p in per], 0), np.mean([p[i][1] for p in per], 0)) for i in range(len(per[0]))]
    v6 = [models.index(m) for m in V6]
    picked, gains = collections.Counter(), []
    for seed in a.split_seeds:
        rng = np.random.default_rng(seed)
        half = np.zeros(len(items), int)
        for b in sorted({it["benchmark"] for it in items}):
            idx = rng.permutation([i for i, it in enumerate(items) if it["benchmark"] == b])
            half[idx[: len(idx) // 2]] = 1
        for h in (0, 1):
            ws, we = w * (half == h), w * (half != h)
            ws, we = ws / ws.sum(), we / we.sum()
            pool, lam = greedy(R, S, C, ws, cands)
            lam6 = max(sc.score(v6, R, S, C, ws).items(), key=lambda x: x[1][0])[0]
            s1, s6 = sc.score(pool, R, S, C, we)[lam], sc.score(v6, R, S, C, we)[lam6]
            picked.update(pool)
            gains.append(s1[0] - s6[0])
            print(f"seed {seed} half {h}: {[models[j].split('/')[-1] for j in pool]} lam {lam} | other half "
                  f"{s1[0]:.2f} (acc {s1[1] * 100:.1f}% ${s1[2]:.3f}/1K) vs v6 {s6[0]:.2f}  gain {s1[0] - s6[0]:+.2f}",
                  flush=True)
    print(f"\nheld-out gain over the v6 pair: mean {np.mean(gains):+.2f}, sd {np.std(gains, ddof=1):.2f}, "
          f"n {len(gains)}")
    nsel = 2 * len(a.split_seeds)
    print("selection counts:", {models[j]: c for j, c in picked.most_common()})
    for name, pool in (("consensus", [j for j, c in picked.items() if c >= nsel / 2]),
                       ("greedy all", greedy(R, S, C, w, cands)[0]), ("v6 pair", v6)):
        res = sc.score(pool, R, S, C, w)
        lam = max(res, key=lambda x: res[x][0])
        print(f"{name:10} {[models[j] for j in pool]}\n           " +
              " ".join(f"{x}:{res[x][0]:.2f}" for x in sc.LAMS) +
              f"   best lam {lam}: {res[lam][0]:.2f} acc {res[lam][1] * 100:.1f}% ${res[lam][2]:.3f}/1K", flush=True)
