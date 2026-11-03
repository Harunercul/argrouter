"""Fit and freeze the argrouter cluster router (pre-registration part 9) on OUR calibration set only.

Objective (RouterArena-calibrated, bench/v7/calib_ra.py): equal benchmark shares x (0.25 + reference error), items no
candidate answers get 0, costs x 0.6. Pool = consensus of 10 nested half-split selections (bench/v7/nested_cal.py);
embedding and lam from bench/v7/final_cal.py. The router: k-means (k=30) on L2-normalised embeddings of the three
renderings of every item; per cluster and pool model, weighted mean correctness and mean cost ($/1K), each shrunk to
the model's overall mean with pseudo-count 5; a query goes to its nearest centroid and then to argmax P - lam * K.

    python bench/v7/fit_router_cluster.py --out route_data/v7/router_cluster.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare_emb as ce  # noqa: E402
import fit_router_v7 as f  # noqa: E402
from final_cal import POOL  # noqa: E402
from nested_cal import KIND  # noqa: E402

EMB = "google/gemini-embedding-2"
LAM, K, PSEUDO = 0.01, 30, 5.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=f.ROOT / "route_data/v7/router_cluster.npz")
    a = ap.parse_args()
    from sklearn.cluster import KMeans
    items, models, S, C, ref, skip = f.load_v7(json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text()))
    cands = [j for j, m in enumerate(models) if m not in skip]
    w = ce.weights(KIND, items, S, ref, cands)
    S, C = np.nan_to_num(S), np.nan_to_num(C)
    pool = [models.index(m) for m in POOL]
    E = ce.load_emb(EMB.replace("/", "__"))
    X = np.vstack([E[r] for r in f.RENDER])
    ww = np.tile(w, len(f.RENDER))
    ww = ww / ww.mean()
    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(X)
    lab = km.labels_
    sw = np.bincount(lab, ww, K) + PSEUDO
    P, Kc = np.zeros((len(pool), K)), np.zeros((len(pool), K))
    for i, j in enumerate(pool):
        y, c = np.tile(S[j], len(f.RENDER)), np.tile(C[j] * 1000, len(f.RENDER))
        P[i] = (np.bincount(lab, ww * y, K) + PSEUDO * np.average(y, weights=ww)) / sw
        Kc[i] = (np.bincount(lab, ww * c, K) + PSEUDO * np.average(c, weights=ww)) / sw
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, pool=np.array(POOL), lam=LAM, emb_model=EMB, max_chars=f.MAX_CHARS,
             centroids=km.cluster_centers_.astype(np.float32), p=P, k=Kc)
    pick = np.array(POOL)[(P - LAM * Kc).argmax(0)]
    sizes = np.bincount(lab, minlength=K) // len(f.RENDER)
    print(f"pool {POOL}\nlam {LAM}, k {K}, pseudo {PSEUDO}, embedding {EMB}, objective {KIND}")
    for c in np.argsort(-sizes):
        top = __import__("collections").Counter(items[i]["benchmark"] for i in np.where(lab[: len(items)] == c)[0])
        print(f"  cluster {c:2} n={sizes[c]:4} -> {pick[c]:30} " + " ".join(f"{b}:{v}" for b, v in top.most_common(3)))
    print("clusters per model:", {m: int((pick == m).sum()) for m in POOL})
    print(f"saved {a.out}  sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
