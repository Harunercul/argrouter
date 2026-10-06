"""Ileri secimle model havuzu: her adimda CV arena puanini en cok artiran modeli ekle.

    python bench/routerarena/select_pool.py --candidates m1 m2 ... [--max 8]
Sadece dis egitim verisi kullanilir (RouterArena verisi yok).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import offline_eval as oe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LAMBDAS = [0, 5, 10, 20, 30, 50, 80, 120]


def fold_predictions(E, S, C, fold, folds=5):
    """Her modelin tahmini havuzdan bagimsiz: bir kez hesapla, kombinasyonlarda yeniden kullan."""
    PS = np.zeros_like(S)
    PC = np.zeros_like(C)
    for f in range(folds):
        tr, te = fold != f, fold == f
        PS[te] = 0.5 * oe.lr_predict(E[tr], S[tr], E[te]) + 0.5 * oe.knn_predict(
            E[tr], S[tr], E[te], 50
        )
        PC[te] = oe.knn_predict(E[tr], C[tr], E[te], 50)
        print(f"  kat {f + 1}/{folds} hazir", flush=True)
    return PS, PC


def cv_score(S, C, PS, PC, cols):
    """Verilen model kolonlariyla router'in en iyi lambda'daki CV arena puani."""
    best = (-1.0, None, None, None)
    for lam in LAMBDAS:
        choice = (PS[:, cols] - lam * PC[:, cols]).argmax(1)
        acc, c1k, sc = oe.evaluate(choice, S[:, cols], C[:, cols])
        if sc > best[0]:
            best = (sc, lam, acc, c1k)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=0.0005)
    a = ap.parse_args()

    items, models, S, IN, OUT = oe.load(ROOT / "route_data" / "train_all.jsonl", a.candidates)
    P = oe.price_table(models)
    C = IN * np.array([P[m][0] for m in models]) + OUT * np.array([P[m][1] for m in models])
    E = oe.embed(items)
    oe.W = oe.ra_weights(items)
    fold = np.random.default_rng(0).permutation(len(items)) % 5
    print(f"{len(items)} soru (tum adaylarda cevabi olanlar), {len(models)} aday", flush=True)
    PS, PC = fold_predictions(E, S, C, fold)

    chosen: list[int] = []
    current = -1.0
    while len(chosen) < a.max:
        trials = []
        for j in range(len(models)):
            if j in chosen:
                continue
            sc, lam, acc, c1k = cv_score(S, C, PS, PC, chosen + [j])
            trials.append((sc, j, lam, acc, c1k))
        trials.sort(reverse=True)
        sc, j, lam, acc, c1k = trials[0]
        if sc - current < a.min_gain:
            print(f"dur: en iyi aday {models[j]} sadece {100 * (sc - current):+.2f} puan ekliyor")
            break
        chosen.append(j)
        current = sc
        print(
            f"+ {models[j]:42} arena={sc * 100:.2f}  acc={acc:.3f}  ${c1k:.3f}/1k  lam={lam}",
            flush=True,
        )
    print("\nHAVUZ:", " ".join(models[j] for j in chosen))


if __name__ == "__main__":
    main()
