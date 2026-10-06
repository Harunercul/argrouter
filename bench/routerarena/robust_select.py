"""Zorluk agirlikli, saglam havuz + lambda secimi (sadece dis veri).

Zorluk vekili: tam matrisi olan TUM modellerden kacinin soruyu yanlis bildigi (RouterArena'nin
zorluk tanimiyla ayni fikir: kac model dogru biliyor). Agirlik = aile payi x (1 + alpha * zorluk),
aile ici yeniden olceklenir (aile paylari degismez). Birden fazla alpha'da degerlendirip en kotu
durumda en iyi olan secilir.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import offline_eval as oe  # noqa: E402
import select_pool as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ALPHAS = [0, 1, 2, 4, 8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--judges", nargs="+", required=True, help="zorluk icin kullanilan modeller")
    a = ap.parse_args()

    allm = list(dict.fromkeys(a.candidates + a.judges))
    items, models, S, IN, OUT = oe.load(ROOT / "route_data" / "train_all.jsonl", allm)
    P = oe.price_table(models)
    C = IN * np.array([P[m][0] for m in models]) + OUT * np.array([P[m][1] for m in models])
    E = oe.embed(items)
    jcols = [models.index(m) for m in a.judges]
    diff = 1 - S[:, jcols].mean(1)  # 0 = herkes bildi, 1 = kimse bilemedi
    fam = np.array([it["Dataset name"].split("_")[0] for it in items])
    base = oe.ra_weights(items)
    print(
        f"{len(items)} soru; zorluk ortalamasi {diff.mean():.3f}; zor(>%70 yanlis) payi "
        f"{(diff > 0.7).mean():.1%}",
        flush=True,
    )

    def weights(alpha):
        w = base * (1 + alpha * diff)
        for f in set(fam):  # aile paylarini koru
            m = fam == f
            w[m] *= base[m].sum() / w[m].sum()
        return w

    fold = np.random.default_rng(0).permutation(len(items)) % 5
    cand = [models.index(m) for m in a.candidates]
    PS, PC = sp.fold_predictions(E, S, C, fold)

    # her alpha'da: zorluk agirlikli tek-model dogrulugu (ne kadar zorlasiyor?)
    g = (
        models.index("google/gemini-3.8-flash@low")
        if "google/gemini-3.8-flash@low" in models
        else cand[0]
    )
    for al in ALPHAS:
        oe.W = weights(al)
        acc, c1k, sc = oe.evaluate(np.full(len(items), g), S, C)
        print(f"alpha={al}: gemini-3.8-flash@low acc={acc:.3f}", flush=True)

    # aday havuzlar: 2..5 modelli tum alt kumeler (adaylardan)
    results = []
    for k in range(2, 6):
        for pool in itertools.combinations(cand, k):
            cols = list(pool)
            row = []
            for al in ALPHAS:
                oe.W = weights(al)
                row.append(sp.cv_score(S, C, PS, PC, cols))
            results.append((min(r[0] for r in row), pool, row))
    results.sort(key=lambda r: -r[0])
    print("\n== En kotu durumda en iyi 8 havuz (her alpha icin arena / lambda) ==")
    for worst, pool, row in results[:8]:
        names = ", ".join(models[j].split("/")[1] for j in pool)
        cells = "  ".join(
            f"a{al}:{r[0] * 100:.2f}/l{r[1]}" for al, r in zip(ALPHAS, row, strict=True)
        )
        print(f"min={worst * 100:.2f} | {names}\n      {cells}")

    # en iyi havuz icin sabit lambda'larda saglamlik
    best_pool = list(results[0][1])
    print("\n== En iyi havuzda sabit lambda: her alpha'daki puan (acc, $/1k) ==")
    for lam in sp.LAMBDAS:
        cells = []
        worst = 1.0
        for al in ALPHAS:
            oe.W = weights(al)
            choice = (PS[:, best_pool] - lam * PC[:, best_pool]).argmax(1)
            acc, c1k, sc = oe.evaluate(choice, S[:, best_pool], C[:, best_pool])
            worst = min(worst, sc)
            cells.append(f"{sc * 100:.2f}({acc:.3f},${c1k:.2f})")
        print(f"lam={lam:4} min={worst * 100:.2f} | " + "  ".join(cells))


if __name__ == "__main__":
    main()
