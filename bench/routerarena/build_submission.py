"""Nihai router'i tum egitim verisiyle egit, RouterArena sorulari icin model sec.

    python bench/routerarena/build_submission.py --lam 30 --models ...
Cikti: route_data/submission/{router.npz, picks_full.json, picks_robustness.json}
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import offline_eval as oe  # noqa: E402

from thriftllm.router import Router, embed  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RA = Path.home() / "Desktop" / "RouterArena"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lam", type=float, default=30)
    ap.add_argument("--k", type=int, default=50)
    a = ap.parse_args()

    items, models, S, IN, OUT = oe.load(ROOT / "route_data" / "train_all.jsonl", a.models)
    P = oe.price_table(models)
    C = IN * np.array([P[m][0] for m in models]) + OUT * np.array([P[m][1] for m in models])
    E = oe.embed(items)
    router = Router.fit(models, E, S, C, lam=a.lam, k=a.k)
    out = ROOT / "route_data" / "submission"
    out.mkdir(parents=True, exist_ok=True)
    router.save(out / "router.npz")

    for split, fname in [("full", "router_data.json"), ("robustness", "router_robustness.json")]:
        data = json.loads((RA / "dataset" / fname).read_text())
        prompts = [d["prompt_formatted"] for d in data]
        p, c = router.predict(embed(prompts))
        pick = (p - a.lam * c).argmax(1)
        picks = {d["global index"]: models[j] for d, j in zip(data, pick, strict=True)}
        (out / f"picks_{split}.json").write_text(json.dumps(picks, indent=1))
        cnt = collections.Counter(picks.values())
        exp_cost = float(c[np.arange(len(pick)), pick].mean() * 1000)
        exp_acc = float(p[np.arange(len(pick)), pick].mean())
        print(f"{split}: n={len(picks)}  beklenen acc~{exp_acc:.3f}  beklenen ${exp_cost:.3f}/1k")
        for m, n in cnt.most_common():
            print(f"   {m:38} {n:5}  ({n / len(picks) * 100:.1f}%)")


if __name__ == "__main__":
    main()
