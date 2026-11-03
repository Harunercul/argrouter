"""Compare effort variants (route_data/v7/variant_pilot) with every labelled model on the same pilot items; RA-share
and equal-share benchmark weights (LCB is not in the pilot). Grader = grade_v4.

    python bench/v7/analyze_variant_pilot.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fit_router_v7 as f  # noqa: E402

if __name__ == "__main__":
    prices = json.loads((f.ROOT / "route_data/v7/prices_ra.json").read_text())
    pilot = f.jsonl(f.ROOT / "route_data/v7/variant_pilot.jsonl")
    ids = {it["id"]: it for it in pilot}
    share = json.loads((f.ROOT / "route_data/v7/ra_mix.json").read_text())["share"]
    bc = collections.Counter(it["benchmark"] for it in pilot)
    tot = sum(share[b] for b in bc)
    wr = {i: share[it["benchmark"]] / tot / bc[it["benchmark"]] for i, it in ids.items()}
    we = {i: 1 / len(bc) / bc[it["benchmark"]] for i, it in ids.items()}
    folders = f.FOLDERS + [f.ROOT / "route_data/v7/variant_pilot", f.ROOT / "route_data/v7/triage"]
    res = collections.defaultdict(dict)
    for folder in folders:
        for p in sorted(folder.glob("*.jsonl")):
            for r in f.jsonl(p):
                if r.get("error") or r["gidx"] not in ids:
                    continue
                pr = prices.get(r["model"].split("@")[0])
                cost = ((r.get("input_tokens") or 0) * pr[0] + (r.get("output_tokens") or 0) * pr[1]) / 1e6 if pr \
                    else r.get("cost_usd") or 0
                res[r["model"]][r["gidx"]] = (f.grade(ids[r["gidx"]], r.get("response") or ""), cost)
    rows = []
    for m, d in res.items():
        if len(d) < 0.95 * len(ids):
            continue
        ks = list(d)
        for name, W in (("ra", wr), ("eq", we)):
            ws = np.array([W[k] for k in ks]); ws /= ws.sum()
            acc = ws @ np.array([d[k][0] for k in ks]); cost = ws @ np.array([d[k][1] for k in ks]) * 1000
            rows.append((m, name, acc, cost, f.arena(acc, cost), len(ks)))
    for name in ("ra", "eq"):
        print(f"\n== {name} weights")
        for m, _, acc, cost, ar, n in sorted([r for r in rows if r[1] == name], key=lambda r: -r[4]):
            print(f"  {m:40} n={n:3} acc {acc * 100:5.1f}%  ${cost:6.3f}/1K  arena {ar:.2f}")
