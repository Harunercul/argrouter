"""Full v4 matrix analysis on OUR calibration set only (route_data/v4src/all.jsonl, equal benchmark weights,
own grader). Per-model accuracy/cost, per-benchmark accuracy, and draft-and-escalate cascades
(two cheap drafts agree -> accept, else escalate). Nothing here reads RouterArena data.

    python bench/v4/analyze_full_v4.py [--lcb route_data/v4/lcb_graded.jsonl] [--prices list.json]

--prices: {"model": [in_per_M, out_per_M]} recomputes cost from tokens (reasoning billed as output);
without it the billed cost_usd is used.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from grade_v4 import extract_letter, final_text, grade  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "route_data" / "v4src" / "all.jsonl"
MATRIX = ROOT / "route_data" / "v4" / "full_matrix"


def arena(acc: float, cost_per_1k: float) -> float:
    """Same shape as the public leaderboard formula (beta 0.1, log cost between 0.0044 and 200)."""
    c = (math.log2(200) - math.log2(max(cost_per_1k, 0.0044))) / (math.log2(200) - math.log2(0.0044))
    return 1.1 * acc * c / (0.1 * acc + c) * 100


def norm_answer(it: dict, resp: str) -> str:
    """What two drafts must agree on: the chosen option for mc/binary, else the normalised final text."""
    if it["task"] in ("mc", "binary"):
        k = extract_letter(resp, len(it["options"]), it["options"])
        return "" if k is None else f"opt{k}"
    return " ".join(final_text(resp).lower().split()).rstrip(".")


def load(lcb_path: Path | None, prices: dict | None):
    items = [json.loads(x) for x in ITEMS.read_text().split("\n") if x.strip()]
    idx = {it["id"]: i for i, it in enumerate(items)}
    lcb = collections.defaultdict(dict)
    if lcb_path and lcb_path.exists():
        for x in lcb_path.read_text().split("\n"):
            if x.strip():
                r = json.loads(x)
                lcb[r["model"]][r.get("id") or r["gidx"]] = float(r["score"])
    models, S, C, A = [], [], [], []
    for f in sorted(MATRIX.glob("*.jsonl")):
        s = np.full(len(items), np.nan)
        c = np.full(len(items), np.nan)
        a = [""] * len(items)
        name = None
        for x in f.read_text().split("\n"):
            if not x.strip():
                continue
            r = json.loads(x)
            if r.get("error") or r["gidx"] not in idx:
                continue
            name = r["model"]
            i = idx[r["gidx"]]
            it = items[i]
            resp = r.get("response") or ""
            g = lcb.get(name, {}).get(it["id"]) if it["task"] == "code" else grade(it, resp)
            if g is not None:
                s[i] = g
            if prices:
                pin, pout = prices[name.split("@")[0]]
                out = (r.get("output_tokens") or 0)
                c[i] = ((r.get("input_tokens") or 0) * pin + out * pout) / 1e6
            else:
                c[i] = r.get("cost_usd") or 0.0
            a[i] = norm_answer(it, resp) if it["task"] != "code" else ""
        if name:
            models.append(name)
            S.append(s)
            C.append(c)
            A.append(a)
    return items, models, np.array(S), np.array(C), A


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcb", type=Path)
    ap.add_argument("--prices", type=Path)
    a = ap.parse_args()
    prices = json.loads(a.prices.read_text()) if a.prices else None
    items, models, S, C, A = load(a.lcb, prices)
    benches = [it["benchmark"] for it in items]
    keep = ~np.isnan(S).any(0)
    print(f"{len(items)} items, {int(keep.sum())} graded by all {len(models)} models "
          f"(cost: {'list prices from ' + str(a.prices) if prices else 'billed'})")
    ki = np.where(keep)[0]
    S, C = S[:, keep], np.nan_to_num(C[:, keep])
    A = [[ans[i] for i in ki] for ans in A]
    items = [items[i] for i in ki]
    benches = [benches[i] for i in ki]
    bc = collections.Counter(benches)
    w = np.array([1.0 / bc[b] for b in benches])
    w = w / w.sum()
    short = [m.split("/")[-1] for m in models]

    print("\nmodel                                   acc%   $/1K   arena")
    for j in sorted(range(len(models)), key=lambda j: -(w @ S[j])):
        acc, cost = w @ S[j], (w @ C[j]) * 1000
        print(f"{models[j]:38} {acc * 100:5.1f} {cost:6.3f} {arena(acc, cost):6.2f}")

    print("\nper benchmark accuracy %")
    print(" " * 17 + " ".join(f"{m[:9]:>9}" for m in short))
    for b in sorted(bc):
        r = [i for i, x in enumerate(benches) if x == b]
        print(f"{b:16} " + " ".join(f"{S[j, r].mean() * 100:9.0f}" for j in range(len(models))))

    n = S.shape[1]
    ar = np.arange(n)
    print("\ndraft-and-escalate: drafts d1,d2 (both always paid); agree -> d1's answer, else escalate e")
    res = []
    for d1, d2 in itertools.permutations(range(len(models)), 2):
        agree = np.array([A[d1][i] != "" and A[d1][i] == A[d2][i] for i in range(n)])
        for e in range(len(models)):
            if e in (d1, d2):
                continue
            pick = np.where(agree, d1, e)
            acc = w @ S[pick, ar]
            cost = (w @ (C[d1] + C[d2] + np.where(agree, 0, C[e]))) * 1000
            res.append((arena(acc, cost), short[d1], short[d2], short[e], acc, cost, w @ agree))
    res.sort(reverse=True)
    for sc, d1, d2, e, acc, cost, ag in res[:12]:
        print(f"  {sc:6.2f}  {d1:24} + {d2:24} -> {e:24} acc {acc * 100:5.1f}% ${cost:.3f}/1K agree {ag * 100:.0f}%")
