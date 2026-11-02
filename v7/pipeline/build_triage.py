"""Triage set for new v7 candidates: the hardest items of our own v4 calibration set plus QANTA first-sentence items.

Hard = highest error rate among the 8 fixed v4 reference models (labels from part 3); 12 per benchmark, ties broken
by a seeded shuffle; LiveCodeBench left out (needs the sandbox grader). QANTA first-sentence items use the release's
own `first_sentence` field (the opening clue, as in QANTA's start-of-question evaluation), same answer and aliases.
No RouterArena data is used.

    python bench/v7/build_triage.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "v5"))
sys.path.insert(0, str(ROOT / "route_data" / "v4src" / "loaders"))
import fit_router_v5 as f  # noqa: E402
from qanta import FOLDS, REPO, REV  # noqa: E402
from rowkit import hf_file  # noqa: E402

PER_BENCH, N_S1, SEED = 12, 60, 0
OUT = ROOT / "route_data" / "v7" / "triage_items.jsonl"

if __name__ == "__main__":
    items, models, S, C, ref, skip = f.load_v5(json.loads((ROOT / "route_data/v5/prices_ra.json").read_text()))
    err = 1 - np.nan_to_num(S[ref]).mean(0)
    rng = random.Random(SEED)
    out = []
    for b in sorted({it["benchmark"] for it in items} - {"livecodebench"}):
        idx = [i for i, it in enumerate(items) if it["benchmark"] == b]
        rng.shuffle(idx)
        idx.sort(key=lambda i: -err[i])
        out += [dict(items[i], meta=dict(items[i]["meta"], triage="hard", ref_err=round(float(err[i]), 3)))
                for i in idx[:PER_BENCH]]
    first = {}
    for fold in FOLDS:
        df = pd.read_parquet(hf_file(REPO, f"mode=full,char_skip=25/{fold}-00000-of-00001.parquet", REV, "qanta"),
                             columns=["qanta_id", "first_sentence"])
        first.update(zip(df.qanta_id.astype(int), df.first_sentence))
    qs = [it for it in items if it["benchmark"] == "qanta"]
    rng.shuffle(qs)
    for it in qs[:N_S1]:
        out.append(dict(it, id=it["id"] + "/s1", question=first[it["meta"]["qanta_id"]],
                        meta=dict(it["meta"], triage="qanta_first_sentence")))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"wrote {len(out)} items to {OUT}")
