"""QANTA start-of-question items for v7: every QANTA item of the v4 calibration set, re-asked with only its first
sentence (the release's `first_sentence` field, QANTA's start-of-question setting). Same answer, aliases and grader.
Ids get the suffix "/s1". No RouterArena data is used.

    python bench/v7/build_qanta_s1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "route_data" / "v4src" / "loaders"))
from qanta import FOLDS, REPO, REV  # noqa: E402
from rowkit import hf_file  # noqa: E402

SRC = ROOT / "route_data" / "v4src" / "all.jsonl"
OUT = ROOT / "route_data" / "v7" / "qanta_s1.jsonl"

if __name__ == "__main__":
    first = {}
    for fold in FOLDS:
        df = pd.read_parquet(hf_file(REPO, f"mode=full,char_skip=25/{fold}-00000-of-00001.parquet", REV, "qanta"),
                             columns=["qanta_id", "first_sentence"])
        first.update(zip(df.qanta_id.astype(int), df.first_sentence))
    items = [json.loads(x) for x in SRC.read_text().split("\n") if x.strip()]
    out = [dict(it, id=it["id"] + "/s1", question=first[it["meta"]["qanta_id"]],
                meta=dict(it["meta"], variant="first_sentence"))
           for it in items if it["benchmark"] == "qanta"]
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"wrote {len(out)} items to {OUT}")
