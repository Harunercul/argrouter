"""Benchmark shares of the RouterArena query set (counts of the public `global index` prefixes in
dataset/router_data.json; no answers, no results). RA's accuracy is the plain mean over its 8,400 queries, so
these shares, not equal benchmark shares, are what an item of each benchmark is worth there.

    python bench/v7/ra_mix.py ~/Desktop/argrouter-ra/RouterArena/dataset/router_data.json
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = {"arcmmlu": "arcmmlu", "mmlupro": "mmlupro", "livecodebench": "livecodebench", "narrativeqa": "narrativeqa",
       "geobench": "geobench", "geographydata": "geobench", "pubmedqa": "pubmedqa", "medmcqa": "medmcqa",
       "musictheorybench": "musictheorybench", "qanta": "qanta", "mathqa": "mathqa", "chessinstruct": "chessinstruct",
       "opentdb": "opentdb", "mmlu": "mmlu", "superglue": "superglue", "ethics": "ethics", "wmt19": "wmt19",
       "asdiv": "asdiv", "finqa": "finqa", "socialiqa": "socialiqa", "math": "math", "aime": "aime", "gsm8k": "gsm8k"}

# RouterArena subset name -> our v4src "subset" values (None = the whole benchmark); used by the "rasub" weights
SUPERGLUE = {"wic": ["WiC"], "wsc": ["WSC"], "causalreasoning": ["COPA"], "entailment": ["RTE", "CB", "AX-b", "AX-g"],
             "qa": ["BoolQ"], "rc": ["MultiRC"], "clozetest": ["ReCoRD"]}


def subset(bench: str, gidx: str) -> list[str] | None:
    rest = re.sub(r"_\d+$", "", gidx)
    if bench == "mmlupro":
        return [rest.split("_", 1)[1]]
    if bench in ("opentdb", "qanta", "mmlu", "ethics"):
        return [rest.split("_", 1)[1]]
    if bench == "superglue":
        return SUPERGLUE[rest.split("-", 1)[1].lower()]
    if bench == "wmt19":
        return [rest.split("-", 1)[1]]
    return None


if __name__ == "__main__":
    rows = json.loads(Path(sys.argv[1]).read_text())
    c, sub = collections.Counter(), collections.defaultdict(collections.Counter)
    for r in rows:
        src = re.split(r"[_\-]", r["global index"])[0].lower()
        c[MAP[src]] += 1
        ss = subset(MAP[src], r["global index"])
        if ss:
            sub[MAP[src]]["|".join(ss)] += 1
    share = {k: v / len(rows) for k, v in sorted(c.items(), key=lambda x: -x[1])}
    out = ROOT / "route_data" / "v7" / "ra_mix.json"
    out.write_text(json.dumps({"n": len(rows), "count": dict(c), "share": share,
                                "subset_count": {b: dict(v) for b, v in sub.items()}}, indent=1))
    for k, v in share.items():
        print(f"{k:18} {c[k]:5} {v * 100:5.1f}%")
