"""Pilot set for reasoning-effort variants: 18 random items per benchmark (seed 7) from the v4 calibration set,
LiveCodeBench left out (needs the sandbox). The base variants of these models already have labels on the same
items, so a variant is compared with its base on identical items before paying for a full matrix.

    python bench/v7/build_variant_pilot.py   -> route_data/v7/variant_pilot.jsonl
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fit_router_v7 as f  # noqa: E402

if __name__ == "__main__":
    by = collections.defaultdict(list)
    for it in f.jsonl(f.ITEMS):
        if it["benchmark"] != "livecodebench":
            by[it["benchmark"]].append(it)
    rng = random.Random(7)
    out = [it for b in sorted(by) for it in rng.sample(by[b], min(18, len(by[b])))]
    p = f.ROOT / "route_data" / "v7" / "variant_pilot.jsonl"
    p.write_text("".join(json.dumps(it) + "\n" for it in out))
    print(len(out), "items ->", p)
