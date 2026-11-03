"""Write the LiveCodeBench responses of the given models (from route_data/v7/matrix) to a file for the sandboxed
grader (/private/tmp/lcbgrade, lcb_grade_v4.py): one row {gidx, model, response} per call.

    python bench/v7/export_lcb.py OUT.jsonl model [model ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fit_router_v7 as f  # noqa: E402

if __name__ == "__main__":
    out, models = Path(sys.argv[1]), sys.argv[2:]
    rows = []
    for m in models:
        for r in f.jsonl(f.ROOT / "route_data/v7/matrix" / (m.replace("/", "__") + ".jsonl")):
            if not r.get("error") and r["gidx"].startswith("livecodebench/"):
                rows.append({"gidx": r["gidx"], "model": r["model"], "response": r.get("response") or ""})
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "rows ->", out)
