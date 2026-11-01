"""Retry the failed calls of a v5 label run with the unchanged part-2 runner (run_v4.py).

run_v4.done_keys chokes on the trailing newline of a finished file, so resuming fails before any call; this wrapper
only swaps in a done_keys that skips blank lines. Prompts, request body and output format are run_v4's.

    python bench/v5/retry_v5.py --items route_data/v4src/all.jsonl --out-dir route_data/v5/matrix --models qwen/qwen3.8-flash
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v4"))
import run_v4  # noqa: E402


def done_keys(model: str) -> set[str]:
    p = run_v4.out_path(model)
    if not p.exists():
        return set()
    rows = (json.loads(x) for x in p.read_text().split("\n") if x.strip())
    return {r["gidx"] for r in rows if not r.get("error")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    run_v4.OUT = a.out_dir
    run_v4.done_keys = done_keys
    asyncio.run(run_v4.main(a.items, a.models, a.concurrency, None))
