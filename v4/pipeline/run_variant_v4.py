"""run_v4.py with prompt variant B (prompts_v4b) instead of variant A.

    python bench/v4/run_variant_v4.py --items route_data/v4/robust_items.jsonl --out-dir route_data/v4/robust_b --models ...
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_v4  # noqa: E402
from prompts_v4b import build_messages_b  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    run_v4.build_messages = build_messages_b
    run_v4.OUT = a.out_dir
    asyncio.run(run_v4.main(a.items, a.models, a.concurrency, None))
