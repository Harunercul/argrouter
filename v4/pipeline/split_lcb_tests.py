"""Move the inline LiveCodeBench tests out of the v4src data file.

  data/livecodebench.jsonl (1.6 GB, tests in meta)  ->
      data/livecodebench.jsonl           small rows: meta.public_tests/private_tests removed,
                                         meta.n_public_tests / n_private_tests / tests_file added
      data/livecodebench_tests.jsonl.gz  {"id", "gidx", "fn_name", "public_tests", "private_tests"} per row

Streams line by line, verifies the round trip (same ids, same test counts, gz re-read) before replacing the
original, and keeps the original as livecodebench.jsonl.full unless --no-backup. Standard library only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

DEFAULT = Path.home() / "Desktop/thriftllm-lab/route_data/v4src/data/livecodebench.jsonl"


def split(src: Path, backup: bool = True) -> dict:
    tests_path = src.with_name("livecodebench_tests.jsonl.gz")
    small_tmp, tests_tmp = src.with_suffix(".jsonl.small.tmp"), tests_path.with_suffix(".gz.tmp")
    counts = {}
    with open(src, encoding="utf-8") as fi, open(small_tmp, "w", encoding="utf-8") as fs, \
            gzip.open(tests_tmp, "wt", encoding="utf-8", compresslevel=6) as ft:
        for line in fi:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("meta") or {}
            if "public_tests" not in meta and "private_tests" not in meta:
                raise SystemExit(f"{row['id']}: no inline tests (already split?)")
            pub, priv = meta.pop("public_tests", []) or [], meta.pop("private_tests", []) or []
            ft.write(json.dumps({"id": row["id"], "gidx": row["id"], "fn_name": meta.get("fn_name"),
                                 "public_tests": pub, "private_tests": priv}, ensure_ascii=False) + "\n")
            meta.update(n_public_tests=len(pub), n_private_tests=len(priv), tests_file=tests_path.name)
            fs.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[row["id"]] = (len(pub), len(priv))
    # verify before touching the original
    with gzip.open(tests_tmp, "rt", encoding="utf-8") as f:
        back = {r["id"]: (len(r["public_tests"]), len(r["private_tests"])) for r in map(json.loads, f)}
    with open(small_tmp, encoding="utf-8") as f:
        small_ids = [json.loads(l)["id"] for l in f if l.strip()]
    assert back == counts and small_ids == list(counts), "round-trip check failed; original left untouched"
    if backup:
        os.replace(src, src.with_suffix(".jsonl.full"))
    os.replace(tests_tmp, tests_path)
    os.replace(small_tmp, src)
    return {"rows": len(counts), "tests": sum(a + b for a, b in counts.values()),
            "small_bytes": src.stat().st_size, "tests_gz_bytes": tests_path.stat().st_size}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT)
    ap.add_argument("--no-backup", action="store_true", help="do not keep livecodebench.jsonl.full")
    a = ap.parse_args()
    print(json.dumps(split(a.src, backup=not a.no_backup)))
