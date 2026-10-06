"""Sunucuda (ag acik, model kodu CALISTIRMAZ): LiveCodeBench release_v2'yi indir, RouterArena'nin
385 problemini task_id ile sec, gizli testleri coz ve problem basina dosya yaz."""

import base64
import json
import pickle
import sys
import urllib.request
import zlib
from pathlib import Path

BASE = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
with open(sys.argv[1]) as fh:
    mapping = json.load(fh)
want = {v["task_id"]: (g, v["is_stdin"]) for g, v in mapping.items()}
out = Path(sys.argv[2])
out.mkdir(exist_ok=True)
found = 0
for fname in ["test.jsonl", "test2.jsonl"]:
    with urllib.request.urlopen(BASE + fname) as resp:
        for raw in resp:
            row = json.loads(raw)
            hit = want.get(row["question_id"])
            if not hit:
                continue
            gidx, is_stdin = hit
            priv = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(row["private_test_cases"])))
            )
            problem = {
                "prompt": row["question_content"],
                "test": priv,
                "entry_point": row["starter_code"],
                "task_id": row["question_id"],
                "is_stdin": is_stdin,
                "public_test_cases": row["public_test_cases"],
                "difficulty": row["difficulty"],
                "global_idx": gidx,
            }
            (out / f"{gidx}.json").write_text(json.dumps(problem))
            found += 1
    print(fname, "->", found, flush=True)
print("toplam", found, "/", len(want))
