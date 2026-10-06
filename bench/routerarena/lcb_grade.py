"""LiveCodeBench cevaplarini RouterArena'nin livecodebench_util'iyle puanla.

Model kodunu CALISTIRIR: sadece izole konteynerde kullan (ag kapali, bellek/CPU sinirli):
  docker run --rm --network none --memory 4g --cpus 2 -v $PWD:/w -w /w python:3.11-slim \
      python lcb_grade.py responses.jsonl lcb_tests.jsonl graded_lcb.jsonl
Ayni klasorde RouterArena'nin llm_evaluation/livecodebench_util.py dosyasi olmali.
"""

import base64
import json
import pickle
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

from livecodebench_util import check_correctness, has_code, post_process_code


def decode_private(enc):
    try:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(enc))))
    except Exception:
        return None


def is_stdin(public):
    try:
        return any(t.get("testtype") == "stdin" for t in json.loads(public))
    except Exception:
        return False


def grade(args):
    row, problem = args
    found = has_code(row.get("response") or "")
    if not found or problem is None or not problem["test"]:
        return {**row, "score": 0.0}
    code = post_process_code(found[-1])
    score = check_correctness(
        problem=problem, completion=code, timeout=6, is_extracted=not problem["is_stdin"]
    )
    return {**row, "score": float(score)}


def main(resp_path, tests_path, out_path):
    problems = {}
    with open(tests_path) as fh:
        tests = [json.loads(x) for x in fh if x.strip()]
    for t in tests:
        priv = decode_private(t["private_test_cases"])
        problems[t["gidx"]] = (
            None
            if priv is None
            else {
                "test": priv,
                "is_stdin": is_stdin(t["public_test_cases"]),
                "public_test_cases": t["public_test_cases"],
            }
        )
    with open(resp_path) as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    jobs = [(r, problems.get(r["gidx"])) for r in rows]
    with ThreadPoolExecutor(max_workers=4) as ex, open(out_path, "w") as out:
        for i, g in enumerate(ex.map(grade, jobs)):
            out.write(json.dumps(g, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(i + 1, "/", len(jobs), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
