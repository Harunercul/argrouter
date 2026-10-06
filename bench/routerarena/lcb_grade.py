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
from pathlib import Path

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


def split_tests(tests_path, split_dir):
    """Test dosyasini soru basina dosyalara bol (bellege hepsini yuklememek icin)."""
    split = Path(split_dir)
    if split.exists() and any(split.iterdir()):
        return split
    split.mkdir(exist_ok=True)
    with open(tests_path) as fh:
        for line in fh:
            if line.strip():
                t = json.loads(line)
                (split / f"{t['gidx']}.json").write_text(line)
    return split


def load_problem(split, gidx):
    path = split / f"{gidx}.json"
    if not path.exists():
        return None
    t = json.loads(path.read_text())
    priv = decode_private(t["private_test_cases"])
    if priv is None:
        return None
    return {
        "test": priv,
        "is_stdin": is_stdin(t["public_test_cases"]),
        "public_test_cases": t["public_test_cases"],
    }


def main(resp_path, tests_path, out_path):
    split = split_tests(tests_path, "tests_split")
    done = set()
    if Path(out_path).exists():
        with open(out_path) as fh:
            done = {(json.loads(x)["model"], json.loads(x)["gidx"]) for x in fh if x.strip()}
    with open(resp_path) as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    rows = [r for r in rows if (r["model"], r["gidx"]) not in done]
    print(len(done), "zaten puanli,", len(rows), "kaldi", flush=True)

    def job(r):
        return grade((r, load_problem(split, r["gidx"])))

    with ThreadPoolExecutor(max_workers=3) as ex, open(out_path, "a") as out:
        for i, g in enumerate(ex.map(job, rows)):
            out.write(json.dumps(g, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 50 == 0:
                print(i + 1, "/", len(rows), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
