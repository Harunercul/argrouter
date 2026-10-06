"""RouterArena LiveCodeBench cevaplarini, RouterArena'nin hazirladigi problem kayitlariyla puanla.
Sadece izole konteynerde calistir (model kodunu yurutur)."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from livecodebench_util import check_correctness, has_code, post_process_code


def grade(args):
    row, problem = args
    found = has_code(row.get("response") or "")
    if not found or problem is None:
        return {"gidx": row["gidx"], "score": 0.0}
    score = check_correctness(
        problem=problem,
        completion=post_process_code(found[-1]),
        timeout=6,
        is_extracted=not problem.get("is_stdin", False),
    )
    return {"gidx": row["gidx"], "score": float(score)}


def load_problem(problem_dir, gidx):
    try:
        with open(f"{problem_dir}/{gidx}.json") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def main(answers, problem_dir, out):
    with open(answers) as fh:
        rows = [json.loads(x) for x in fh if x.strip()]

    def job(r):
        return grade((r, load_problem(problem_dir, r["gidx"])))

    with ThreadPoolExecutor(max_workers=2) as ex, open(out, "w") as fo:
        for i, g in enumerate(ex.map(job, rows)):
            fo.write(json.dumps(g) + "\n")
            if (i + 1) % 25 == 0:
                print(i + 1, "/", len(rows), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
