"""MATH (Hendrycks et al. 2021) loader for benchmark "math" (file not named math.py: that would shadow the stdlib module).

The authors' own archives (people.eecs.berkeley.edu/~hendrycks/MATH.tar, hf hendrycks/competition_math) are no longer
downloadable (403 / DMCA-disabled), so this reads EleutherAI/hendrycks_math, a parquet mirror of the original release:
one config per subject, splits train/test, fields problem, level ("Level 1".."Level 5", one "Level ?"), type, solution.
Subsets: the 7 subjects. All levels kept (meta.level). Answer: content of the last \\boxed{...} (or \\fbox{...}) in the
solution (balanced braces), task "short". Items without a boxed answer are dropped as malformed.
Native id = row index within the subject/split file (the mirror carries no ids). Hashed text: problem.
"""
import pandas as pd

from loadutil import hf_file, run

REPO, REV = "EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus"]


def last_boxed(s):
    best = None
    for tag in ("\\boxed", "\\fbox"):
        i = s.rfind(tag)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, tag)
    if best is None:
        return None
    i = best[0] + len(best[1])
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j].strip()
            j += 1
        return None
    # "\boxed 5" form
    rest = s[i:].split("$")[0].strip()
    return rest or None


def main():
    cands, dropped = [], {}
    for subj in SUBJECTS:
        for split in ["train", "test"]:
            df = pd.read_parquet(hf_file(REPO, f"{subj}/{split}-00000-of-00001.parquet", REV, "math"))
            for i, r in enumerate(df.to_dict("records")):
                ans = last_boxed(r["solution"])
                if not ans:
                    dropped.setdefault(subj, {}).setdefault("no_boxed_answer", 0)
                    dropped[subj]["no_boxed_answer"] += 1
                    continue
                cands.append({"subset": subj, "split": split, "native_id": i, "hash_texts": [r["problem"]],
                              "row": {"task": "short", "context": "", "question": r["problem"], "options": [],
                                      "answer": ans, "aliases": [],
                                      "meta": {"level": r["level"], "type": r["type"], "solution": r["solution"]}}})
    run("math", "math", f"{REPO}@{REV}", cands, dropped=dropped,
        notes=["authors' MATH.tar / hendrycks/competition_math unavailable (403 / DMCA); EleutherAI parquet mirror used",
               "answer = last \\boxed{} in the solution; full solution in meta.solution",
               "native id = row index within <subject>/<split> parquet (no ids in the mirror), so id-hash matches are "
               "positional and unrelated to the exclusion file's MATH ids"])


if __name__ == "__main__":
    main()
