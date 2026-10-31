"""AIME loader: every AIME year available in public releases.

- di-zhang-fdu/AIME_1983_2024 (AIME_Dataset_1983_2024.csv: ID "1983-1" / "2024-II-13", Year, Problem Number, Question,
  Answer, Part). Covers 1983-2024 (2024 contains only AIME II in this release).
- MathArena/aime_2025 and MathArena/aime_2026 (parquet: problem_idx 1-30, problem, answer; the cards do not say it, but
  1-15 are AIME I and 16-30 AIME II (checked: 2025 #16 is the known AIME II P1, answer 468).
One subset ("aime"; years are metadata, not a partition of the release), split "test" (competition problems).
Native ids: the CSV's ID; for MathArena "<year>-<I|II>-<n>". Hashed text: the problem statement.
Asymptote ([asy]) diagram code is plain text and kept.
"""
import pandas as pd

from loadutil import hf_file, run

SRC = [("di-zhang-fdu/AIME_1983_2024", "3e2cc86390666c5c756622afc0eeb9e6194496bc"),
       ("MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302"),
       ("MathArena/aime_2026", "d61f486805306e6e6b5913504c07763a99e1b112")]


def main():
    cands = []
    repo, rev = SRC[0]
    df = pd.read_csv(hf_file(repo, "AIME_Dataset_1983_2024.csv", rev, "aime"), dtype=str, keep_default_na=False)
    for r in df.to_dict("records"):
        cands.append({"subset": "aime", "split": "test", "native_id": r["ID"], "hash_texts": [r["Question"]],
                      "row": {"task": "numeric", "context": "", "question": r["Question"], "options": [],
                              "answer": r["Answer"].strip(), "aliases": [],
                              "meta": {"year": int(r["Year"]), "part": r["Part"] or None,
                                       "problem_number": int(r["Problem Number"]), "release": f"{repo}@{rev}"}}})
    for repo, rev in SRC[1:]:
        year = int(repo[-4:])
        df = pd.read_parquet(hf_file(repo, "data/train-00000-of-00001.parquet", rev, "aime"))
        for r in df.to_dict("records"):
            k = int(r["problem_idx"])
            part, num = ("I", k) if k <= 15 else ("II", k - 15)
            meta = {"year": year, "part": part, "problem_number": num, "release": f"{repo}@{rev}"}
            if "problem_type" in r and r["problem_type"] is not None:
                meta["problem_type"] = list(r["problem_type"])
            cands.append({"subset": "aime", "split": "test", "native_id": f"{year}-{part}-{num}",
                          "hash_texts": [r["problem"]],
                          "row": {"task": "numeric", "context": "", "question": r["problem"], "options": [],
                                  "answer": str(r["answer"]), "aliases": [], "meta": meta}})
    run("aime", "aime", " + ".join(f"{r}@{v}" for r, v in SRC), cands,
        notes=["1983-2024 from di-zhang-fdu/AIME_1983_2024 (2024: AIME II only in that release); 2025 and 2026 "
               "from MathArena (problem_idx 1-15 = I, 16-30 = II)",
               "single subset; split 'test'; the exclusion file's aime ids are small integers, which cannot match "
               "these string ids (the question hash is the effective rule)"])


if __name__ == "__main__":
    main()
