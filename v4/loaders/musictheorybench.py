"""MusicTheoryBench (ChatMusician, Yuan et al. 2024, arXiv 2402.16153) from m-a-p/MusicTheoryBench (parquet).
Subsets = `subject` (knowledge, reasoning). Splits: dev (preferred) then test.
Native id = the release's `id`. Task mc: options A-D, answer = index of `answer`.
Music excerpts ship as ABC notation text (`abc_score`): kept in full as `context`, so every item is text-only."""
import pandas as pd

from mc_runner import finish, hf_file, row

BENCH, FAMILY = "musictheorybench", "musictheorybench"
REPO, REV = "m-a-p/MusicTheoryBench", "81b65609885be1fba19ce167a2a9604ae676c5ae"
SOURCE = f"{REPO}@{REV}"
FILES = {"dev": "data/dev-00000-of-00001-33c852d919b7c9a2.parquet",
         "test": "data/test-00000-of-00001-0b1d2a25639002c2.parquet"}


def main():
    cands = {}
    for split, path in FILES.items():
        df = pd.read_parquet(hf_file(REPO, path, REV, BENCH)).fillna("")
        for r in df.to_dict("records"):
            assert r["split"] == split
            opts = [r["options"][k] for k in "ABCD"]
            ans = r["answer"].strip()
            assert ans in ("A", "B", "C", "D"), r["id"]
            abc = r["abc_score"] or ""
            out = row(BENCH, r["subject"], r["id"], split, "mc", r["stem"], opts, "ABCD".index(ans), SOURCE, context=abc,
                      meta={"instruction": r["instruction"], "has_abc_score": bool(abc), "analysis": r["analysis"]})
            qt = [r["stem"]] + ([r["stem"] + "\n" + abc] if abc else [])
            cands.setdefault(r["subject"], []).append({"row": out, "native_id": int(r["id"]), "qtexts": qt})
    finish(BENCH, FAMILY, SOURCE, cands)


if __name__ == "__main__":
    main()
