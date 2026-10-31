"""MMLU-Pro (Wang et al. 2024) from TIGER-Lab/MMLU-Pro, parquet files read directly.
Subsets = the 14 `category` values. Splits: validation (5 per category, preferred) then test.
Native id = `question_id`. Task mc: options = the row's `options` list, answer = `answer_index`."""
import pandas as pd

from mc_runner import finish, hf_file, row

BENCH, FAMILY = "mmlupro", "mmlupro"
REPO, REV = "TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be"
SOURCE = f"{REPO}@{REV}"


def main():
    cands = {}
    for split in ["validation", "test"]:
        df = pd.read_parquet(hf_file(REPO, f"data/{split}-00000-of-00001.parquet", REV, BENCH))
        for r in df.itertuples(index=False):
            opts = [str(o) for o in r.options]
            ai = int(r.answer_index)
            assert "ABCDEFGHIJ"[ai] == r.answer, r.question_id
            out = row(BENCH, r.category, r.question_id, split, "mc", r.question, opts, ai, SOURCE,
                      meta={"question_id": int(r.question_id), "src": r.src, "answer_letter": r.answer})
            cands.setdefault(r.category, []).append({"row": out, "native_id": int(r.question_id), "qtexts": [r.question]})
    finish(BENCH, FAMILY, SOURCE, cands)


if __name__ == "__main__":
    main()
