"""MMLU (Hendrycks et al. 2021) from cais/mmlu (per-subject parquet folders, read directly).
Subsets = the 57 subjects (the `all` and `auxiliary_train` folders are not subjects and are skipped).
Splits: dev, validation (preferred, both rank as "validation/dev"), then test.
MMLU ships no ids: native id = 0-based row index inside the subject's split file (used for the id-hash rule);
the output id is "<split>:<index>" so it stays unique. Task mc: options = `choices`, answer = `answer`."""
import pandas as pd
from huggingface_hub import HfApi

from mc_runner import finish, hf_file, row

BENCH, FAMILY = "mmlu", "mmlu"
REPO, REV = "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe"
SOURCE = f"{REPO}@{REV}"


def main():
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
    subjects = sorted({f.split("/")[0] for f in files if f.endswith(".parquet")} - {"all", "auxiliary_train"})
    assert len(subjects) == 57, len(subjects)
    cands = {}
    for sub in subjects:
        for split in ["dev", "validation", "test"]:
            df = pd.read_parquet(hf_file(REPO, f"{sub}/{split}-00000-of-00001.parquet", REV, BENCH))
            for i, r in enumerate(df.itertuples(index=False)):
                assert r.subject == sub
                out = row(BENCH, sub, f"{split}:{i}", split, "mc", r.question, [str(c) for c in r.choices], int(r.answer),
                          SOURCE, meta={"row_index": i})
                cands.setdefault(sub, []).append({"row": out, "native_id": i, "qtexts": [r.question]})
    finish(BENCH, FAMILY, SOURCE, cands)


if __name__ == "__main__":
    main()
