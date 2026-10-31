"""MedMCQA (Pal et al. 2022) from openlifescienceai/medmcqa (parquet files read directly).
Subsets = the 21 `subject_name` values (including the release's own "Unknown" label).
Splits: train (preferred), validation; the test split ships without answers (cop = -1) and is not usable.
Native id = the release's `id` (uuid). Task mc: options opa..opd, answer = `cop` (0-based)."""
import math

import pandas as pd

from mc_runner import finish, hf_file, row

BENCH, FAMILY = "medmcqa", "medmcqa"
REPO, REV = "openlifescienceai/medmcqa", "91c6572c454088bf71b679ad90aa8dffcd0d5868"
SOURCE = f"{REPO}@{REV}"


def s(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def main():
    cands, unlabeled = {}, 0
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(hf_file(REPO, f"data/{split}-00000-of-00001.parquet", REV, BENCH))
        for r in df.itertuples(index=False):
            cop = int(r.cop)
            if cop not in (0, 1, 2, 3):
                unlabeled += 1
                continue
            out = row(BENCH, r.subject_name, r.id, split, "mc", r.question, [s(r.opa), s(r.opb), s(r.opc), s(r.opd)], cop,
                      SOURCE, meta={"choice_type": s(r.choice_type), "topic_name": s(r.topic_name), "explanation": s(r.exp)})
            cands.setdefault(r.subject_name, []).append({"row": out, "native_id": r.id, "qtexts": [r.question]})
    finish(BENCH, FAMILY, SOURCE, cands, {"dropped_unlabeled_test": unlabeled})


if __name__ == "__main__":
    main()
