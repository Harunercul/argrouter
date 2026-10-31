"""PubMedQA (Jin et al., EMNLP 2019), labelled subset PQA-L (1000 expert-annotated questions), from the Hugging Face
release qiaojin/PubMedQA (pqa_labeled/train-00000-of-00001.parquet, read directly).

Fields: pubid, question, context{contexts[], labels[], meshes[], ...}, long_answer, final_decision (yes/no/maybe).
Context = the abstract's sections in order, each as "<LABEL>: <text>" joined by blank lines (the conclusion
`long_answer` is the answer rationale and goes to meta only, as in the paper's reasoning-required setting).
Task mc: options ["yes", "no", "maybe"], answer = index of final_decision.
Splits: the HF release ships all 1000 as `train`; the authors' repository (github.com/pubmedqa/pubmedqa,
data/test_ground_truth.json) fixes 500 PMIDs as the official test set. Those are labelled "test", the other 500 "train".
Single subset "pqa_labeled". Native id candidates = pubid and the row index in the parquet.
"""
import json

import pandas as pd

from rowkit import check, finish, hf_file, url_file

BENCH, FAMILY = "pubmedqa", "pubmedqa"
REPO, REV = "qiaojin/PubMedQA", "9001f2853fb87cab8d220904e0de81ac6973b318"
GH_REV = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"
TEST_URL = f"https://raw.githubusercontent.com/pubmedqa/pubmedqa/{GH_REV}/data/test_ground_truth.json"
SOURCE = f"{REPO}@{REV}"
OPTIONS = ["yes", "no", "maybe"]


def main():
    df = pd.read_parquet(hf_file(REPO, "pqa_labeled/train-00000-of-00001.parquet", REV, BENCH))
    tpath, tsha = url_file(TEST_URL, BENCH, "test_ground_truth.json")
    with open(tpath) as f:
        test_ids = set(json.load(f))
    assert len(test_ids) == 500
    cands = {"pqa_labeled": []}
    for i, r in enumerate(df.itertuples(index=False)):
        ctx = r.context
        secs = [f"{lab}: {txt}" if lab else str(txt) for lab, txt in zip(ctx["labels"], ctx["contexts"])]
        split = "test" if str(r.pubid) in test_ids else "train"
        row = {
            "id": f"{BENCH}/pqa_labeled/{r.pubid}", "benchmark": BENCH, "subset": "pqa_labeled", "split": split,
            "task": "mc", "context": "\n\n".join(secs), "question": r.question, "options": OPTIONS,
            "answer": OPTIONS.index(r.final_decision), "aliases": [],
            "meta": {"pubid": int(r.pubid), "final_decision": r.final_decision, "long_answer": r.long_answer,
                     "section_labels": [str(x) for x in ctx["labels"]], "meshes": [str(x) for x in ctx["meshes"]],
                     "row_index": i},
            "source": SOURCE,
        }
        cands["pqa_labeled"].append((row, check([r.question], FAMILY, [int(r.pubid), i])))
    finish(BENCH, FAMILY, SOURCE, cands, extra={
        "split_source": f"{TEST_URL} (sha256 {tsha}): 500 official test PMIDs -> test, rest -> train",
        "native_id": "pubid, and row index in pqa_labeled parquet",
    })


if __name__ == "__main__":
    main()
