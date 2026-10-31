"""GSM8K loader (openai/gsm8k, config "main"; parquet read directly).

Card: each row has `question` and `answer`; the answer is a step-by-step solution ending in "#### <final number>".
Subsets: one ("main"; "socratic" is the same problems with different solution formatting, so not a separate subset).
Native id: row index within the split file (the release has no ids). Hashed text: `question`.
"""
import pandas as pd

from loadutil import hf_file, run

REPO, REV = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"


def main():
    cands = []
    for split in ["train", "test"]:
        df = pd.read_parquet(hf_file(REPO, f"main/{split}-00000-of-00001.parquet", REV, "gsm8k"))
        for i, r in enumerate(df.to_dict("records")):
            final = r["answer"].split("####")[-1].strip()
            cands.append({"subset": "main", "split": split, "native_id": i, "hash_texts": [r["question"]],
                          "row": {"task": "numeric", "context": "", "question": r["question"], "options": [],
                                  "answer": final, "aliases": [], "meta": {"solution": r["answer"]}}})
    run("gsm8k", "gsm8k", f"{REPO}@{REV}", cands,
        notes=["answer = text after '####' in the source solution; full solution in meta.solution",
               "native id = row index within split (train/test); id-hash matches are therefore positional"])


if __name__ == "__main__":
    main()
