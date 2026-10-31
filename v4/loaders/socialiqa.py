"""Social IQa loader (AI2 release https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip).

Files: train.jsonl / dev.jsonl (context, question, answerA, answerB, answerC) with labels in train-labels.lst /
dev-labels.lst (1-based: 1=A, 2=B, 3=C). The test labels are not public, so only train and dev (-> "validation").
One subset. Context = context, question = question, options = [answerA, answerB, answerC], answer = label-1 (mc).
Native id = row index within the split file. Hashed text: question (the native question field).
Note: Social IQa questions are mostly templates ("How would Others feel as a result?"), and many such template
strings are on the exclusion list, so the question rule removes every item that shares one of them (~10%).
"""
import json
import zipfile

from loadutil import run, url_file

URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"


def main():
    path, sha = url_file(URL, "socialiqa")
    z = zipfile.ZipFile(path)
    cands = []
    for fname, split in [("train", "train"), ("dev", "validation")]:
        rows = [json.loads(l) for l in z.read(f"socialiqa-train-dev/{fname}.jsonl").decode("utf-8").splitlines()
                if l.strip()]
        labels = [l.strip() for l in z.read(f"socialiqa-train-dev/{fname}-labels.lst").decode().splitlines()
                  if l.strip()]
        assert len(rows) == len(labels), fname
        for i, (r, lab) in enumerate(zip(rows, labels)):
            cands.append({"subset": "socialiqa", "split": split, "native_id": i, "hash_texts": [r["question"]],
                          "row": {"task": "mc", "context": r["context"], "question": r["question"],
                                  "options": [r["answerA"], r["answerB"], r["answerC"]], "answer": int(lab) - 1,
                                  "aliases": [], "meta": {}}})
    run("socialiqa", "socialiqa", f"{URL}@sha256:{sha}", cands,
        notes=["labels are 1-based in the .lst files (1=A); stored 0-based",
               "question rule applied to the native question field, which is a template for most items: every item "
               "sharing a listed template is dropped (over-exclusion, see excluded_question_hash)",
               "native id = row index within split; id-hash matches are positional"])


if __name__ == "__main__":
    main()
