"""MathQA loader (authors' release https://math-qa.github.io/math-QA/data/MathQA.zip; train/dev/test.json).

Fields: Problem, Rationale, options ("a ) 38 , b ) 27.675 , ..." or a python-list-literal string), correct (a-e),
annotated_formula, linear_formula, category (gain/general/geometry/other/physics/probability).
Subsets: the 6 categories. challenge_test.json (an extra adversarial test file) is not a standard split and is skipped.
Options are parsed into 5 strings (label prefixes removed); items whose options cannot be parsed into a-e are dropped as
malformed. Native id = row index within the split file. Hashed text: Problem.
"""
import ast
import json
import re
import zipfile

from loadutil import run, url_file

URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"
LABEL = re.compile(r"^(?:[a-e]\s*\)\s*)+")


def parse_options(o):
    o = o.strip()
    if o.startswith("["):
        try:
            parts = ast.literal_eval(o)  # data literal only, never executed
        except (ValueError, SyntaxError):
            return None
    else:
        parts = [p for p in re.split(r"\s*,\s*(?=[a-e] \) )", o) if p.strip()]
    if len(parts) != 5:
        return None
    labels = [p.strip()[:1] for p in parts]
    if labels != list("abcde"):
        return None
    return [LABEL.sub("", p.strip()).strip() for p in parts]


def main():
    path, sha = url_file(URL, "mathqa")
    z = zipfile.ZipFile(path)
    cands, dropped = [], {}
    for fname, split in [("train.json", "train"), ("dev.json", "validation"), ("test.json", "test")]:
        for i, r in enumerate(json.loads(z.read(fname))):
            opts = parse_options(r["options"])
            cat = r["category"]
            if opts is None or r["correct"] not in "abcde":
                dropped.setdefault(cat, {}).setdefault("unparseable_options", 0)
                dropped[cat]["unparseable_options"] += 1
                continue
            cands.append({"subset": cat, "split": split, "native_id": i, "hash_texts": [r["Problem"]],
                          "row": {"task": "mc", "context": "", "question": r["Problem"], "options": opts,
                                  "answer": "abcde".index(r["correct"]), "aliases": [],
                                  "meta": {"rationale": r["Rationale"], "annotated_formula": r["annotated_formula"],
                                           "linear_formula": r["linear_formula"], "raw_options": r["options"]}}})
    run("mathqa", "mathqa", f"{URL}@sha256:{sha}", cands, dropped=dropped,
        notes=["subsets = the 6 MathQA categories; dev.json mapped to split 'validation'; challenge_test.json skipped",
               "native id = row index within the split file, so id-hash matches are positional"])


if __name__ == "__main__":
    main()
