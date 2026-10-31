"""ASDiv loader (authors' repository chaochun/nlu-asdiv-dataset, dataset/ASDiv.xml at a pinned commit).

XML: <Problem ID="nluds-0001" Grade="1" Source=...><Body/><Question/><Solution-Type/><Answer/><Formula/></Problem>.
The release is a single corpus (5-fold CV files are only an index), so there is one subset ("all") and split "all".
Answer "9 (apples)" -> canonical "9" (task numeric) with the raw string as alias; non-numeric answers (names, times,
yes/no, multi-part "a;b") keep the raw string (task short). Context = Body, question = Question.
Native id = Problem ID. Hashed texts: Body + " " + Question (the full problem), Question, Body.
"""
import re
import xml.etree.ElementTree as ET

from loadutil import run, url_file

COMMIT = "883f90a9a65bf00304ba8f37423910fe743abc47"
URL = f"https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/{COMMIT}/dataset/ASDiv.xml"
NUM = re.compile(r"^\s*(-?[\d,]*\.?\d+)\s*(\([^()]*\))?\s*$")


def main():
    path, sha = url_file(URL, "asdiv")
    root = ET.parse(path).getroot()
    cands = []
    for p in root.iter("Problem"):
        body = (p.findtext("Body") or "").strip()
        q = (p.findtext("Question") or "").strip()
        raw = (p.findtext("Answer") or "").strip()
        m = NUM.match(raw)
        if m:
            task, ans, aliases = "numeric", m.group(1), ([raw] if raw != m.group(1) else [])
        else:
            task, ans, aliases = "short", raw, []
        meta = {"grade": p.get("Grade"), "solution_type": p.findtext("Solution-Type"),
                "formula": p.findtext("Formula"), "raw_answer": raw, "origin": p.get("Source")}
        cands.append({"subset": "all", "split": "all", "native_id": p.get("ID"),
                      "hash_texts": [f"{body} {q}", q, body],
                      "row": {"task": task, "context": body, "question": q, "options": [], "answer": ans,
                              "aliases": aliases, "meta": meta}})
    run("asdiv", "asdiv", f"{URL}@sha256:{sha}", cands,
        notes=["single corpus, no train/test split in the release -> subset 'all', split 'all'",
               "numeric answers stripped of the unit in parentheses (raw kept in aliases/meta.raw_answer); "
               "non-numeric answers are task 'short'",
               "native id = Problem ID (nluds-xxxx). The exclusion file's asdiv ids are integers: they coincide with "
               "the 0-based XML row index up to ~row 733 and are shifted by one after it, so they index some other "
               "ordering; every one of them is caught by the question hash (Question field), so the row index is "
               "not used as an id"])


if __name__ == "__main__":
    main()
