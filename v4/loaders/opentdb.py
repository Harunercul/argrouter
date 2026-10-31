"""OpenTDB (Open Trivia Database, opentdb.com, CC BY-SA 4.0). opentdb.com has no bulk download, so this uses the
public full dump of its verified questions in GitHub kickturn/opentdb-scraper (opentdb.json, raw API objects, HTML
entity encoded; scraped 2026-07-28), pinned to a commit. 5,250 questions, 24 categories.
Subsets = all `category` values (HTML-unescaped). No splits are shipped: split = "train".
No ids are shipped: native id = 0-based row index in opentdb.json `results`.
type "multiple" -> task mc, options = correct + incorrect answers sorted alphabetically (deterministic, the API's own
order is random); type "boolean" -> task binary, options ["True", "False"]. Text is HTML-unescaped; the question-hash
rule is applied to both the unescaped and the raw (entity-encoded) question."""
import html
import json

from mc_runner import finish, row
from src_fetch import url_file

BENCH, FAMILY = "opentdb", "opentdb"
GH, COMMIT = "kickturn/opentdb-scraper", "40221acc1fc3440a5c2b07621cbd05460e4eafac"
URL = f"https://raw.githubusercontent.com/{GH}/{COMMIT}/opentdb.json"
SOURCE = f"https://github.com/{GH} (opentdb.json)@{COMMIT}"


def main():
    path, digest = url_file(URL, BENCH, "opentdb.json")
    res = json.load(open(path, encoding="utf-8"))["results"]
    cands = {}
    for i, x in enumerate(res):
        cat = html.unescape(x["category"])
        q = html.unescape(x["question"])
        correct = html.unescape(x["correct_answer"])
        wrong = [html.unescape(w) for w in x["incorrect_answers"]]
        meta = {"row_index": i, "type": x["type"], "difficulty": x["difficulty"], "category": cat}
        if x["type"] == "boolean":
            opts = ["True", "False"]
            assert correct in opts and set(wrong) | {correct} == set(opts), i
            out = row(BENCH, cat, i, "train", "binary", q, opts, opts.index(correct), SOURCE, meta=meta)
        else:
            opts = sorted(set([correct] + wrong))
            assert len(opts) == len(wrong) + 1, i
            out = row(BENCH, cat, i, "train", "mc", q, opts, opts.index(correct), SOURCE, meta=meta)
        cands.setdefault(cat, []).append({"row": out, "native_id": i, "qtexts": [q, x["question"]]})
    finish(BENCH, FAMILY, SOURCE, cands, {"file_sha256": digest, "source_note": "community dump of the opentdb.com API"})


if __name__ == "__main__":
    main()
