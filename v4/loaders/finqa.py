"""FinQA loader (authors' repository czyssrs/FinQA, dataset/{train,dev,test}.json at a pinned commit).

Each example: id, pre_text (list of sentences), table (list of rows, first row = header), post_text, qa{question,
answer, exe_ans, program, gold_inds, ...}. Context = pre_text + the table serialised as text (one row per line, cells
joined with " | ") + post_text, all kept in full. Answer: qa.answer (the human-readable answer string, e.g. "59.0%");
when it is empty (74 items) the execution result qa.exe_ans is used. str(exe_ans) is added as an alias when it differs
(FinQA's official metric is execution accuracy against exe_ans). One subset; dev -> split "validation".
Native id = the example id ("ADI/2009/page_49.pdf-1"). Hashed text: qa.question. private_test.json has no answers.
"""
import json

from loadutil import run, url_file

COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
BASE = f"https://raw.githubusercontent.com/czyssrs/FinQA/{COMMIT}/dataset"


def table_text(table):
    return "\n".join(" | ".join(str(c).strip() for c in row) for row in table)


def main():
    cands, shas = [], {}
    for fname, split in [("train", "train"), ("dev", "validation"), ("test", "test")]:
        path, sha = url_file(f"{BASE}/{fname}.json", "finqa")
        shas[fname] = sha
        for x in json.load(open(path, encoding="utf-8")):
            qa = x["qa"]
            exe = qa.get("exe_ans")
            exe_s = str(exe) if exe is not None else ""
            ans = str(qa.get("answer", "")).strip() or exe_s
            aliases = [exe_s] if exe_s and exe_s != ans else []
            ctx = "\n".join(x["pre_text"]) + "\n\n" + table_text(x["table"]) + "\n\n" + "\n".join(x["post_text"])
            cands.append({"subset": "finqa", "split": split, "native_id": x["id"], "hash_texts": [qa["question"]],
                          "row": {"task": "numeric", "context": ctx, "question": qa["question"], "options": [],
                                  "answer": ans, "aliases": aliases,
                                  "meta": {"exe_ans": exe, "raw_answer": qa.get("answer", ""),
                                           "program": qa.get("program"), "gold_inds": qa.get("gold_inds"),
                                           "filename": x.get("filename"), "table": x["table"]}}})
    src = f"https://github.com/czyssrs/FinQA@{COMMIT} (dataset/train,dev,test.json; sha256 " + \
          ", ".join(f"{k}={v[:12]}" for k, v in shas.items()) + ")"
    run("finqa", "finqa", src, cands,
        notes=["context = pre_text lines + table (rows as 'a | b | c' lines, header first) + post_text lines",
               "answer = qa.answer, falling back to str(qa.exe_ans) when empty; exe_ans kept in aliases and meta",
               "yes/no questions keep their 'yes'/'no' answers (task stays 'numeric', as in the release)",
               "native id = FinQA example id; the exclusion file's finqa ids are integers (row indices of a test-like "
               "file), so the id rule cannot match these ids"])


if __name__ == "__main__":
    main()
