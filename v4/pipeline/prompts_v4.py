"""Clean-room prompt renderer (v4), for the v4src schema (route_data/v4src/DATA_SPEC_v4.md).

build_messages(row) -> [{"role": "user", "content": ...}]

One generic template per task type (mc/binary, numeric, short, reading, translate, code). The only
data-driven additions are field-driven and benchmark-agnostic:
  * binary rows whose meta.label_semantics reads "1 = <meaning>" get a one-line note naming the option
    letter that carries that meaning;
  * a question containing the literal "@placeholder" field marker gets one line saying what to fill in.
Options are always lettered A, B, C... in row order, so letter i is option index i (= row["answer"]).
"""
from __future__ import annotations

import json
import re
import string

LETTERS = string.ascii_uppercase

LANG = {"cs": "Czech", "de": "German", "en": "English", "fi": "Finnish", "fr": "French", "gu": "Gujarati",
        "kk": "Kazakh", "lt": "Lithuanian", "ru": "Russian", "zh": "Chinese"}

_END = 'End your response with a final line of the form "Answer: {slot}"'

TEMPLATES = {
    "mc": ("{context}{question}\n\nOptions:\n{options}\n{note}\n"
           "Choose the single best option. You may reason briefly first. "
           + _END.format(slot="X") + ", where X is the option letter."),
    "numeric": ("{context}{question}\n\n"
                "Solve the problem. You may show your work. " + _END.format(slot="<final answer>")
                + ", giving only the final result (a number, or yes/no if the question asks for it), without units."),
    "short": ("{context}{question}\n{note}\n"
              "You may reason first if needed. " + _END.format(slot="<your answer>")
              + ", where the answer is as short as possible (a name, word, phrase, number or expression)."),
    "reading": ("Read the passage and answer the question.\n\nPassage:\n{context_raw}\n\nQuestion: {question}\n\n"
                + _END.format(slot="<your answer>") + ", a short phrase or sentence."),
    "translate": ("Translate the following {src} text into {tgt}.\n\n{question}\n\n"
                  + _END.format(slot="<translation>") + ", containing only the translation."),
    "code_stdin": ("{question}\n\n"
                   "Write a complete Python 3 program that reads the input from standard input and writes "
                   "the answer to standard output. Return the whole program in a single ```python code block."),
    "code_func": ("{question}\n\nComplete the following starter code:\n```python\n{starter}\n```\n\n"
                  "Return the complete solution (the full class or function) in a single ```python code block."),
}


def _meta(row: dict) -> dict:
    m = row.get("meta") or {}
    return json.loads(m) if isinstance(m, str) else m


def template_key(row: dict) -> str:
    task = row["task"]
    if task == "binary":
        return "mc"
    if task == "code":
        return "code_func" if (_meta(row).get("starter_code") or "").strip() else "code_stdin"
    return task


def _label_note(row: dict) -> str:
    m = re.match(r"\s*(\d+)\s*=\s*(.+)", str(_meta(row).get("label_semantics", "")))
    if row["task"] == "binary" and m and int(m.group(1)) < len(row["options"]):
        return f"(Option {LETTERS[int(m.group(1))]} means: {m.group(2).strip().rstrip('.')}.)\n"
    return ""


def render(row: dict) -> str:
    key, meta = template_key(row), _meta(row)
    q, ctx = row["question"].strip(), (row.get("context") or "").strip()
    if key == "translate":
        s, t = meta.get("src_lang", ""), meta.get("tgt_lang", "")
        return TEMPLATES[key].format(src=meta.get("src_lang_name") or LANG.get(s, s),
                                     tgt=meta.get("tgt_lang_name") or LANG.get(t, t), question=q)
    if key == "code_func":
        return TEMPLATES[key].format(question=q, starter=meta["starter_code"].rstrip())
    if key == "code_stdin":
        return TEMPLATES[key].format(question=q)
    if key == "reading":
        return TEMPLATES[key].format(context_raw=ctx, question=q)
    note = _label_note(row) if key == "mc" else ""
    if key == "short" and "@placeholder" in q:
        note = 'Give the word or name that should replace "@placeholder".\n'
    options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row.get("options") or []))
    return TEMPLATES[key].format(context=(ctx + "\n\n") if ctx else "", question=q, options=options,
                                 note=note)


def build_messages(row: dict) -> list[dict]:
    return [{"role": "user", "content": render(row)}]


if __name__ == "__main__":  # print one rendered prompt per (benchmark, subset, task)
    import sys
    seen = set()
    for path in sys.argv[1:]:
        for line in open(path):
            r = json.loads(line)
            k = (r["benchmark"], r["subset"], r["task"])
            if k not in seen:
                seen.add(k)
                print("=" * 20, k, "\n" + render(r)[:1200])
