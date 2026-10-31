"""Prompt-format robustness variant B (ours, written without RouterArena's templates): a system message,
different instruction wording, option labels in parentheses and a \\boxed{} answer. Used only to test
whether a policy fitted on variant A labels holds when the prompt format changes; never for training."""

from __future__ import annotations

from prompts_v4 import LANG, LETTERS, _label_note, _meta, template_key

SYSTEM = "You are a careful assistant. Think as much as you need, then give the final answer in \\boxed{}."


def render_b(row: dict) -> str:
    key, meta = template_key(row), _meta(row)
    q, ctx = row["question"].strip(), (row.get("context") or "").strip()
    if key == "translate":
        s, t = meta.get("src_lang", ""), meta.get("tgt_lang", "")
        src, tgt = meta.get("src_lang_name") or LANG.get(s, s), meta.get("tgt_lang_name") or LANG.get(t, t)
        return f"{src} -> {tgt}:\n{q}\n\nPut only the {tgt} translation inside \\boxed{{}}."
    if key == "reading":
        return f"{ctx}\n\n---\n{q}\nAnswer briefly; put the answer inside \\boxed{{}}."
    pre = (ctx + "\n\n") if ctx else ""
    if key == "mc":
        opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(row.get("options") or []))
        return f"{pre}Question: {q}\n{opts}\n{_label_note(row)}Put the letter of the correct choice inside \\boxed{{}}."
    if key == "numeric":
        return f"{pre}{q}\n\nGive the final result as a bare number (or yes/no) inside \\boxed{{}}."
    hint = ' (the word or name replacing "@placeholder")' if "@placeholder" in q else ""
    return f"{pre}{q}\n\nPut your final answer{hint}, as short as possible, inside \\boxed{{}}."


def build_messages_b(row: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render_b(row)}]
