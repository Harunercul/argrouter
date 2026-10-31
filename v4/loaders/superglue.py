"""SuperGLUE loader (official release https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip).

All 10 SuperGLUE tasks have text-answerable labels: BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC and the two
diagnostics AX-b and AX-g (shipped as a single labelled file, used as split "test"). For the 8 main tasks the test
split is unlabelled, so train (then val -> "validation") is used. Subsets = the 10 tasks.
Label orders follow the release (and the HF super_glue features built from it):
  BoolQ/MultiRC/WiC/WSC labels are booleans -> options ["False", "True"]; CB ["entailment", "contradiction",
  "neutral"]; RTE/AX-b/AX-g ["entailment", "not_entailment"]; COPA label 0/1 = choice1/choice2.
Rendering (native fields also kept in meta): BoolQ context=passage, question=question. CB/RTE/AX: context=premise
(AX-b sentence1), question=hypothesis (AX-b sentence2). COPA: context=premise, question "What was the cause?" /
"What happened as a result?", options [choice1, choice2]. WiC: context = the two sentences, question asks whether the
word has the same meaning in both. WSC: context=text, question asks whether span2 refers to span1. MultiRC: one item
per (question, candidate answer), binary. ReCoRD: context=passage (with @highlight lines), question=query
(@placeholder), answer = first gold entity text, aliases = the other gold texts, meta.entities = candidate entities.
Native ids: the release's idx (MultiRC "<passage>-<question>-<answer>", ReCoRD "<passage>-<query>").
Hashed texts: the item's question/hypothesis-like field first, then its premise/passage-like field.
"""
import json
import zipfile

from loadutil import run, url_file

URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip"
TF = ["False", "True"]
NLI2 = ["entailment", "not_entailment"]
CB3 = ["entailment", "contradiction", "neutral"]


def items(task, r):
    """yield (native_id, hash_texts, row) for one jsonl record."""
    if task == "BoolQ":
        yield r["idx"], [r["question"]], dict(task="binary", context=r["passage"], question=r["question"],
                                               options=TF, answer=int(bool(r["label"])), meta={})
    elif task in ("CB", "RTE", "AX-g"):
        opts = CB3 if task == "CB" else NLI2
        yield r["idx"], [r["hypothesis"], r["premise"]], dict(
            task="mc" if task == "CB" else "binary", context=r["premise"], question=r["hypothesis"], options=opts,
            answer=opts.index(r["label"]), meta={k: v for k, v in r.items() if k not in ("premise", "hypothesis",
                                                                                         "label", "idx")})
    elif task == "AX-b":
        yield r["idx"], [r["sentence2"], r["sentence1"]], dict(
            task="binary", context=r["sentence1"], question=r["sentence2"], options=NLI2,
            answer=NLI2.index(r["label"]), meta={k: v for k, v in r.items() if k not in ("sentence1", "sentence2",
                                                                                         "label", "idx")})
    elif task == "COPA":
        q = "What was the cause?" if r["question"] == "cause" else "What happened as a result?"
        yield r["idx"], [r["premise"]], dict(task="mc", context=r["premise"], question=q,
                                              options=[r["choice1"], r["choice2"]], answer=int(r["label"]),
                                              meta={"question_type": r["question"]})
    elif task == "WiC":
        ctx = f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}"
        q = f'Is the word "{r["word"]}" used with the same meaning in both sentences?'
        yield r["idx"], [r["sentence1"], r["sentence2"]], dict(
            task="binary", context=ctx, question=q, options=TF, answer=int(bool(r["label"])),
            meta={k: r[k] for k in ("word", "sentence1", "sentence2", "start1", "end1", "start2", "end2")})
    elif task == "WSC":
        t = r["target"]
        q = f'In the passage, does the pronoun "{t["span2_text"]}" refer to "{t["span1_text"]}"?'
        yield r["idx"], [r["text"]], dict(task="binary", context=r["text"], question=q, options=TF,
                                           answer=int(bool(r["label"])), meta={"target": t})
    elif task == "MultiRC":
        p = r["passage"]
        for q in p["questions"]:
            for a in q["answers"]:
                nid = f"{r['idx']}-{q['idx']}-{a['idx']}"
                yield nid, [q["question"], p["text"]], dict(
                    task="binary", context=p["text"], question=f"{q['question']}\nCandidate answer: {a['text']}",
                    options=TF, answer=int(a["label"]),
                    meta={"question": q["question"], "candidate_answer": a["text"]})
    elif task == "ReCoRD":
        p = r["passage"]
        ents = [p["text"][e["start"]:e["end"] + 1] for e in p["entities"]]
        for q in r["qas"]:
            golds = list(dict.fromkeys(a["text"] for a in q["answers"]))
            if not golds:
                continue
            yield f"{r['idx']}-{q['idx']}", [q["query"], p["text"]], dict(
                task="short", context=p["text"], question=q["query"], options=[], answer=golds[0],
                aliases=golds[1:], meta={"entities": list(dict.fromkeys(ents)), "source": r.get("source")})
    else:
        raise ValueError(task)


def main():
    path, sha = url_file(URL, "superglue")
    z = zipfile.ZipFile(path)
    files = {"AX-b": [("AX-b/AX-b.jsonl", "test")], "AX-g": [("AX-g/AX-g.jsonl", "test")]}
    for t in ["BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC", "WSC"]:
        files[t] = [(f"{t}/train.jsonl", "train"), (f"{t}/val.jsonl", "validation")]
    cands = []
    for task, fl in files.items():
        for fname, split in fl:
            for line in z.read(fname).decode("utf-8").splitlines():
                if not line.strip():
                    continue
                for nid, texts, row in items(task, json.loads(line)):
                    row.setdefault("aliases", [])
                    cands.append({"subset": task, "split": split, "native_id": nid, "hash_texts": texts,
                                  "row": row})
    run("superglue", "superglue", f"{URL}@sha256:{sha}", cands,
        notes=["subsets = all 10 SuperGLUE tasks incl. diagnostics AX-b/AX-g (labelled; only file is the test "
               "split); the 8 main tasks use train/val only (test unlabelled)",
               "COPA/WiC/WSC/MultiRC question strings are fixed renderings of the native fields (kept in meta)",
               "native id = release idx; id-hash matches are positional (exclusion file's superglue ids are ints)"])


if __name__ == "__main__":
    main()
