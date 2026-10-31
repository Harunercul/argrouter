"""ETHICS loader (hendrycks/ethics, the authors' HF repo; data/<subset>/{train,test,test_hard}.csv read directly).

Label semantics (Hendrycks et al., ICLR 2021, "Aligning AI With Shared Human Values", and the repo's README/code):
- commonsense (label,input,is_short,edited): label 1 = the first-person action is clearly morally WRONG, 0 = not wrong.
  options ["not wrong", "wrong"]. Long items (is_short=False, Reddit AITA posts) are kept in full.
- deontology (label,scenario,excuse): scenario = a request/role, excuse = the response; label 1 = the excuse is
  reasonable. options ["unreasonable", "reasonable"].
- justice (label,scenario): label 1 = the justification is reasonable. options ["unreasonable", "reasonable"].
- virtue (label,scenario "<sentence> [SEP] <trait>"): label 1 = the trait is exhibited by the character in the
  sentence. options ["no", "yes"].
- utilitarianism (baseline,less_pleasant): no label column; by construction the first scenario (baseline) is more
  pleasant for the person than the second. Rendered as a 2-way choice "which is more pleasant"; the presentation
  order is randomised per item with random.Random("20261031:utilitarianism:<split>:<row>") (deterministic) and the
  answer is the index of the baseline.
Splits: train, test, test_hard (kept as named; train preferred). commonsense ambig.csv has no labels and is skipped.
Subsets: the 5 ETHICS subsets. Native id = row index within <subset>/<split>.csv.
Hashed texts: commonsense input; deontology scenario (+ "scenario excuse"); justice scenario; virtue the sentence
(before [SEP]) and the full scenario field; utilitarianism both scenarios. Scenarios shared by several rows
(deontology/virtue) are excluded together.
"""
import random

import pandas as pd

from loadutil import hf_file, run

REPO, REV = "hendrycks/ethics", "b8b47c589f8bee77175b8648e5497278b68da48a"
SUBSETS = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]


def build(subset, split, i, r):
    if subset == "commonsense":
        return ([r["input"]], {"task": "binary", "context": "", "question": r["input"],
                               "options": ["not wrong", "wrong"], "answer": int(r["label"]), "aliases": [],
                               "meta": {"is_short": r["is_short"], "edited": r["edited"],
                                        "label_semantics": "1 = clearly morally wrong"}})
    if subset == "deontology":
        return ([r["scenario"], f"{r['scenario']} {r['excuse']}"],
                {"task": "binary", "context": r["scenario"], "question": r["excuse"],
                 "options": ["unreasonable", "reasonable"], "answer": int(r["label"]), "aliases": [],
                 "meta": {"scenario": r["scenario"], "excuse": r["excuse"],
                          "label_semantics": "1 = the excuse/response is reasonable"}})
    if subset == "justice":
        return ([r["scenario"]], {"task": "binary", "context": "", "question": r["scenario"],
                                  "options": ["unreasonable", "reasonable"], "answer": int(r["label"]),
                                  "aliases": [], "meta": {"label_semantics": "1 = reasonable"}})
    if subset == "virtue":
        sent, _, trait = r["scenario"].partition(" [SEP] ")
        return ([sent, r["scenario"]], {"task": "binary", "context": sent.strip(), "question": trait.strip(),
                                        "options": ["no", "yes"], "answer": int(r["label"]), "aliases": [],
                                        "meta": {"scenario": r["scenario"], "trait": trait.strip(),
                                                 "label_semantics": "1 = the character exhibits the trait"}})
    if subset == "utilitarianism":
        flip = random.Random(f"20261031:utilitarianism:{split}:{i}").random() < 0.5
        opts = [r["less_pleasant"], r["baseline"]] if flip else [r["baseline"], r["less_pleasant"]]
        return ([r["baseline"], r["less_pleasant"]],
                {"task": "mc", "context": "", "question": "Which scenario is more pleasant for the person?",
                 "options": opts, "answer": 1 if flip else 0, "aliases": [],
                 "meta": {"baseline": r["baseline"], "less_pleasant": r["less_pleasant"], "order_flipped": flip,
                          "label_semantics": "baseline is more pleasant than less_pleasant (by construction)"}})
    raise ValueError(subset)


def main():
    cands = []
    for subset in SUBSETS:
        for split in ["train", "test", "test_hard"]:
            df = pd.read_csv(hf_file(REPO, f"data/{subset}/{split}.csv", REV, "ethics"), dtype=str,
                             keep_default_na=False)
            for i, r in enumerate(df.to_dict("records")):
                texts, row = build(subset, split, i, r)
                cands.append({"subset": subset, "split": split, "native_id": i, "hash_texts": texts, "row": row})
    run("ethics", "ethics", f"{REPO}@{REV}", cands,
        notes=["label semantics per subset in meta.label_semantics (commonsense 1 = wrong; deontology/justice "
               "1 = reasonable; virtue 1 = trait fits)",
               "utilitarianism has no labels: rendered as a 2-way choice with deterministic per-item order "
               "randomisation; answer = index of the baseline (more pleasant) scenario",
               "native id = row index within <subset>/<split>.csv; id-hash matches are positional (the exclusion "
               "file's ethics ids are 0..99)"])


if __name__ == "__main__":
    main()
