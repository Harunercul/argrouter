"""GeoBench / GeoBenchmark (K2 geoscience LLM, Deng et al. 2023, arXiv 2306.05064) from daven3/geobench (json files).
The release has the AP Study objective set (geobenchmark_apstudy.json, 1,395 MC) and the NPEE file with six task types
(noun, choice, completion, tf, qa, discussion). Subsets used = the objective, auto-gradable ones:
  apstudy (mc), npee_choice (mc, options parsed from "Choose from:" lines), npee_tf (binary True/False).
npee_noun / npee_completion / npee_qa / npee_discussion are the authors' subjective (human-graded, free-form) tasks
and are recorded but not used. No splits are shipped: split = "test".
No ids are shipped (apstudy `id` is a constant string): native id = 0-based row index in the subset's list."""
import json
import re

from mc_runner import finish, hf_file, row

BENCH, FAMILY = "geobench", "geobench"
REPO, REV = "daven3/geobench", "fa6f583ad38f93a7e7f5a3ea1582fe6593437747"
SOURCE = f"{REPO}@{REV}"
OPT_RE = re.compile(r"^([A-H])[.．、]\s*(.*)$")


def parse_choice(q):
    stem, sep, rest = q.partition("\nChoose from:")
    if not sep:
        return None
    opts, labels = [], []
    for line in rest.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = OPT_RE.match(line)
        if m:
            labels.append(m.group(1))
            opts.append(m.group(2).strip())
        elif opts:
            opts[-1] += "\n" + line
        else:
            return None
    if sorted(labels) != list("ABCDEFGH"[: len(labels)]) or len(opts) < 2:
        return None
    # a few items list the labels out of order (e.g. A, C, B, D): order the options by their label
    return stem.strip(), [o for _, o in sorted(zip(labels, opts))]


def main():
    cands, unparsed = {}, []
    ap = json.load(open(hf_file(REPO, "geobenchmark_apstudy.json", REV, BENCH), encoding="utf-8"))
    for i, x in enumerate(ap):
        ch = x["question"]["choices"]
        labels = [c["label"] for c in ch]
        out = row(BENCH, "apstudy", i, "test", "mc", x["question"]["stem"], [c["text"] for c in ch],
                  labels.index(x["answerKey"]), SOURCE, meta={"row_index": i, "labels": labels, "source_id": x["id"]})
        cands.setdefault("apstudy", []).append({"row": out, "native_id": i, "qtexts": [x["question"]["stem"]]})
    npee = json.load(open(hf_file(REPO, "geobenchmark_npee.json", REV, BENCH), encoding="utf-8"))
    for i, (q, a) in enumerate(zip(npee["choice"]["question"], npee["choice"]["answer"])):
        p = parse_choice(q)
        a = a.strip()
        if p is None or len(a) != 1 or "ABCDEFGH".index(a) >= len(p[1]):
            unparsed.append(i)
            continue
        out = row(BENCH, "npee_choice", i, "test", "mc", p[0], p[1], "ABCDEFGH".index(a), SOURCE,
                  meta={"row_index": i, "raw_question": q})
        cands.setdefault("npee_choice", []).append({"row": out, "native_id": i, "qtexts": [q, p[0]]})
    for i, (q, a) in enumerate(zip(npee["tf"]["question"], npee["tf"]["answer"])):
        a = a.strip()
        assert a in ("True", "False"), a
        out = row(BENCH, "npee_tf", i, "test", "binary", q, ["True", "False"], ["True", "False"].index(a), SOURCE,
                  meta={"row_index": i})
        cands.setdefault("npee_tf", []).append({"row": out, "native_id": i, "qtexts": [q]})
    unused = {k: len(npee[k]["question"]) for k in ("noun", "completion", "qa", "discussion")}
    finish(BENCH, FAMILY, SOURCE, cands, {"npee_choice_unparsed_dropped": unparsed,
                                           "subjective_subsets_not_used": unused})


if __name__ == "__main__":
    main()
