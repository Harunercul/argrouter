"""Shared finishing step for the MMLU-Pro / MMLU / ArcMMLU / MedMCQA / OpenTDB / GeoBench / MusicTheoryBench loaders.

finish(benchmark, family, source, candidates, extra=None)
  candidates: {subset: [cand, ...]} where cand = {"row": <output row in the spec schema>,
                                                   "native_id": <id used for the id-hash rule>,
                                                   "qtexts": [question text(s) checked by the question-hash rule]}
  Applies the exclusion (question hash on every listed text, then "<family>:<native_id>"), allocates 200 with
  common.allocate, writes data/<benchmark>.jsonl and the manifest entry. Returns the manifest info.
"""
import collections
import json

import common
import src_fetch
from huggingface_hub import hf_hub_download

BUDGET = 200


def hf_file(repo, path, revision, bench):
    """Pinned HF dataset file: reuse the user's default HF cache when it already holds this exact commit's file,
    otherwise download via src_fetch.hf_file into raw/<bench>/."""
    try:
        return hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_files_only=True)
    except Exception:
        return src_fetch.hf_file(repo, path, revision, bench)


def excluded_reason(family, cand):
    texts = cand["qtexts"]
    for t in texts:
        if common.q_hash(t) in common.Q_HASHES:
            return "question"
    return common.is_excluded(texts[0], family, cand["native_id"])


def finish(benchmark, family, source, candidates, extra=None):
    per = {}
    kept = {}
    tot = collections.Counter()
    for sub in sorted(candidates):
        c = collections.Counter({"available": 0, "excluded_question_hash": 0, "excluded_id_hash": 0})
        keep = []
        for cand in candidates[sub]:
            c["available"] += 1
            r = excluded_reason(family, cand)
            if r == "question":
                c["excluded_question_hash"] += 1
            elif r == "id":
                c["excluded_id_hash"] += 1
            else:
                keep.append(cand["row"])
        c["usable"] = len(keep)
        kept[sub] = keep
        per[sub] = c
        tot.update(c)
    rows = common.allocate(kept, BUDGET)
    taken = collections.Counter(r["subset"] for r in rows)
    for sub in per:
        per[sub]["taken"] = taken.get(sub, 0)
    path = common.write(benchmark, rows)
    info = {
        "benchmark": benchmark,
        "exclusion_family": family,
        "source": source,
        "subsets": sorted(candidates),
        "budget": BUDGET,
        "available": tot["available"],
        "excluded_question_hash": tot["excluded_question_hash"],
        "excluded_id_hash": tot["excluded_id_hash"],
        "usable": tot["usable"],
        "taken": len(rows),
        "shortfall": max(0, BUDGET - len(rows)),
        "taken_by_split": dict(collections.Counter(r["split"] for r in rows)),
        "per_subset": {k: dict(v) for k, v in per.items()},
        "output": "data/" + path.rsplit("/", 1)[-1],
    }
    if extra:
        info.update(extra)
    common.update_manifest(benchmark, info)
    print(json.dumps({k: v for k, v in info.items() if k != "per_subset"}, ensure_ascii=False, indent=1))
    return info


def row(benchmark, subset, out_id, split, task, question, options, answer, source, context="", aliases=None, meta=None):
    return {
        "id": f"{benchmark}/{subset}/{out_id}",
        "benchmark": benchmark,
        "subset": subset,
        "split": split,
        "task": task,
        "context": context or "",
        "question": question,
        "options": list(options),
        "answer": answer,
        "aliases": aliases or [],
        "meta": meta or {},
        "source": source,
    }
