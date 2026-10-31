"""Small shared driver for the narrativeqa / qanta / pubmedqa / wmt19 / livecodebench loaders (depends only on common).

  hf_file(repo, path, revision, bench) / url_file(url, bench, name=None): pinned downloads into ../raw/<bench>/.

  check(texts, family, ids) -> "question" | "id" | None
      Applies the spec's two rules through common: sha256(norm(text)) for every text in `texts` (the question as we
      emit it first, then any extra source fields that are the same question, e.g. QANTA's first sentence), then
      sha256("<family>:<id>") for every candidate native id in `ids` (the source's own id and/or the row index).
  finish(benchmark, family, source, candidates, extra=None)
      candidates: {subset: [(row, rule_or_None), ...]}. Counts available / excluded per rule / usable per subset,
      runs common.allocate (budget 200, seed 20261031), writes data/<benchmark>.jsonl and MANIFEST/<benchmark>.json.
      hydrate(rows) -> rows, optional: fills heavy fields (e.g. decoded test cases) only for the allocated rows.
"""
import collections
import hashlib
import os
import urllib.request

from huggingface_hub import hf_hub_download

import common as C

RAW = os.path.join(C.ROOT, "raw")  # downloads: raw/<benchmark>/..., parsed as data only, never executed


def raw_dir(bench):
    d = os.path.join(RAW, bench)
    os.makedirs(d, exist_ok=True)
    return d


def hf_file(repo, path, revision, bench):
    """Pinned Hugging Face dataset file (exact commit sha) -> local path."""
    return hf_hub_download(repo, path, repo_type="dataset", revision=revision, cache_dir=os.path.join(raw_dir(bench), "hf"))


def url_file(url, bench, name=None):
    """Pinned URL (e.g. raw.githubusercontent.com/<org>/<repo>/<sha>/...) -> (local path, sha256 of the file)."""
    path = os.path.join(raw_dir(bench), name or url.rstrip("/").split("/")[-1])
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "argrouter-v4-loader"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return path, h.hexdigest()


def check(texts, family, ids):
    for t in texts:
        if t is not None and C.norm(t) != "" and C.q_hash(t) in C.Q_HASHES:  # sha256("") is itself listed
            return "question"
    for i in ids:
        if i is not None and C.id_hash(family, i) in C.ID_HASHES:
            return "id"
    return None


def finish(benchmark, family, source, candidates, extra=None, budget=200, hydrate=None):
    per, usable = {}, {}
    tot = collections.Counter()
    for sub in sorted(candidates):
        c = collections.Counter()
        keep = []
        for row, rule in candidates[sub]:
            c["available"] += 1
            if rule:
                c["excluded_%s_hash" % rule] += 1
            else:
                keep.append(row)
        c["usable"] = len(keep)
        usable[sub] = keep
        per[sub] = dict(c)
        tot.update(c)
    rows = C.allocate(usable, budget=budget)
    if hydrate:
        rows = hydrate(rows)
    for r in rows:
        per[r["subset"]]["taken"] = per[r["subset"]].get("taken", 0) + 1
    for sub in per:
        per[sub].setdefault("taken", 0)
    info = {
        "benchmark": benchmark,
        "exclusion_family": family,
        "source": source,
        "subsets": sorted(candidates),
        "budget": budget,
        "available": tot["available"],
        "excluded_question_hash": tot["excluded_question_hash"],
        "excluded_id_hash": tot["excluded_id_hash"],
        "usable": tot["usable"],
        "taken": len(rows),
        "shortfall": max(0, budget - len(rows)),
        "taken_by_split": dict(collections.Counter(r["split"] for r in rows)),
        "per_subset": per,
        "output": "data/%s.jsonl" % benchmark,
    }
    if extra:
        info.update(extra)
    C.write(benchmark, rows)
    C.update_manifest(benchmark, info)
    print("%s: available=%d excluded_q=%d excluded_id=%d usable=%d taken=%d shortfall=%d" % (
        benchmark, info["available"], info["excluded_question_hash"], info["excluded_id_hash"], info["usable"],
        info["taken"], info["shortfall"]))
    return rows, info
