"""Helpers shared by the gsm8k/asdiv/math/aime/mathqa/finqa/chessinstruct/socialiqa/ethics/superglue loaders.

  hf_file(repo, path, revision, bench)        -> local path (hf_hub_download, repo_type=dataset, exact commit sha)
  url_file(url, bench, name=None)             -> (local path, sha256 of the file)
  excluded(texts, family, native_id)          -> "question" | "id" | None
  run(benchmark, family, source, cands, ...)  -> exclusion + allocation (common.allocate) + write + manifest

Downloads land in ../raw/<bench>/ and are only parsed as data (json/csv/parquet/xml/zip), never executed.
"""
import collections
import hashlib
import os
import urllib.request

from huggingface_hub import hf_hub_download

import common

ROOT = common.ROOT
RAW = os.path.join(ROOT, "raw")


def _dir(bench):
    d = os.path.join(RAW, bench)
    os.makedirs(d, exist_ok=True)
    return d


def hf_file(repo, path, revision, bench):
    return hf_hub_download(repo, path, repo_type="dataset", revision=revision, cache_dir=os.path.join(_dir(bench), "hf"))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def url_file(url, bench, name=None):
    path = os.path.join(_dir(bench), name or url.rstrip("/").split("/")[-1])
    if not os.path.exists(path):
        tmp = path + ".part"
        req = urllib.request.Request(url, headers={"User-Agent": "argrouter-v4-loader"})
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, path)
    return path, file_sha256(path)


def excluded(texts, family, native_id, alt_ids=()):
    """Spec exclusion. texts[0] is the item's native question field; further texts are other natural renderings of the
    same item (checked too, conservatively). alt_ids: other native ids of the same item (e.g. the row index of a
    single-file release whose rows also carry their own id strings), each checked with the id rule. Empty strings are never hashed (sha256("") is itself on the list)."""
    texts = [t for t in texts if t is not None and common.norm(t) != ""]
    for t in texts:
        if common.is_excluded(t, family, None) == "question":
            return "question"
    for nid in [native_id, *alt_ids]:
        if nid is not None and common.id_hash(family, nid) in common.ID_HASHES:
            return "id"
    return None


def run(benchmark, family, source, cands, notes=None, dropped=None, extra=None, budget=200):
    """cands: list of dicts {"subset", "split", "native_id", ["alt_ids"], "hash_texts": [str], "row": <output row minus id/benchmark/
    subset/split/source>}. dropped: {subset: {reason: n}} for items removed before exclusion (modality/malformed)."""
    dropped = dropped or {}
    per = collections.OrderedDict()
    usable = collections.defaultdict(list)
    for c in cands:
        st = per.setdefault(c["subset"], collections.Counter())
        st["available"] += 1
        why = excluded(c["hash_texts"], family, c["native_id"], c.get("alt_ids", ()))
        if why:
            st["excluded_" + why + "_hash"] += 1
            continue
        row = {"id": f"{benchmark}/{c['subset']}/{c['native_id']}", "benchmark": benchmark, "subset": c["subset"],
               "split": c["split"]}
        r = dict(c["row"])
        row.update({k: r.pop(k) for k in ["task", "context", "question", "options", "answer", "aliases", "meta"]})
        row["source"] = source
        usable[c["subset"]].append(row)
    for s in dropped:
        per.setdefault(s, collections.Counter())
    taken = common.allocate(dict(usable), budget=budget)
    ids = [r["id"] for r in taken]
    assert len(ids) == len(set(ids)), "duplicate ids"
    common.write(benchmark, taken)
    tc = collections.Counter(r["subset"] for r in taken)
    info = {"benchmark": benchmark, "exclusion_family": family, "source": source, "subsets": sorted(per),
            "budget": budget,
            "available": sum(p["available"] for p in per.values()),
            "dropped_before_exclusion": {s: dict(v) for s, v in dropped.items() if v},
            "excluded_question_hash": sum(p["excluded_question_hash"] for p in per.values()),
            "excluded_id_hash": sum(p["excluded_id_hash"] for p in per.values()),
            "usable": sum(len(v) for v in usable.values()), "taken": len(taken),
            "shortfall": max(0, budget - len(taken)),
            "taken_by_split": dict(collections.Counter(r["split"] for r in taken)),
            "per_subset": {s: {"available": per[s]["available"],
                               "excluded_question_hash": per[s]["excluded_question_hash"],
                               "excluded_id_hash": per[s]["excluded_id_hash"],
                               "usable": len(usable.get(s, [])), "taken": tc.get(s, 0)} for s in sorted(per)},
            "notes": notes or []}
    if extra:
        info.update(extra)
    common.update_manifest(benchmark, info)
    print(f"{benchmark}: available={info['available']} excl_q={info['excluded_question_hash']} "
          f"excl_id={info['excluded_id_hash']} usable={info['usable']} taken={info['taken']} "
          f"shortfall={info['shortfall']} by_subset={dict(sorted(tc.items()))}")
    return taken, info
