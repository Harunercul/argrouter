"""Shared helpers for the argrouter v4 calibration loaders (see ../DATA_SPEC_v4.md).

API (kept deliberately small):
  norm(s)                                   -> spec normalisation
  is_excluded(question, family, native_id)  -> "question" | "id" | None
  allocate(subsets, budget=200)             -> list of rows (round-robin, seed 20261031)
  write(benchmark, rows)                    -> writes data/<benchmark>.jsonl
  update_manifest(benchmark, info)          -> writes MANIFEST/<benchmark>.json and rebuilds MANIFEST.json (file-locked)
"""
import fcntl
import hashlib
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../route_data/v4src
DATA_DIR = os.path.join(ROOT, "data")
MANIFEST_DIR = os.path.join(ROOT, "MANIFEST")
SEED = 20261031
SPLIT_ORDER = {"train": 0, "validation": 1, "dev": 1, "val": 1, "test": 2}

with open(os.path.join(ROOT, "exclude_hashes.json")) as _f:
    _EX = json.load(_f)
Q_HASHES = frozenset(_EX["question_sha256"])
ID_HASHES = frozenset(_EX["id_sha256"])


def norm(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()[:300]


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def q_hash(question):
    return sha(norm(question))


def id_hash(family, native_id):
    return sha(f"{family}:{native_id}")


def is_excluded(question, family, native_id):
    """Return "question" if sha256(norm(question)) is listed, else "id" if sha256("<family>:<native_id>") is listed,
    else None. The question rule is checked first so each candidate is counted under exactly one rule."""
    if q_hash(question) in Q_HASHES:
        return "question"
    if native_id is not None and id_hash(family, native_id) in ID_HASHES:
        return "id"
    return None


def allocate(subsets, budget=200, seed=SEED):
    """subsets: {subset_name: [row, ...]} with excluded/unusable rows already removed; each row has "split".
    Within a subset, rows are ordered by split preference (train, validation/dev, test), uniformly shuffled within
    each split using one random.Random(seed) consumed in sorted subset order. Then round-robin over sorted subsets,
    one row per subset per pass; an exhausted subset simply drops out (its share returns to the round-robin)."""
    rng = random.Random(seed)
    queues = {}
    for name in sorted(subsets):
        by_split = {}
        for r in subsets[name]:
            by_split.setdefault(r.get("split", "test"), []).append(r)
        q = []
        for sp in sorted(by_split, key=lambda s: (SPLIT_ORDER.get(s, 3), s)):
            items = list(by_split[sp])
            rng.shuffle(items)
            q.extend(items)
        queues[name] = q
    out, pos = [], {n: 0 for n in queues}
    while len(out) < budget:
        progressed = False
        for n in sorted(queues):
            if len(out) >= budget:
                break
            if pos[n] < len(queues[n]):
                out.append(queues[n][pos[n]])
                pos[n] += 1
                progressed = True
        if not progressed:
            break
    return out


def write(benchmark, rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{benchmark}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def update_manifest(benchmark, info):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(os.path.join(MANIFEST_DIR, f"{benchmark}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    lock_path = os.path.join(ROOT, ".manifest.lock")
    with open(lock_path, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            allm = {}
            for fn in sorted(os.listdir(MANIFEST_DIR)):
                if fn.endswith(".json"):
                    with open(os.path.join(MANIFEST_DIR, fn), encoding="utf-8") as f:
                        allm[fn[:-5]] = json.load(f)
            tmp = os.path.join(ROOT, "MANIFEST.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(allm, f, ensure_ascii=False, indent=2)
            os.replace(tmp, os.path.join(ROOT, "MANIFEST.json"))
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
