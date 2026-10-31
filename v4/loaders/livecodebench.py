"""LiveCodeBench code generation (Jain et al. 2024), Hugging Face release livecodebench/code_generation_lite, all six
jsonl files (test.jsonl .. test6.jsonl = version tag release_latest / release_v6, 1055 problems, May 2023 - Apr 2025),
read line by line with json only. The repository's loading script is NOT run (it only concatenates the files in this
order into one `test` split; that order defines our row index).

Why lite: the maintainers' card says the original livecodebench/code_generation (release_v1 only, 400 problems,
~9.4 GB of uncompressed tests) is superseded and "going forward, livecodebench will be using this lite version".
Fields: question_title, question_content, platform, question_id, contest_id, contest_date, starter_code, difficulty,
public_test_cases (JSON string), private_test_cases (base64 -> zlib -> pickle of a JSON string), metadata (JSON; LeetCode
rows carry func_name).
Private tests are decoded with a restricted unpickler that refuses every global / persistent id, so only builtin
containers, strings and numbers can be produced; zlib output is size-capped. Nothing from the data is executed.

Subsets = `platform` (atcoder, codeforces, leetcode). Split = test (the only split). Task code: question =
question_content, answer = "" (no reference solution; correctness is by tests), meta = public_tests, private_tests
(decoded lists of {input, output, testtype}), starter_code, fn_name, contest_date, difficulty, title, ids.
Native id candidates = question_id and the row index in the concatenated release.
Two illustrative-image mentions exist (arc196_b, leetcode 3687); the statements are fully specified in text, so nothing
is dropped for modality.
"""
import base64
import io
import json
import os
import pickle
import zlib

from huggingface_hub import hf_hub_download

from rowkit import RAW, check, finish

BENCH, FAMILY = "livecodebench", "livecodebench"
REPO, REV = "livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
SOURCE = f"{REPO}@{REV}"
FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
MAX_DECOMPRESSED = 1 << 30


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global {module}.{name} refused")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent id refused")


def decode_private(s):
    s = s.strip()
    if s.startswith("["):
        obj = s
    else:
        d = zlib.decompressobj()
        raw = d.decompress(base64.b64decode(s), MAX_DECOMPRESSED)
        if d.unconsumed_tail:
            raise ValueError("decompressed private tests exceed cap")
        obj = _NoGlobals(io.BytesIO(raw)).load()
    if isinstance(obj, bytes):
        obj = obj.decode("utf-8")
    if isinstance(obj, str):
        obj = json.loads(obj)
    assert isinstance(obj, list) and all(isinstance(t, dict) for t in obj), type(obj)
    return obj


def _path(fn):
    return hf_hub_download(REPO, fn, repo_type="dataset", revision=REV,
                           local_dir=os.path.join(RAW, BENCH, "code_generation_lite"))


def main():
    paths = [_path(fn) for fn in FILES]
    cands, g = {}, 0
    for fn, p in zip(FILES, paths):
        with open(p, encoding="utf-8") as fh:
            for li, line in enumerate(fh):
                r = json.loads(line)
                md = json.loads(r["metadata"] or "{}")
                plat = r["platform"]
                row = {
                    "id": f"{BENCH}/{plat}/{r['question_id']}", "benchmark": BENCH, "subset": plat, "split": "test",
                    "task": "code", "context": "", "question": r["question_content"], "options": [], "answer": "",
                    "aliases": [],
                    "meta": {"question_id": r["question_id"], "title": r["question_title"], "platform": plat,
                             "contest_id": r["contest_id"], "contest_date": r["contest_date"],
                             "difficulty": r["difficulty"], "starter_code": r["starter_code"],
                             "fn_name": md.get("func_name"), "metadata": md,
                             "public_tests": json.loads(r["public_test_cases"]),
                             "file": fn, "line": li, "row_index": g},
                    "source": SOURCE,
                }
                cands.setdefault(plat, []).append((row, check([r["question_content"]], FAMILY, [r["question_id"], g])))
                g += 1

    def hydrate(rows):
        want = {(r["meta"]["file"], r["meta"]["line"]): r for r in rows}
        for fn, p in zip(FILES, paths):
            with open(p, encoding="utf-8") as fh:
                for li, line in enumerate(fh):
                    r = want.get((fn, li))
                    if r is not None:
                        src = json.loads(line)
                        assert src["question_id"] == r["meta"]["question_id"]
                        r["meta"]["private_tests"] = decode_private(src["private_test_cases"])
        return rows

    finish(BENCH, FAMILY, SOURCE, cands, hydrate=hydrate, extra={
        "files": FILES,
        "native_id": "question_id, and row index in the concatenated release (test.jsonl..test6.jsonl)",
        "dropped_modality": 0,
        "private_tests": "base64->zlib->restricted unpickle (no globals)->json; stored decoded in meta.private_tests",
    })


if __name__ == "__main__":
    main()
