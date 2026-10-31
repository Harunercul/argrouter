"""Clean-room LiveCodeBench grader (v4). Standard library only (runs in python:3.11-slim).

SECURITY: this executes model-generated code. Run it ONLY inside an isolated container, e.g.
  docker run --rm --network none --memory 4g --cpus 2 -v $PWD:/w -w /w python:3.11-slim \
    python lcb_grade_v4.py --responses resp.jsonl --tests livecodebench_tests.jsonl.gz --out scores.jsonl

Input
  --responses  JSONL rows {"gidx" (or "id"), "response", ["model"]}; code = last ```python block in the response.
  --tests      one or more JSONL / JSONL.gz files, matched by id/gidx:
               * v4src data rows (route_data/v4src/data/livecodebench.jsonl) with meta.public_tests,
                 meta.private_tests, meta.fn_name;
               * split test files from split_lcb_tests.py ({"id", "fn_name", "public_tests", "private_tests"});
               * HF-release-style rows {"public_test_cases", "private_test_cases", "metadata"}, where test lists
                 are JSON strings or base64 -> zlib -> pickle(str) -> JSON. The pickle is loaded with a restricted
                 Unpickler that refuses every global, so only plain str/list/dict decode.
Output
  JSONL rows {"gidx", "model", "score": 0|1, "reason"}; score 1 only if every test passes.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import math
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor

TIMEOUT_S = 6.0
MEM_MB = 2048

FUNC_HARNESS = r'''
import sys, json
from typing import *
import collections, heapq, math, bisect, itertools, functools, string, re, operator
from collections import *
from heapq import *
from math import *
from bisect import *
from itertools import *
from functools import *
sys.setrecursionlimit(1 << 20)
_src, _func, _inp, _out = sys.argv[1:5]
_ns = dict(globals())
exec(compile(open(_src).read(), "solution.py", "exec"), _ns)
_args = [json.loads(l) for l in open(_inp).read().split("\n") if l.strip()]
_f = getattr(_ns["Solution"](), _func) if "Solution" in _ns else _ns[_func]
_res = _f(*_args)
with open(_out, "w") as fh:
    json.dump(_res, fh)
'''


# ------------------------------------------------------------------ decoding / extraction
class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def decode_tests(s: str) -> list[dict]:
    if not s:
        return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        obj = _NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load()
        return json.loads(obj) if isinstance(obj, str) else obj


def extract_code(text: str) -> str | None:
    """Last ```python (or ```py) block; else last fenced block of any kind."""
    blocks = re.findall(r"```[ \t]*(\w*)[^\n]*\n(.*?)```", text or "", re.S)
    py = [b for lang, b in blocks if lang.lower() in ("python", "python3", "py")]
    pick = py or [b for _, b in blocks]
    return pick[-1] if pick and pick[-1].strip() else None


# ------------------------------------------------------------------ comparison
def _tok_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)


def stdout_matches(got: str, expected: str) -> bool:
    """Per-line rstrip, ignore trailing blank lines; tokens may differ only in float formatting."""
    def lines(s):
        out = [l.rstrip() for l in s.replace("\r\n", "\n").split("\n")]
        while out and not out[-1]:
            out.pop()
        return out
    g, e = lines(got), lines(expected)
    if len(g) != len(e):
        return False
    for lg, le in zip(g, e):
        if lg != le:
            tg, te = lg.split(), le.split()
            if len(tg) != len(te) or not all(_tok_equal(x, y) for x, y in zip(tg, te)):
                return False
    return True


def json_matches(got, expected) -> bool:
    if isinstance(got, bool) or isinstance(expected, bool):
        return type(got) is type(expected) and got == expected
    if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(got, list) and isinstance(expected, list):
        return len(got) == len(expected) and all(json_matches(a, b) for a, b in zip(got, expected))
    if isinstance(got, dict) and isinstance(expected, dict):
        return got.keys() == expected.keys() and all(json_matches(got[k], expected[k]) for k in got)
    return got == expected


# ------------------------------------------------------------------ sandboxed run
# Limits are set in the child just before exec (thread-safe, unlike preexec_fn).
LAUNCHER = r'''
import os, sys
try:
    import resource
    mem, cpu = int(sys.argv[1]) * 1024 * 1024, int(float(sys.argv[2])) + 1
    for lim, val in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, cpu),
                     (resource.RLIMIT_FSIZE, 64 * 1024 * 1024), (resource.RLIMIT_CORE, 0)):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass
except ImportError:
    pass
os.execv(sys.argv[3], sys.argv[3:])
'''


def _run(cmd: list[str], stdin_path: str, cwd: str, timeout: float, mem_mb: int):
    """Returns (status, stdout, stderr) with status in ok / timeout / runtime_error."""
    with open(stdin_path, "rb") as fin:
        cmd = [sys.executable, "-I", "-c", LAUNCHER, str(mem_mb), str(timeout)] + cmd
        p = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
                             start_new_session=True,
                             env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0"})
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                p.kill()
            p.communicate()
            return "timeout", "", ""
    status = "ok" if p.returncode == 0 else "runtime_error"
    return status, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")[-300:]


def run_tests(code: str, tests: list[dict], func_name: str | None = None,
              timeout: float = TIMEOUT_S, mem_mb: int = MEM_MB) -> tuple[int, str]:
    """Run every test in a fresh subprocess; stop at the first failure. Returns (score, reason)."""
    if not tests:
        return 0, "no_tests"
    with tempfile.TemporaryDirectory(prefix="lcbv4_") as d:
        src, harness, inp, res = (os.path.join(d, n) for n in ("solution.py", "harness.py", "input.txt", "result.json"))
        with open(src, "w") as f:
            f.write(code)
        with open(harness, "w") as f:
            f.write(FUNC_HARNESS)
        for i, t in enumerate(tests):
            with open(inp, "w") as f:
                f.write(t["input"])
            functional = t.get("testtype") == "functional"
            if functional:
                if not func_name:
                    return 0, "no_func_name"
                if os.path.exists(res):
                    os.remove(res)
                cmd = [sys.executable, "-I", harness, src, func_name, inp, res]
            else:
                cmd = [sys.executable, "-I", src]
            status, out, err = _run(cmd, inp, d, timeout, mem_mb)
            if status != "ok":
                tail = err.strip().splitlines()[-1:] if err.strip() else []
                return 0, f"test {i}: {status}" + (f" ({tail[0][:120]})" if tail else "")
            if functional:
                try:
                    with open(res) as f:
                        got = json.load(f)
                    ok = json_matches(got, json.loads(t["output"]))
                except (OSError, ValueError):
                    ok = False
            else:
                ok = stdout_matches(out, t["output"])
            if not ok:
                return 0, f"test {i}: wrong_answer"
    return 1, f"pass ({len(tests)} tests)"


# ------------------------------------------------------------------ IO
def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def _row_key(row: dict) -> str | None:
    return row.get("id") or row.get("gidx") or row.get("Global Index")


def normalise_tests(row: dict) -> dict:
    """Any supported row -> {"tests": [...], "fn_name": str|None}. Supported: v4src data rows (tests in meta),
    split test-file rows ({"id", "fn_name", "public_tests", "private_tests"}), and the HF-release-style rows
    ({"public_test_cases", "private_test_cases" (JSON or base64/zlib/pickle), "metadata"})."""
    if isinstance(row.get("tests"), list):  # already normalised
        return {"tests": row["tests"], "fn_name": row.get("fn_name")}
    meta = row.get("meta") or {}
    meta = json.loads(meta) if isinstance(meta, str) else meta
    src = meta if "public_tests" in meta else row
    if "public_tests" in src:
        tests = list(src.get("public_tests") or []) + list(src.get("private_tests") or [])
        fn = src.get("fn_name") or meta.get("fn_name")
    else:
        tests = decode_tests(row.get("public_test_cases")) + decode_tests(row.get("private_test_cases"))
        md = row.get("metadata") or "{}"
        fn = (json.loads(md) if isinstance(md, str) else md).get("func_name")
    for t in tests:
        t.setdefault("testtype", "functional" if fn else "stdin")
    return {"tests": tests, "fn_name": fn}


def load_tests(paths, wanted: set[str] | None = None) -> dict[str, dict]:
    """Stream one or more JSONL(.gz) files; keep normalised tests for the wanted ids only."""
    out = {}
    for path in [paths] if isinstance(paths, str) else paths:
        with _open(path) as f:
            for line in f:
                if wanted is not None and not any(f'"{g}"' in line[:400] for g in wanted):
                    continue
                row = json.loads(line)
                k = _row_key(row)
                if wanted is None or k in wanted:
                    out[k] = normalise_tests(row)
    return out


def grade_row(resp: dict, entry: dict | None, model: str, timeout: float, mem_mb: int) -> dict:
    gidx = resp.get("gidx") or resp.get("id")
    base = {"gidx": gidx, "model": resp.get("model") or model}
    if entry is None:
        return {**base, "score": 0, "reason": "no_tests_for_gidx"}
    if "tests" not in entry:  # a raw row was passed
        try:
            entry = normalise_tests(entry)
        except Exception as e:  # noqa: BLE001
            return {**base, "score": 0, "reason": f"test_decode_error: {type(e).__name__}"}
    code = extract_code(resp.get("response") or "")
    if code is None:
        return {**base, "score": 0, "reason": "no_code"}
    score, reason = run_tests(code, entry["tests"], entry["fn_name"], timeout, mem_mb)
    return {**base, "score": score, "reason": reason}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--tests", required=True, nargs="+",
                    help="data/livecodebench.jsonl (tests inline) and/or livecodebench_tests.jsonl.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="", help="model name if rows have no 'model' field")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per test")
    ap.add_argument("--mem-mb", type=int, default=MEM_MB, help="address-space limit per test process")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    resps = [json.loads(l) for l in open(a.responses) if l.strip()]
    keys = {r.get("gidx") or r.get("id") for r in resps}
    tests = load_tests(a.tests, keys)
    with open(a.out, "w") as fo, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(grade_row, r, tests.get(r.get("gidx") or r.get("id")), a.model, a.timeout, a.mem_mb)
                for r in resps]
        n = s = 0
        for fu in futs:
            row = fu.result()
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            n, s = n + 1, s + row["score"]
    print(f"graded {n} responses, {s} passed", file=sys.stderr)


if __name__ == "__main__":
    main()
