"""pytest for lcb_grade_v4 using real LiveCodeBench tasks and hand-written solutions (our own code)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import lcb_grade_v4 as G  # noqa: E402

TESTS = Path.home() / "Desktop/thriftllm/route_data/lcb_tests.jsonl"
GIDS = ["LiveCodeBench_abc383_a", "LiveCodeBench_abc341_a", "LiveCodeBench_3026", "LiveCodeBench_3676"]


def wrap(code, prose="Here is my solution."):
    return f"{prose}\n\n```python\n{code.strip()}\n```\nThis runs in O(n)."


SOL = {
    ("LiveCodeBench_abc383_a", "correct"): """
n = int(input())
water, last = 0, 0
for _ in range(n):
    t, v = map(int, input().split())
    water = max(0, water - (t - last)) + v
    last = t
print(water)
""",
    ("LiveCodeBench_abc383_a", "wrong"): """
n = int(input())
print(sum(int(input().split()[1]) for _ in range(n)))
""",
    ("LiveCodeBench_abc341_a", "correct"): "print('10' * int(input()) + '1')\n",
    ("LiveCodeBench_abc341_a", "wrong"): "print('01' * int(input()) + '0')\n",
    ("LiveCodeBench_3026", "correct"): """
class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        m = min(n, target // 2)
        r = n - m
        return (m * (m + 1) // 2 + r * target + r * (r - 1) // 2) % (10**9 + 7)
""",
    ("LiveCodeBench_3026", "wrong"): """
class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        return n * (n + 1) // 2
""",
    ("LiveCodeBench_3676", "correct"): """
class Solution:
    def smallestNumber(self, n: int) -> int:
        return (1 << n.bit_length()) - 1
""",
    ("LiveCodeBench_3676", "wrong"): """
class Solution:
    def smallestNumber(self, n: int) -> int:
        return n
""",
}


@pytest.fixture(scope="module")
def tests():
    if not TESTS.exists():
        pytest.skip("lcb_tests.jsonl not available")
    return G.load_tests(str(TESTS), set(GIDS))


@pytest.mark.parametrize("gid", GIDS)
def test_correct_and_wrong(tests, gid):
    good = G.grade_row({"gidx": gid, "response": wrap(SOL[(gid, "correct")])}, tests[gid], "m", 6, 2048)
    bad = G.grade_row({"gidx": gid, "response": wrap(SOL[(gid, "wrong")])}, tests[gid], "m", 6, 2048)
    assert good["score"] == 1, good
    assert bad["score"] == 0 and "wrong_answer" in bad["reason"], bad


def test_messy_response_last_python_block(tests):
    gid = "LiveCodeBench_abc341_a"
    resp = ("First a draft:\n```python\nprint('draft')\n```\nFinal version:\n```Python\n"
            + SOL[(gid, "correct")] + "```\n")
    assert G.grade_row({"gidx": gid, "response": resp}, tests[gid], "m", 6, 2048)["score"] == 1


def test_failure_modes(tests):
    gid = "LiveCodeBench_abc341_a"
    r = G.grade_row({"gidx": gid, "response": "no code here"}, tests[gid], "m", 6, 2048)
    assert (r["score"], r["reason"]) == (0, "no_code")
    r = G.grade_row({"gidx": gid, "response": wrap("raise ValueError('boom')")}, tests[gid], "m", 6, 2048)
    assert r["score"] == 0 and "runtime_error" in r["reason"] and "boom" in r["reason"]
    r = G.grade_row({"gidx": gid, "response": wrap("while True:\n    pass")}, tests[gid], "m", 1, 2048)
    assert r["score"] == 0 and "timeout" in r["reason"]
    r = G.grade_row({"gidx": "missing", "response": wrap("print(1)")}, None, "m", 6, 2048)
    assert r["reason"] == "no_tests_for_gidx"


def test_output_comparison():
    assert G.stdout_matches("3  \n\n\n", "3")
    assert G.stdout_matches("0.3333333333\n", "0.333333333333333")
    assert not G.stdout_matches("0.334\n", "0.333333333333333")
    assert not G.stdout_matches("Yes\n", "yes")
    assert not G.stdout_matches("1 2\n", "1\n2")
    assert G.json_matches([1, 2.0000000001], [1, 2]) and not G.json_matches(True, 1)
    assert G.extract_code("```py\nx=1\n```") == "x=1\n"


def test_cli_end_to_end(tests, tmp_path):
    gid = "LiveCodeBench_3676"
    resp = tmp_path / "r.jsonl"
    resp.write_text(json.dumps({"gidx": gid, "response": wrap(SOL[(gid, "correct")])}) + "\n"
                    + json.dumps({"gidx": gid, "response": wrap(SOL[(gid, "wrong")]), "model": "x"}) + "\n")
    tfile = tmp_path / "t.jsonl"
    tfile.write_text(json.dumps({"gidx": gid, **tests[gid]}) + "\n")
    out = tmp_path / "o.jsonl"
    G.main(["--responses", str(resp), "--tests", str(tfile), "--out", str(out), "--model", "m"])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [(r["model"], r["score"]) for r in rows] == [("m", 1), ("x", 0)]


# ---------------------------------------------------------------- v4src schema (tests split into .jsonl.gz)
V4DATA = Path.home() / "Desktop/thriftllm-lab/route_data/v4src/data"
V4IDS = ["livecodebench/atcoder/abc389_b", "livecodebench/leetcode/3476"]
V4SOL = {
    ("livecodebench/atcoder/abc389_b", "correct"): """
x = int(input())
n, f = 1, 1
while f < x:
    n += 1
    f *= n
print(n)
""",
    ("livecodebench/atcoder/abc389_b", "wrong"): "import math\nprint(round(math.log2(int(input()))))\n",
    ("livecodebench/leetcode/3476", "correct"): """
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        return sum(1 for x in nums if x % 3)
""",
    ("livecodebench/leetcode/3476", "wrong"): """
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        return sum(x % 3 for x in nums)
""",
}


@pytest.fixture(scope="module")
def v4tests():
    gz = V4DATA / "livecodebench_tests.jsonl.gz"
    if not gz.exists():
        pytest.skip("v4src split tests not available")
    return G.load_tests([str(gz)], set(V4IDS))


@pytest.mark.parametrize("gid", V4IDS)
def test_v4_correct_and_wrong(v4tests, gid):
    entry = v4tests[gid]
    assert len(entry["tests"]) > 10 and (entry["fn_name"] is None) == ("atcoder" in gid)
    good = G.grade_row({"id": gid, "response": wrap(V4SOL[(gid, "correct")])}, entry, "m", 6, 2048)
    bad = G.grade_row({"gidx": gid, "response": wrap(V4SOL[(gid, "wrong")])}, entry, "m", 6, 2048)
    assert good["score"] == 1 and good["gidx"] == gid, good
    assert bad["score"] == 0 and "wrong_answer" in bad["reason"], bad


def test_v4_small_rows_and_inline_rows_normalise_the_same(v4tests):
    small = {json.loads(l)["id"]: json.loads(l) for l in open(V4DATA / "livecodebench.jsonl")}
    assert len(small) == 200 and all("private_tests" not in r["meta"] for r in small.values())
    gid = "livecodebench/leetcode/3476"
    m = small[gid]["meta"]
    assert m["n_public_tests"] + m["n_private_tests"] == len(v4tests[gid]["tests"])
    inline = dict(small[gid], meta=dict(m, public_tests=v4tests[gid]["tests"][:m["n_public_tests"]],
                                        private_tests=v4tests[gid]["tests"][m["n_public_tests"]:]))
    assert G.normalise_tests(inline) == v4tests[gid]
