"""pytest for the clean-room v4 renderer/grader on the v4src schema.
Real rows from route_data/v4src/data, hand-written synthetic responses (correct / wrong / messy-correct)."""
import glob
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from grade_v4 import grade  # noqa: E402
from prompts_v4 import LETTERS, build_messages, template_key  # noqa: E402

DATA = Path.home() / "Desktop/thriftllm-lab/route_data/v4src/data"
ROWS = {}
for f in sorted(glob.glob(str(DATA / "*.jsonl"))):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        ROWS[r["id"]] = r

R = "Let me think step by step.\n"
CASES = [
    # (id, response, expected)
    # --- aime (numeric)
    ("aime/aime/2010-I-15", R + "Answer: 45", 1), ("aime/aime/2010-I-15", "Answer: 46", 0),
    ("aime/aime/2003-I-3", R + "The sum is $\\boxed{484}$.", 1), ("aime/aime/2003-I-3", "Answer: 048", 0),
    ("aime/aime/2005-II-13", "**Answer:** 418.", 1), ("aime/aime/2005-II-13", "Answer: 417", 0),
    # --- arcmmlu (mc, Chinese)
    ("arcmmlu/archive/dev:0", R + "Answer: C", 1), ("arcmmlu/archive/dev:0", "Answer: A", 0),
    ("arcmmlu/data_science/dev:4", "决策树用于预测。\n答案是 D。\nAnswer: (D)", 1),
    ("arcmmlu/information/dev:0", "Answer: 布拉德福定律", 1), ("arcmmlu/information/dev:0", "Answer: D", 0),
    # --- asdiv (numeric + multi-part short)
    ("asdiv/all/nluds-0626", R + "Answer: 66,018 people", 1), ("asdiv/all/nluds-0626", "Answer: 66017", 0),
    ("asdiv/all/nluds-1140", "22:18 simplifies.\nAnswer: 11:9", 1), ("asdiv/all/nluds-1140", "Answer: 9:11", 0),
    ("asdiv/all/nluds-1448", "Answer: 17 units by 9 units", 1), ("asdiv/all/nluds-1448", "Answer: 17 by 8", 0),
    # --- chessinstruct (short)
    ("chessinstruct/FIND_LAST_MOVE/7323", R + "Answer: f1d1", 1), ("chessinstruct/FIND_LAST_MOVE/7323", "Answer: f1f2", 0),
    ("chessinstruct/FIND_LAST_MOVE/7323", 'Answer: {"missing move": "f1d1"}', 1),
    ("chessinstruct/FIND_ADVANTAGED_PLAYER/50867", "White is up material.\nAnswer: White", 1),
    ("chessinstruct/FIND_ADVANTAGED_PLAYER/50867", "Answer: Black", 0),
    ("chessinstruct/FIND_FINAL_SCORE/21121", "It ends in a draw.\nAnswer: \"1/2-1/2\"", 1),
    ("chessinstruct/FIND_FINAL_SCORE/21121", "Answer: 1-0", 0),
    # --- ethics (binary + 2-way mc)
    ("ethics/virtue/6729", R + "Answer: A", 1), ("ethics/virtue/6729", "Answer: B", 0),
    ("ethics/utilitarianism/9919", "Losing the fish is worse.\nThe answer is (A).", 1),
    ("ethics/commonsense/1682", "Answer: not wrong", 1), ("ethics/commonsense/1682", "Answer: wrong", 0),
    # --- finqa (numeric: %, rounding to gold precision, fraction, units, yes/no)
    ("finqa/finqa/DRE/2008/page_29.pdf-3", R + "Answer: 3.54%", 1), ("finqa/finqa/DRE/2008/page_29.pdf-3", "Answer: 0.0354", 1),
    ("finqa/finqa/DRE/2008/page_29.pdf-3", "Answer: 4.5%", 0),
    ("finqa/finqa/BDX/2009/page_80.pdf-3", "Total = 92,280 (in thousands)\nAnswer: $92,280", 1),
    ("finqa/finqa/BDX/2009/page_80.pdf-3", "Answer: 92.28 million", 1), ("finqa/finqa/BDX/2009/page_80.pdf-3", "Answer: 9228", 0),
    ("finqa/finqa/ETR/2017/page_35.pdf-1", "Answer: 3.5", 1), ("finqa/finqa/ETR/2017/page_35.pdf-1", "Answer: -3.5%", 0),
    # --- geobench (mc + binary)
    ("geobench/apstudy/271", R + "Answer: D", 1), ("geobench/apstudy/271", "Answer: E", 0),
    ("geobench/npee_tf/111", "Structural water is part of the lattice, so the claim is false.\nAnswer: **B**", 1),
    ("geobench/npee_choice/77", "Answer: D. Cretaceous", 1), ("geobench/npee_choice/77", "Answer: C", 0),
    # --- gsm8k (numeric)
    ("gsm8k/main/7268", R + "Answer: 20", 1), ("gsm8k/main/7268", "Answer: 21", 0),
    ("gsm8k/main/4614", "She had $150.00 before.\nFinal Answer: $150", 1), ("gsm8k/main/1856", "Answer: 45 slices", 1),
    ("gsm8k/main/1856", "There are 15 slices each... total 30\nAnswer: 30", 0),
    # --- livecodebench: not graded here
    ("livecodebench/atcoder/abc360_d", "```python\nprint(5)\n```", None), ("livecodebench/leetcode/3573", "", None),
    ("livecodebench/atcoder/abc356_b", "anything", None),
    # --- math (short, LaTeX)
    ("math/counting_and_probability/644", R + "Answer: \\dfrac{216}{625}", 1),
    ("math/counting_and_probability/644", "Answer: 216/625", 1), ("math/counting_and_probability/644", "Answer: 0.3456", 1),
    ("math/counting_and_probability/644", "Answer: \\frac{216}{3125}", 0),
    ("math/precalculus/618", "So $(t,u) = \\boxed{\\left(\\frac43,-\\frac13\\right)}$.", 1),
    ("math/precalculus/618", "Answer: (4/3, 1/3)", 0),
    ("math/prealgebra/1100", "Answer: 82^{\\circ}", 1), ("math/prealgebra/1100", "Answer: 98", 0),
    # --- mathqa (mc)
    ("mathqa/gain/3235", R + "Answer: C", 1), ("mathqa/gain/3235", "Answer: A", 0),
    ("mathqa/general/221", "n^2 mod 23... \\boxed{D}", 1), ("mathqa/geometry/9997", "Answer: 315 : 26", 1),
    ("mathqa/geometry/9997", "Answer: B", 0),
    # --- medmcqa
    ("medmcqa/Anaesthesia/f98a8b2d-90a5-4d81-850c-1c16b374b06f", R + "Answer: C", 1),
    ("medmcqa/Anaesthesia/f98a8b2d-90a5-4d81-850c-1c16b374b06f", "Answer: B", 0),
    ("medmcqa/Biochemistry/250374e7-7d82-479a-a6b8-b7787380cb48", "Vitamin E is fat soluble.\nAnswer: Water soluble vitamin", 1),
    ("medmcqa/Anatomy/3822851e-d930-497c-b22e-528446edb572", "Answer: D", 1),
    ("medmcqa/Anatomy/3822851e-d930-497c-b22e-528446edb572", "Answer: A", 0),
    # --- mmlu
    ("mmlu/abstract_algebra/test:52", R + "Answer: A", 1), ("mmlu/anatomy/test:67", "Answer: B", 0),
    ("mmlu/astronomy/test:145", "Bolometric means all wavelengths. The correct answer is D.", 1),
    # --- mmlupro (10 options)
    ("mmlupro/biology/24", R + "Answer: H", 1), ("mmlupro/business/19", "Answer: A", 0),
    ("mmlupro/chemistry/26", "Stability increases up the group.\nAnswer: (J)", 1),
    # --- musictheorybench
    ("musictheorybench/knowledge/47", R + "Answer: D", 1), ("musictheorybench/reasoning/323", "Answer: A", 0),
    ("musictheorybench/knowledge/108", "Answer:\nC", 1),
    # --- narrativeqa (reading; max over two references)
    ("narrativeqa/gutenberg/train-11931", R + "Answer: While Cliges is away visiting King Arthur.", 1),
    ("narrativeqa/gutenberg/train-11931", "Answer: After the tournament at Oxford.", 0),
    ("narrativeqa/movie/train-20891", "Answer: He shows them the handbill from his first bounty.", 1),
    ("narrativeqa/gutenberg/train-21261", "They had no water left.\nAnswer: Thirst.", 1),
    ("narrativeqa/gutenberg/train-21261", "Answer: Scurvy", 0),
    # --- opentdb (mc + binary)
    ("opentdb/Animals/3791", R + "Answer: A", 1), ("opentdb/Art/2287", "Answer: A", 0),
    ("opentdb/Celebrities/649", "Gilliam was born in Minnesota.\nAnswer: A) True", 1),
    # --- pubmedqa
    ("pubmedqa/pqa_labeled/23792130", R + "Answer: A", 1), ("pubmedqa/pqa_labeled/10811329", "Answer: C", 0),
    ("pubmedqa/pqa_labeled/26085176", "answer: a", 1),
    # --- qanta (short; quiz-bowl answer lines as aliases)
    ("qanta/Current Events/153951", R + "Answer: Ted Cruz", 1), ("qanta/Current Events/153951", "Answer: Cruz", 1),
    ("qanta/Current Events/153951", "Answer: John Cornyn", 0),
    ("qanta/Fine Arts/20931", "Answer: piano quintets", 0), ("qanta/Fine Arts/20931", "Answer: the Piano Quintet", 1),
    ("qanta/Geography/18018", "**Answer: Boston, Massachusetts**", 1),
    # --- socialiqa
    ("socialiqa/socialiqa/17669", R + "Answer: A", 1), ("socialiqa/socialiqa/7338", "Answer: C", 0),
    ("socialiqa/socialiqa/4541", "Carson will be happy.\nAnswer: B) be happy", 1),
    # --- superglue (ReCoRD short, WSC binary, CB mc)
    ("superglue/ReCoRD/40021-61556", R + "Answer: World Cup", 1), ("superglue/ReCoRD/40021-61556", "Answer: England", 0),
    ("superglue/ReCoRD/40021-61556", "The placeholder refers to the tournament.\nAnswer: the World Cup", 1),
    ("superglue/WSC/28", "'It' is the refrigerator.\nAnswer: A", 1), ("superglue/WSC/28", "Answer: B", 0),
    ("superglue/CB/77", "Answer: A", 1),
    # --- wmt19 (translate, chrF >= 40, any direction)
    ("wmt19/de-fr/855", "Answer: Elle a souligné que l'accent « n'est pas mis sur un pays en particulier ».", 1),
    ("wmt19/de-fr/855", "Answer: Sie betonte, der Fokus liegt nicht auf einem bestimmten Land.", 0),
    ("wmt19/kk-en/478", "Here is the translation:\n\nAnswer:\nHowever, it would be a big mistake to ignore the role of "
     "banknotes and coins in the economy.", 1),
    ("wmt19/kk-en/478", "Answer: I cannot translate Kazakh.", 0),
    ("wmt19/en-zh/230", "Answer: 没有最高，只有更高，贵州茅台的股价在创新高的路上一直停不下来。", 1),
]


@pytest.mark.parametrize("rid,resp,expected", CASES, ids=[f"{c[0][:45]}-{i}" for i, c in enumerate(CASES)])
def test_grade(rid, resp, expected):
    assert rid in ROWS, f"row {rid} not in data (data rebuilt?)"
    assert grade(ROWS[rid], resp) == expected


def test_every_benchmark_covered():
    benches = {r["benchmark"] for r in ROWS.values()}
    for b in benches:
        rows = {c[0] for c in CASES if ROWS.get(c[0], {}).get("benchmark") == b}
        assert len(rows) >= 3, b


def test_gold_answer_always_passes():
    """'Answer: <gold>' (letter for mc/binary) must pass on every non-code row; a junk answer must fail."""
    bad, junk = [], []
    for r in ROWS.values():
        if r["task"] == "code":
            continue
        good = f"Answer: {LETTERS[r['answer']]}" if r["task"] in ("mc", "binary") else f"Answer: {r['answer']}"
        if grade(r, good) != 1:
            bad.append(r["id"])
        if grade(r, "Answer: zebra 987654321") != 0:
            junk.append(r["id"])
    assert not bad and not junk, (bad[:10], junk[:10])


def test_render_all_rows():
    for r in ROWS.values():
        text = build_messages(r)[0]["content"]
        key = template_key(r)
        if key.startswith("code"):
            assert "```python" in text
        else:
            assert '"Answer: ' in text
        if key == "mc":
            assert "\nA. " in text and "option letter" in text
        if r["task"] == "translate":
            assert " text into " in text and r["meta"]["tgt_lang_name"] in text
