"""Clean-room answer grader (v4), for the v4src schema (route_data/v4src/DATA_SPEC_v4.md).

grade(row, response_text) -> float in [0, 1], or None for task "code" (graded by lcb_grade_v4.py).
Dispatch is by row["task"]; a few benchmarks get a specialised matcher for their answer format
(math: LaTeX, finqa: rounding/percent/units, qanta: quiz-bowl answer lines, chessinstruct: moves/lists,
asdiv: multi-part answers). References = [row["answer"]] + row["aliases"]; the best match counts.
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

import json
import math

from prompts_v4 import LETTERS

CHRF_PASS = 40.0  # translate: sentence chrF (0-100) >= 40 counts as a pass; see CLEANROOM.md
F1_PASS = 0.5     # reading: token F1 >= 0.5 (or the content-recall rule) counts as a pass

# ---------------------------------------------------------------- extraction
_ANS_LINE = re.compile(r"^[\s>*#_`]*(?:final\s+)?answer\s*[*_]*\s*[:：]\s*(.*)$", re.I | re.M)


def boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...} (brace-matched)."""
    i = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if i < 0:
        return None
    i = text.index("{", i) + 1
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1] if depth == 0 else None


def answer_text(resp: str) -> str | None:
    """Text after the last 'Answer:' marker (next non-empty line if the marker line is empty)."""
    m = list(_ANS_LINE.finditer(resp or ""))
    if not m:
        return None
    s = m[-1].group(1).strip()
    if not s:
        rest = [l.strip() for l in resp[m[-1].end():].splitlines() if l.strip()]
        s = rest[0] if rest else ""
    s = s.strip().strip("*_`").strip()
    b = boxed(s)
    return b.strip() if b is not None else s


def final_text(resp: str) -> str:
    """Best guess at the final answer string: Answer line > \\boxed{} > last non-empty line."""
    a = answer_text(resp)
    if a:
        return a
    b = boxed(resp or "")
    if b is not None:
        return b.strip()
    lines = [l.strip() for l in (resp or "").splitlines() if l.strip()]
    return lines[-1] if lines else ""


def extract_letter(resp: str, n_opts: int, options: list[str] | None = None) -> int | None:
    """Return the chosen 0-based option index, or None."""
    valid = LETTERS[:n_opts]

    def letter(s):
        if s is None:
            return None
        s = re.sub(r"\\text\{(.*?)\}", r"\1", s).strip().strip("*_`").strip()
        m = re.match(r"^(?:option\s+)?\(?\[?([A-Z])\]?\)?(?:$|[\s.:,)\]])", s, re.I)
        if m and m.group(1).upper() in valid and (m.group(1).isupper() or len(s) <= 3):
            return valid.index(m.group(1).upper())
        if options:  # answer given as the option text itself
            norm = [_norm(o) for o in options]
            if _norm(s) in norm:
                return norm.index(_norm(s))
        return None

    for cand in (answer_text(resp), boxed(resp or "")):
        r = letter(cand)
        if r is not None:
            return r
    pats = [r"answer\s+is\s*:?\s*\(?([A-Z])\)?\b", r"(?:option|choice)\s+\(?([A-Z])\)?\s+is\s+correct",
            r"correct\s+(?:option|answer|choice)\s+is\s*:?\s*\(?([A-Z])\)?\b"]
    for p in pats:
        hits = [h for h in re.findall(p, resp or "", re.I) if h.upper() in valid and h.isupper()]
        if hits:
            return valid.index(hits[-1])
    lines = [l.strip() for l in (resp or "").splitlines() if l.strip()]
    return letter(lines[-1]) if lines else None


def extract_yesno(resp: str) -> str | None:
    a = (answer_text(resp) or "").lower()
    m = re.match(r"\W*(yes|no)\b", a)
    if m:
        return m.group(1)
    hits = re.findall(r"\b(yes|no)\b", (resp or "").lower())
    return hits[-1] if hits else None


# ---------------------------------------------------------------- numbers / LaTeX
_NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+")


def to_number(s: str) -> float | None:
    """First number in s; handles $ , % commas, a/b, \\frac{a}{b}."""
    if s is None:
        return None
    s = s.replace("\\!", "").replace("−", "-").replace("\\$", "").replace("$", "")
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    m = re.search(r"\(?\s*(-?[\d.,]+)\s*\)?\s*/\s*\(?\s*(-?[\d.,]+)\s*\)?", s)
    if m:
        try:
            return float(m.group(1).replace(",", "")) / float(m.group(2).replace(",", ""))
        except (ValueError, ZeroDivisionError):
            pass
    m = _NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def num_equal(pred: float | None, gold: float | None, rel: float = 1e-6) -> bool:
    return pred is not None and gold is not None and abs(pred - gold) <= rel * max(1.0, abs(gold))


def norm_latex(s: str) -> str:
    s = s or ""
    for a, b in [("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("^{\\circ}", ""), ("^\\circ", ""), ("\\circ", ""), ("\\$", ""), ("$", ""), ("\\!", ""),
                 ("\\,", ""), ("\\;", ""), ("\\%", ""), ("%", ""), ("\\displaystyle", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = re.sub(r"\s+", "", s).rstrip(".,")
    s = re.sub(r"^\(([A-Z])\)$", r"\1", s)  # "(A)" -> "A" (multiple-choice style MATH answers)
    s = re.sub(r"^[a-zA-Z]\w*=(?!.*=)", "", s)  # drop a leading 'x=' when only one '='
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)  # 1,000 -> 1000
    if re.fullmatch(r"-?\d+\.\d*", s):
        s = s.rstrip("0").rstrip(".")
    return s


# ---------------------------------------------------------------- free text
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().replace("_", " ")
    s = "".join(" " if c in string.punctuation else c for c in s)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return float(p == g)
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


STOP = set("a an the of to in on at for by with from and or but is are was were be been being it its this that "
           "these those he she they him her them his hers their i you we us our your my me as so because into about "
           "after before than then there which who whom what when where why how do does did has have had not no will "
           "would can could should may might also just very all any some each own one".split())


def _content(s: str) -> list[str]:
    """Normalised non-stopword tokens with a crude suffix stem (ing/ed/es/s/e)."""
    out = []
    for w in _norm(s).split():
        if w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s", "e"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[:-len(suf)]
                break
        out.append(w)
    return out


def content_recall_pass(pred: str, gold: str) -> bool:
    """NarrativeQA secondary rule: the answer covers >= half of the gold's content words (at least 2 of them when
    the gold has 2+), and is not more than 4x longer than the gold (min 3 words), so padding cannot game it."""
    p, g = _content(pred), _content(gold)
    if not p or not g or len(p) > 4 * max(len(g), 3):
        return False
    hit = sum((Counter(p) & Counter(g)).values())
    return hit >= max(min(2, len(g)), (len(g) + 1) // 2)


def _contains(hay: str, needle: str) -> bool:
    return bool(needle) and f" {needle} " in f" {hay} "


def refs(row: dict) -> list[str]:
    out = [str(row["answer"])] + [str(a) for a in row.get("aliases") or []]
    return list(dict.fromkeys(a for a in out if a.strip()))


def qanta_candidates(alias: str) -> list[str]:
    """Acceptable strings from a quiz-bowl answer line: '{Khmer} Empire [accept {Kambuja} Empire]' ->
    ['Khmer Empire', 'Khmer', 'Kambuja Empire']; 'prompt on' alternatives are not accepted."""
    a = re.sub(r"&lt;.*?&gt;|<.*?>", "", alias.replace("_", " "))
    out, extras = [], []
    for inner in re.findall(r"[\[(]([^\])]*)[\])]", a):
        for part in re.split(r";|,? or |,? accept ", inner):
            part = part.strip()
            if part and not re.match(r"(?i)(prompt|do not|don't|anti-?prompt|until|before|after)\b", part):
                extras.append(re.sub(r"(?i)^(or|accept)\s+", "", part))
    main = re.sub(r"[\[(][^\])]*[\])]", "", a)
    for part in re.split(r"\s+or\s+", main):
        if part.strip():
            out.append(part.replace("{", "").replace("}", ""))
            out += re.findall(r"\{([^}]*)\}", part)  # the required (underlined) part alone
    for e in extras:
        out.append(e.replace("{", "").replace("}", ""))
    return [c.strip().strip('"“”') for c in out if c.strip()]


def qanta_match(pred: str, gold: str) -> bool:
    g = _norm(re.sub(r"_?\([^)]*\)$", "", gold.strip()))  # drop wiki disambiguation "(play)"
    p = _norm(pred)
    if not p or not g:
        return False
    if p == g or _contains(p, g):
        return True
    gt, pt = g.split(), p.split()  # surname / trailing-part answer, e.g. "Fremont" for "John C. Fremont"
    return len(pt) < len(gt) and gt[-len(pt):] == pt and len(pt[-1]) >= 4


def chrf(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF
    return CHRF().sentence_score(hyp or "", [ref]).score


def latex_equal(pred: str, gold: str) -> bool:
    a, b = norm_latex(pred), norm_latex(gold)
    if a == b:
        return True
    if re.fullmatch(r"-?\d+(?:\.\d+)?|-?\d+/\d+|-?\\frac\{\d+\}\{\d+\}", b):
        return num_equal(to_number(a), to_number(b))
    return False


def finqa_match(pred: str, gold: str) -> bool:
    """Rounding to the gold's precision, percent <-> fraction, and thousand/million/billion unit choice."""
    g, p = to_number(gold), to_number(pred)
    if g is None or p is None:
        return False
    m = re.search(r"\.(\d+)", gold)
    dp = len(m.group(1)) if m else 0
    cands = [p]
    if ("%" in gold) != ("%" in pred):
        cands.append(p * 100 if "%" in gold else p / 100)
    if "%" not in gold and "%" not in pred and abs(p) <= 1 < abs(g) <= 100:  # "0.235" for a gold of "23.5"
        cands.append(p * 100)
    if any(abs(round(c, dp) - g) <= 10 ** -dp / 2 + 1e-9 or num_equal(c, g, 1e-3) for c in cands):
        return True
    # unit choice ("in millions"): amounts of magnitude >= 1 that agree to 0.1% after a power-of-1000 rescale
    return abs(g) >= 1 and any(abs(p * 1000.0 ** k - g) <= 1e-3 * abs(g) for k in (-3, -2, -1, 1, 2, 3))


def _flat(s: str) -> str:
    return re.sub(r"[\s\"'`\[\](){}]", "", (s or "").lower())


def chess_match(pred: str, gold: str, meta: dict) -> bool:
    if "{" in pred:  # model echoed the JSON output format
        try:
            obj = json.loads(pred[pred.index("{"): pred.rindex("}") + 1])
            if isinstance(obj, dict) and obj:
                pred = str(obj.get(meta.get("answer_key"), next(iter(obj.values()))))
        except ValueError:
            pass
    if _flat(pred) == _flat(gold):
        return True
    g, words = gold.strip().lower(), re.findall(r"[a-z]+", pred.lower())
    if g in ("white", "black"):
        return g in words and ({"white", "black"} - {g}).isdisjoint(words)
    if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", g):
        mv = re.findall(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", pred.lower())
        return bool(mv) and mv[0] == g
    return False


def multi_part_match(pred: str, gold: str) -> bool:
    """Answers like '17 (units); 9 (units)', '11:9', '11:45 p.m.', 'History'."""
    g = re.sub(r"\([^)]*\)", " ", gold)
    gnums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", g.replace(",", ""))]
    if gnums:
        pnums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))]
        return pnums == gnums
    return _norm(pred) == _norm(g)


def short_match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return False
    return p == g or (_contains(p, g) and len(p.split()) <= len(g.split()) + 3)


# ---------------------------------------------------------------- grade
def grade(row: dict, resp: str) -> float | None:
    task, bench = row["task"], row.get("benchmark", "")
    meta = row.get("meta") or {}
    meta = json.loads(meta) if isinstance(meta, str) else meta
    resp = resp or ""
    if task == "code":
        return None
    if task in ("mc", "binary"):
        return float(extract_letter(resp, len(row["options"]), row["options"]) == int(row["answer"]))
    pred, golds = final_text(resp), refs(row)
    if task == "translate":
        return float(max(chrf(pred, g) for g in golds) >= CHRF_PASS)
    if task == "reading":
        return float(any(token_f1(pred, g) >= F1_PASS or content_recall_pass(pred, g) for g in golds))
    if task == "numeric":
        if golds[0].strip().lower() in ("yes", "no"):
            return float(extract_yesno(resp) == golds[0].strip().lower())
        if bench == "finqa":
            return float(any(finqa_match(pred, g) for g in golds))
        p = to_number(pred)
        return float(any(num_equal(p, to_number(g)) for g in golds))
    # task == "short"
    if bench == "math":
        return float(any(latex_equal(pred, g) for g in golds))
    if bench == "qanta":
        return float(any(qanta_match(pred, c) for g in golds for c in qanta_candidates(g)))
    if bench == "chessinstruct":
        return float(any(chess_match(pred, g, meta) for g in golds))
    if bench == "asdiv":
        return float(any(multi_part_match(pred, g) for g in golds))
    return float(any(short_match(pred, g) for g in golds))
