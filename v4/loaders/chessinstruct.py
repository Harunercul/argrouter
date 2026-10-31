"""ChessInstruct loader (Thytu/ChessInstruct, train.csv/test.csv read directly).

Card: columns task (instruction), input (JSON string: moves in UCI, sometimes FEN/result), expected_output (JSON string
with one key), KIND (task type). Subsets: the 6 KINDs. Question = task + "\\n\\n" + input (both verbatim); answer = the
single value of expected_output (task "short"). Native id = row index within the split file. Hashed text: input.

Gold validation with python-chess (results per item in meta.check, totals in the manifest). Generator code checked in
the dataset's own repo (github.com/Thytu/StockLLM @b98c5e8, src/tasks/*):
- FIND_NEXT_BEST_MOVE: the gold is Stockfish's move for FENs[cut], but moves_to_FENs() omits the start position, so
  that FEN is one ply AFTER the shown move list. The gold is therefore never legal for the side to move; it is a move
  for the opponent, in a position after an unseen move.
- MLM_ON_MOVES: the input shows the unmasked move list (no "?" anywhere); the gold "missing moves" are a subsequence of
  the shown list.
- SORT_FENS: the input FENs are already in the gold order.
- FIND_LAST_MOVE: replayed; the gold move is checked for legality and for the final result.
- FIND_FINAL_SCORE: replayed; when the final position is terminal the outcome is compared with the gold.
Validity rule (DATA_SPEC_v4.md, Amendment 1): an item is dropped before exclusion/allocation when its check flag is
not in VALID (gold fails the mechanical check, or the answer is visible in / identical to the input); the budget is
refilled by the spec round-robin over the remaining task types. Nothing is repaired.
"""
import collections
import json

import chess
import pandas as pd

from loadutil import hf_file, run

REPO, REV = "Thytu/ChessInstruct", "dcfd1a8075dffd498fa190916fe28b1e0f8300e9"
VALID = {"legal_for_side_to_move", "legal_result_agrees", "legal_nonterminal", "terminal_agrees",
         "nonterminal_unverifiable", "masked", "shuffled", "history_legal"}
VALIDITY_RULE = ("Items whose released gold fails a mechanical python-chess check against the generator's own "
                 "semantics (best move not legal for the side to move; last move illegal or contradicting the "
                 "result; final score contradicting a terminal position; illegal move history), or whose answer is "
                 "visible in the input (MLM_ON_MOVES without '?' masks; SORT_FENS inputs already in gold order), are "
                 "dropped before exclusion; the 200 budget is refilled by the spec round-robin (seed 20261031) over "
                 "the remaining valid items/task types.")


def replay(moves):
    b = chess.Board()
    for m in moves:
        try:
            mv = chess.Move.from_uci(m)
        except (ValueError, TypeError):
            return None
        if mv not in b.legal_moves:
            return None
        b.push(mv)
    return b


def as_list(s):
    return [t.strip() for t in str(s).strip().strip("[]").split(",") if t.strip()]


def check(kind, inp, gold):
    moves = inp.get("moves", [])
    if kind == "FIND_NEXT_BEST_MOVE":
        b = replay(moves)
        if b is None:
            return "history_illegal"
        if gold == "chess mate":
            return "gold_chess_mate"
        try:
            mv = chess.Move.from_uci(gold)
        except ValueError:
            return "gold_unparseable"
        if mv in b.legal_moves:
            return "legal_for_side_to_move"
        if not b.is_check():
            b.push(chess.Move.null())
            if mv in b.legal_moves:
                return "legal_only_for_opponent"
        return "illegal_in_shown_position"
    if kind == "FIND_LAST_MOVE":
        if not moves or moves[-1] != "?":
            return "no_placeholder"
        b = replay(moves[:-1])
        if b is None:
            return "history_illegal"
        try:
            mv = chess.Move.from_uci(gold)
        except ValueError:
            return "gold_unparseable"
        if mv not in b.legal_moves:
            return "gold_illegal"
        b.push(mv)
        o = b.outcome(claim_draw=True)
        if o is None:
            return "legal_nonterminal"
        return "legal_result_agrees" if o.result() == inp.get("result") else "legal_result_disagrees"
    if kind == "FIND_FINAL_SCORE":
        b = replay(moves)
        if b is None:
            return "history_illegal"
        o = b.outcome(claim_draw=True)
        if o is None:
            return "nonterminal_unverifiable"
        return "terminal_agrees" if o.result() == gold else "terminal_disagrees"
    if kind == "MLM_ON_MOVES":
        g = as_list(gold)
        it = iter(moves)
        sub = all(x in it for x in g)
        if "?" in moves:
            return "masked"
        return "unmasked_gold_in_input" if sub else "unmasked"
    if kind == "SORT_FENS":
        fens, g = inp.get("FENs", []), as_list(gold)
        if sorted(fens) != sorted(g):
            return "gold_not_permutation"
        return "input_already_sorted" if fens == g else "shuffled"
    if kind == "FIND_ADVANTAGED_PLAYER":
        return "history_legal" if replay(moves) is not None else "history_illegal"
    return "unchecked"


def main():
    cands, stats = [], collections.defaultdict(collections.Counter)
    dropped = collections.defaultdict(collections.Counter)
    for split in ["train", "test"]:
        df = pd.read_csv(hf_file(REPO, f"{split}.csv", REV, "chessinstruct"), dtype=str, keep_default_na=False)
        for i, r in enumerate(df.to_dict("records")):
            inp = json.loads(r["input"])
            out = json.loads(r["expected_output"])
            (key, gold), = out.items()
            gold = str(gold)
            flag = check(r["KIND"], inp, gold)
            stats[r["KIND"]][flag] += 1
            if flag not in VALID:
                dropped[r["KIND"]]["invalid:" + flag] += 1
                continue
            cands.append({"subset": r["KIND"], "split": split, "native_id": i, "hash_texts": [r["input"]],
                          "row": {"task": "short", "context": "", "question": r["task"].strip() + "\n\n" + r["input"],
                                  "options": [], "answer": gold, "aliases": [],
                                  "meta": {"kind": r["KIND"], "instruction": r["task"], "input": r["input"],
                                           "expected_output": r["expected_output"], "answer_key": key,
                                           "check": flag}}})
    dropped_counts = {k: sum(dropped[k].values()) for k in sorted(stats)}
    run("chessinstruct", "chessinstruct", f"{REPO}@{REV}", cands, dropped={k: dict(v) for k, v in dropped.items()},
        extra={"gold_validation": {k: dict(v) for k, v in sorted(stats.items())},
               "validity_rule": VALIDITY_RULE, "validity_dropped_per_task_type": dropped_counts,
               "valid_task_types": sorted(k for k in stats if dropped_counts[k] < sum(stats[k].values()))},
        notes=["subsets = the 6 KIND values; question = instruction + input JSON verbatim; answer = expected_output "
               "value",
               "FIND_NEXT_BEST_MOVE golds are one ply off (generator indexes FENs without the start position): never "
               "legal for the side to move in the shown position",
               "MLM_ON_MOVES inputs carry no '?' masks; SORT_FENS inputs are already in gold order",
               "Amendment 1 validity rule applied: invalid/degenerate items dropped, budget refilled by round-robin "
               "over the remaining task types; per-item meta.check, totals in gold_validation",
               "native id = row index within split; id-hash matches are positional"])


if __name__ == "__main__":
    main()
