"""Cevap matrisini RouterArena'nin KENDI puanlayicisiyla puanla.

RouterArena ortaminda calistirilir (onun .venv'i ve kodu):
    cd ~/Desktop/RouterArena/llm_evaluation && \
      ../.venv/bin/python ~/Desktop/thriftllm/bench/routerarena/grade_matrix.py

route_data/matrix/*.jsonl -> route_data/graded/*.jsonl (score + RouterArena fiyatiyla maliyet).
LiveCodeBench burada puanlanmaz: model kodunu calistirmak gerekiyor, izole ortamda yapilacak.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

RA = Path.home() / "Desktop" / "RouterArena"
sys.path[:0] = [str(RA), str(RA / "llm_evaluation")]
logging.disable(logging.WARNING)

from eval_reasoning import get_scorers_for_dataset  # noqa: E402
from evaluate_models import ModelEvaluator, load_eval_config_for_dataset  # noqa: E402

TL = Path.home() / "Desktop" / "thriftllm" / "route_data"


def main() -> None:
    truth = {}
    for f in (TL / "train").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                truth[r["Global Index"]] = r["Answer"]
    ev = ModelEvaluator(cached_results_dir=str(RA / "cached_results"))
    out_dir = TL / "graded"
    out_dir.mkdir(exist_ok=True)
    scorer_cache: dict = {}
    for f in sorted((TL / "matrix").glob("*.jsonl")):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        seen, graded = set(), []
        for r in reversed(rows):  # tekrar varsa en son basarili cevap
            if "error" in r or r["gidx"] in seen:
                continue
            seen.add(r["gidx"])
            ds = ev.determine_dataset_from_global_index(r["gidx"])
            if ds == "LiveCodeBench":
                continue
            if ds not in scorer_cache:
                scorer_cache[ds] = get_scorers_for_dataset(ds, load_eval_config_for_dataset(ds))
            scorer, _ = scorer_cache[ds][0]
            score, _ = ev._evaluate_single_entry(r["response"] or "", truth[r["gidx"]], scorer, ds)
            r["score"] = float(score)
            graded.append(r)
        (out_dir / f.name).write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in graded)
        )
        acc = sum(x["score"] for x in graded) / max(1, len(graded))
        print(f"{f.stem:45} n={len(graded):4} acc={acc:.3f}")


if __name__ == "__main__":
    main()
