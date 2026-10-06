"""RouterArena tahminlerini RouterArena koduyla puanla.

LiveCodeBench burada calistirilmaz; o izole konteynerde puanlanip dosyayla verilir.

cd ~/Desktop/RouterArena/llm_evaluation && ../.venv/bin/python .../ra_score.py [lcb_graded.jsonl]
"""

import json
import logging
import sys
from pathlib import Path

RA = Path.home() / "Desktop" / "RouterArena"
sys.path[:0] = [str(RA), str(RA / "llm_evaluation")]
logging.disable(logging.WARNING)

from evaluate_models import ModelEvaluator  # noqa: E402
from run import (  # noqa: E402
    compute_arena_score,
    evaluate_single_prediction,
    load_ground_truth_dataset,
)

preds = json.loads((RA / "router_inference/predictions/thriftllm.json").read_text())
gt = load_ground_truth_dataset("full")
ev = ModelEvaluator(cached_results_dir=str(RA / "cached_results"))
lcb_scores = {}
if len(sys.argv) > 1:
    with open(sys.argv[1]) as fh:
        for line in fh:
            r = json.loads(line)
            lcb_scores[r["gidx"]] = r["score"]

lcb_out = []
for p in preds:
    gidx = p["global index"]
    if gidx.startswith("LiveCodeBench"):
        gr = p["generated_result"]
        model = gr.get("model_used") or p["prediction"]
        p["cost"] = ev.calculate_inference_cost(
            model if ev.has_price(model) else p["prediction"], gr.get("token_usage", {})
        )
        p["accuracy"] = lcb_scores.get(gidx)
        lcb_out.append({"gidx": gidx, "response": gr.get("generated_answer", "")})
    else:
        assert evaluate_single_prediction(p, gt, ev), gidx

Path(RA / "router_inference/predictions/thriftllm_lcb_answers.jsonl").write_text(
    "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lcb_out)
)
scored = [p for p in preds if p["accuracy"] is not None]
acc = sum(p["accuracy"] for p in scored) / len(scored)
cost1k = sum(p["cost"] for p in preds) / len(preds) * 1000
arena = compute_arena_score(cost1k, acc) * 100
print(f"puanlanan {len(scored)}/{len(preds)}  acc={acc:.4f}  ${cost1k:.4f}/1k  arena={arena:.2f}")
nonlcb = [p for p in preds if not p["global index"].startswith("LiveCodeBench")]
a2 = sum(p["accuracy"] for p in nonlcb) / len(nonlcb)
print(f"LCB haric: n={len(nonlcb)} acc={a2:.4f}")
