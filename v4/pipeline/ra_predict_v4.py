"""Write RouterArena prediction files with the frozen v4 router (run once, after pre-registration part 3).

Same output as RouterArena's router_inference/generate_prediction_file.py: one entry per query, plus, for the
full split, "for_optimality" entries for every other pool model on the sub_10 queries. The router gets the
query's prompt text exactly as the dataset gives it ("prompt_formatted", else "prompt").

    python v4/pipeline/ra_predict_v4.py --weights router_v4.npz --ra ~/RouterArena --split full
    python v4/pipeline/ra_predict_v4.py --weights router_v4.npz --ra ~/RouterArena --split robustness
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from route_v4 import RouterV4  # noqa: E402

ROUTER_NAME = "argrouter"
# pool model -> RouterArena model name ("-reasoning-<effort>": same model with that OpenRouter reasoning effort)
RA_NAME = {
    "google/gemma-4-31b-it": "google/gemma-4-31b-it",
    "openai/gpt-6-luna": "openai/gpt-6-luna",
    "google/gemini-3.1-flash-lite@none": "google/gemini-3.1-flash-lite-reasoning-none",
    "anthropic/claude-haiku-5.5": "anthropic/claude-haiku-5.5",
    "deepseek/deepseek-v4-flash-0731": "deepseek/deepseek-v4-flash-0731",
    "google/gemini-3-flash-preview@none": "google/gemini-3-flash-preview-reasoning-none",
}
SPLITS = {"sub_10": "router_data_10.json", "full": "router_data.json", "robustness": "router_robustness.json"}


def entry(gidx, prompt, model, optimality):
    return {"global index": gidx, "prompt": prompt, "prediction": model, "generated_result": None,
            "cost": None, "accuracy": None, "for_optimality": optimality}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--ra", type=Path, required=True, help="RouterArena checkout")
    ap.add_argument("--split", choices=SPLITS, required=True)
    a = ap.parse_args()
    router = RouterV4(a.weights)
    pool = [RA_NAME[m] for m in router.pool]
    data = json.loads((a.ra / "dataset" / SPLITS[a.split]).read_text())
    rows = [(d.get("global index"), d.get("prompt_formatted") or d.get("prompt")) for d in data]
    rows = [(g, p) for g, p in rows if g and p]
    picks = [RA_NAME[m] for m in router.pick_many([p for _, p in rows])]
    out = [entry(g, p, m, False) for (g, p), m in zip(rows, picks, strict=True)]
    if a.split in ("sub_10", "full"):
        sub10 = {d.get("global index") for d in json.loads((a.ra / "dataset" / SPLITS["sub_10"]).read_text())}
        out += [entry(g, p, other, True) for (g, p), m in zip(rows, picks, strict=True) if g in sub10
                for other in pool if other != m]
    name = ROUTER_NAME + ("-robustness" if a.split == "robustness" else "")
    path = a.ra / "router_inference" / "predictions" / f"{name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    mix = {m: picks.count(m) for m in pool}
    print(f"wrote {len(out)} entries ({len(rows)} queries) to {path}; picks {mix}")
