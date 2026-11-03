"""Write RouterArena prediction files with the frozen cluster router (pre-registration part 9; run once, after it).

Same output as RouterArena's router_inference/generate_prediction_file.py: one entry per query, plus, for the
full split, "for_optimality" entries for every other pool model on the sub_10 queries. The router gets the
query's prompt text exactly as the dataset gives it ("prompt_formatted", else "prompt"). Each query's embedding
call is recorded in the row's router_usage, so RouterArena bills it at the embedding model's price.

    python ra_predict_cluster.py --weights router_cluster.npz --ra ~/RouterArena --split full
    python ra_predict_cluster.py --weights router_cluster.npz --ra ~/RouterArena --split robustness
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from route_cluster import RouterCluster  # noqa: E402

ROUTER_NAME = "argrouter"
SPLITS = {"sub_10": "router_data_10.json", "full": "router_data.json", "robustness": "router_robustness.json"}


def entry(gidx, prompt, model, optimality, usage=None):
    e = {"global index": gidx, "prompt": prompt, "prediction": model, "generated_result": None,
         "cost": None, "accuracy": None, "for_optimality": optimality}
    if usage is not None:
        e["router_usage"] = [usage]
    return e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--ra", type=Path, required=True, help="RouterArena checkout")
    ap.add_argument("--split", choices=SPLITS, required=True)
    a = ap.parse_args()
    router = RouterCluster(a.weights, os.environ["OPENROUTER_API_KEY"])
    data = json.loads((a.ra / "dataset" / SPLITS[a.split]).read_text())
    rows = [(d.get("global index"), d.get("prompt_formatted") or d.get("prompt")) for d in data]
    rows = [(g, p) for g, p in rows if g and p]
    picks, tokens = router.pick_many([p for _, p in rows])
    usage = [{"model": router.emb_model, "input_tokens": t, "output_tokens": 0, "purpose": "embedding"}
             for t in tokens]
    out = [entry(g, p, m, False, u) for (g, p), m, u in zip(rows, picks, usage, strict=True)]
    if a.split in ("sub_10", "full"):
        sub10 = {d.get("global index") for d in json.loads((a.ra / "dataset" / SPLITS["sub_10"]).read_text())}
        out += [entry(g, p, other, True) for (g, p), m in zip(rows, picks, strict=True) if g in sub10
                for other in router.pool if other != m]
    name = ROUTER_NAME + ("-robustness" if a.split == "robustness" else "")
    path = a.ra / "router_inference" / "predictions" / f"{name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    mix = {m: picks.count(m) for m in router.pool}
    print(f"wrote {len(out)} entries ({len(rows)} queries, {sum(tokens)} embedding tokens) to {path}; picks {mix}")
