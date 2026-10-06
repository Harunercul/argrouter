"""Puanlanmis matris uzerinde router'lari capraz dogrulamayla karsilastir (RouterArena puani).

python bench/routerarena/offline_eval.py --items route_data/pilot.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prompts import build_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RA = Path.home() / "Desktop" / "RouterArena"


def arena_score(cost_per_1k: float, acc: float, beta=0.1, c_max=200, c_min=0.0044) -> float:
    cost_per_1k = min(max(cost_per_1k, c_min), c_max)
    c = (math.log2(c_max) - math.log2(cost_per_1k)) / (math.log2(c_max) - math.log2(c_min))
    return (1 + beta) * acc * c / (beta * acc + c) if acc > 0 else 0.0


def price_table(models: list[str]) -> dict[str, tuple[float, float]]:
    """$/token. Once RouterArena tablosu (resmi puanlama onu kullanir), yoksa OpenRouter listesi."""
    ra = json.loads((RA / "model_cost" / "model_cost.json").read_text())
    orm = {
        m["id"]: m["pricing"]
        for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    }
    out = {}
    for m in models:
        if m in ra:
            p = ra[m]
            out[m] = (
                p["input_token_price_per_million"] / 1e6,
                p["output_token_price_per_million"] / 1e6,
            )
        else:
            p = orm[m.split("@")[0]]
            out[m] = (float(p["prompt"]), float(p["completion"]))
    return out


def load(items_path: Path, models: list[str] | None):
    items = [json.loads(x) for x in items_path.read_text().splitlines() if x.strip()]
    graded = {}
    for f in (ROOT / "route_data" / "graded").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            graded[(r["model"], r["gidx"])] = r
    all_models = sorted({m for m, _ in graded})
    models = models or all_models
    # sadece tum modellerde puani olan sorular
    items = [it for it in items if all((m, it["Global Index"]) in graded for m in models)]
    S = np.array([[graded[(m, it["Global Index"])]["score"] for m in models] for it in items])
    IN = np.array(
        [[graded[(m, it["Global Index"])]["input_tokens"] or 0 for m in models] for it in items],
        float,
    )
    OUT = np.array(
        [[graded[(m, it["Global Index"])]["output_tokens"] or 0 for m in models] for it in items],
        float,
    )
    return items, models, S, IN, OUT


def embed(items: list[dict]) -> np.ndarray:
    cache = ROOT / "route_data" / "emb_cache.json"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [it for it in items if it["Global Index"] not in store]
    if missing:
        from fastembed import TextEmbedding

        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        for it, v in zip(missing, model.embed([build_prompt(it) for it in missing]), strict=True):
            store[it["Global Index"]] = [round(float(x), 5) for x in v]
        cache.write_text(json.dumps(store))
    E = np.array([store[it["Global Index"]] for it in items])
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def handcrafted(prompt: str) -> list[float]:
    """Prompt'tan ucuz ozellikler: uzunluk, secenek sayisi, gorev turu ipuclari."""
    import re

    n_opts = len(re.findall(r"^[A-Z]\. ", prompt, flags=re.M))
    flags = [
        "multiple-choice" in prompt,
        "Translate the following" in prompt,
        "executable Python function" in prompt,
        "answer the question based on its content" in prompt,
        "\\boxed" in prompt,
        "chess" in prompt.lower(),
        "$" in prompt,
    ]
    return [math.log1p(len(prompt)) / 10, min(n_opts, 10) / 10, *map(float, flags)]


def embed_v2(items: list[dict]) -> np.ndarray:
    """Bas + son embedding (uzun baglamda soru sonda kalir) + el yapimi ozellikler."""
    cache = ROOT / "route_data" / "emb2_cache.json"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [it for it in items if it["Global Index"] not in store]
    if missing:
        from fastembed import TextEmbedding

        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        prompts = [build_prompt(it) for it in missing]
        heads = list(model.embed([p[:700] for p in prompts]))
        tails = list(model.embed([p[-1500:] for p in prompts]))
        for it, h, t in zip(missing, heads, tails, strict=True):
            store[it["Global Index"]] = [round(float(x), 5) for x in [*h, *t]]
        cache.write_text(json.dumps(store))
    H = np.array([store[it["Global Index"]] for it in items])
    H = np.hstack(
        [
            H[:, :384] / np.linalg.norm(H[:, :384], axis=1, keepdims=True),
            H[:, 384:] / np.linalg.norm(H[:, 384:], axis=1, keepdims=True),
        ]
    ) / math.sqrt(2)
    F = np.array([handcrafted(build_prompt(it)) for it in items])
    return np.hstack([H, F])


def ra_weights(items: list[dict]) -> np.ndarray:
    """Soruyu RouterArena aile payina gore agirlikla (egitim seti ailelere esit dagilmiyor)."""
    import collections

    import pandas as pd

    ra_path = ROOT / "route_data" / "routerarena" / "full-00000-of-00001.parquet"
    ra = pd.read_parquet(ra_path)
    ra_fam = collections.Counter(n.split("_")[0] for n in ra["Dataset name"])
    fam = [it["Dataset name"].split("_")[0] for it in items]
    ours = collections.Counter(fam)
    return np.array([ra_fam.get(f, 0) / ours[f] for f in fam])


def lr_predict(E_tr, Y_tr, E_te, C: float = 1.0):
    """Model basina lojistik regresyon (puan>=0.5 dogru sayilir)."""
    from sklearn.linear_model import LogisticRegression

    out = np.zeros((len(E_te), Y_tr.shape[1]))
    for j in range(Y_tr.shape[1]):
        y = (Y_tr[:, j] >= 0.5).astype(int)
        if y.min() == y.max():
            out[:, j] = y[0]
            continue
        clf = LogisticRegression(C=C, max_iter=2000).fit(E_tr, y)
        out[:, j] = clf.predict_proba(E_te)[:, 1]
    return out


def knn_predict(E_tr, Y_tr, E_te, k: int):
    sim = E_te @ E_tr.T
    idx = np.argsort(-sim, axis=1)[:, :k]
    w = np.take_along_axis(sim, idx, axis=1).clip(min=0) ** 4  # yakin komsuya daha cok agirlik
    w = w / w.sum(axis=1, keepdims=True).clip(min=1e-9)
    return np.einsum("nk,nkm->nm", w, Y_tr[idx])


W = None  # RA aile agirliklari (main'de atanir)


def evaluate(choice: np.ndarray, S: np.ndarray, C: np.ndarray):
    n = len(choice)
    w = np.ones(n) if W is None else W
    acc = float(np.average(S[np.arange(n), choice], weights=w))
    cost1k = float(np.average(C[np.arange(n), choice], weights=w)) * 1000
    return acc, cost1k, arena_score(cost1k, acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ra-weighted", action="store_true")
    ap.add_argument("--emb", default="v1", choices=["v1", "v2"])
    a = ap.parse_args()

    items, models, S, IN, OUT = load(a.items, a.models)
    P = price_table(models)
    C = IN * np.array([P[m][0] for m in models]) + OUT * np.array([P[m][1] for m in models])
    E = embed_v2(items) if a.emb == "v2" else embed(items)
    n = len(items)
    global W
    W = ra_weights(items) if a.ra_weighted else None
    print(f"{n} soru x {len(models)} model\n")

    print("== Tek model ==")
    for j, m in enumerate(models):
        acc, c1k, sc = evaluate(np.full(n, j), S, C)
        print(f"  {m:38} acc={acc:.3f}  ${c1k:7.3f}/1k  arena={sc * 100:.2f}")
    acc, c1k, sc = evaluate(S.argmax(1), S, C)
    # oracle: dogru olanlarin en ucuzu
    big = np.where(S >= 0.999, C, np.inf)
    orc = np.where(np.isfinite(big.min(1)), big.argmin(1), S.argmax(1))
    oacc, oc1k, osc = evaluate(orc, S, C)
    print(f"\n  {'oracle':38} acc={oacc:.3f}  ${oc1k:7.3f}/1k  arena={osc * 100:.2f}")

    rng = np.random.default_rng(0)
    fold = rng.permutation(n) % a.folds
    lambdas = [0, 3, 10, 20, 30, 50, 100, 200]

    def cv(predict_s, predict_c, lam):
        choice = np.zeros(n, int)
        for f in range(a.folds):
            tr, te = fold != f, fold == f
            ps = predict_s(E[tr], S[tr], E[te])
            pc = predict_c(E[tr], C[tr], E[te])
            choice[te] = (ps - lam * pc).argmax(1)
        return evaluate(choice, S, C)

    variants = {
        "knn25": (lambda a_, b_, c_: knn_predict(a_, b_, c_, 25), 25),
        "knn50": (lambda a_, b_, c_: knn_predict(a_, b_, c_, 50), 50),
        "lr": (lambda a_, b_, c_: lr_predict(a_, b_, c_), 50),
        "lr+knn": (
            lambda a_, b_, c_: 0.5 * lr_predict(a_, b_, c_) + 0.5 * knn_predict(a_, b_, c_, 50),
            50,
        ),
    }
    print("\n== Router'lar (5-kat CV, RouterArena aile agirlikli) ==")
    for name, (ps, kc) in variants.items():
        best = None
        for lam in lambdas:
            acc, c1k, sc = cv(ps, lambda a_, b_, c_, k=kc: knn_predict(a_, b_, c_, k), lam)
            best = max(best or (sc, lam, acc, c1k), (sc, lam, acc, c1k))
            print(f"  {name:7} lam={lam:4}  acc={acc:.3f}  ${c1k:7.3f}/1k  arena={sc * 100:.2f}")
        print(f"  -> {name} en iyi: arena={best[0] * 100:.2f} (lam={best[1]})\n")


if __name__ == "__main__":
    main()
