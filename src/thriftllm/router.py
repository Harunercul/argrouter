"""Quality/cost router: pick the model that maximises P(correct) - lambda * expected cost.

Two signals per candidate model, both learned from a labelled response matrix
(questions x models, with graded scores and billed token counts):

* P(correct): average of a per-model logistic regression on the prompt embedding and a
  similarity-weighted vote of the k nearest training prompts.
* expected cost: similarity-weighted average of what each model actually spent on the k
  nearest prompts. Output length (including hidden reasoning tokens) differs by an order of
  magnitude between models on the same prompt, so list prices alone mis-rank them.

Inference is numpy-only. Training needs scikit-learn (``pip install thriftllm[router]``) and
embedding prompts needs fastembed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"


def _normalise(x: Array) -> Array:
    out: Array = x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)
    return out


@dataclass
class Decision:
    model: str
    p_correct: dict[str, float]
    expected_cost_usd: dict[str, float]
    utility: dict[str, float]


@dataclass
class Router:
    models: list[str]
    coef: Array  # (M, D) logistic regression weights
    intercept: Array  # (M,)
    train_emb: Array  # (N, D) normalised
    train_scores: Array  # (N, M) graded score in [0, 1]
    train_costs: Array  # (N, M) billed USD
    lam: float = 10.0  # USD-to-accuracy exchange rate per query
    k: int = 50
    knn_weight: float = 0.5
    embedder: str = DEFAULT_EMBEDDER

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(
        cls,
        models: list[str],
        emb: Array,
        scores: Array,
        costs: Array,
        *,
        lam: float = 10.0,
        k: int = 50,
        knn_weight: float = 0.5,
        reg: float = 1.0,
        embedder: str = DEFAULT_EMBEDDER,
    ) -> Router:
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

        emb = _normalise(np.asarray(emb, dtype=np.float64))
        coef = np.zeros((len(models), emb.shape[1]))
        intercept = np.zeros(len(models))
        for j in range(len(models)):
            y = (scores[:, j] >= 0.5).astype(int)
            if y.min() == y.max():  # constant label: encode as a saturated intercept
                intercept[j] = 20.0 if y[0] else -20.0
                continue
            clf = LogisticRegression(C=reg, max_iter=2000).fit(emb, y)
            coef[j], intercept[j] = clf.coef_[0], clf.intercept_[0]
        return cls(
            models,
            coef,
            intercept,
            emb,
            np.asarray(scores, float),
            np.asarray(costs, float),
            lam=lam,
            k=k,
            knn_weight=knn_weight,
            embedder=embedder,
        )

    # ----------------------------------------------------------------- inference
    def _knn(self, emb: Array) -> tuple[Array, Array]:
        sim = emb @ self.train_emb.T
        k = min(self.k, sim.shape[1])
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(sim, idx, axis=1).clip(min=0) ** 4
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
        ps = np.einsum("nk,nkm->nm", w, self.train_scores[idx])
        pc = np.einsum("nk,nkm->nm", w, self.train_costs[idx])
        return ps, pc

    def predict(self, emb: Array) -> tuple[Array, Array]:
        """(P(correct), expected USD) for each row of `emb`, shape (n, M)."""
        emb = _normalise(np.atleast_2d(np.asarray(emb, dtype=np.float64)))
        lr = 1.0 / (1.0 + np.exp(-(emb @ self.coef.T + self.intercept)))
        knn_p, cost = self._knn(emb)
        p = (1 - self.knn_weight) * lr + self.knn_weight * knn_p
        return p, cost

    def route_embeddings(self, emb: Array) -> list[int]:
        p, c = self.predict(emb)
        return list((p - self.lam * c).argmax(axis=1))

    def decide(self, emb: Array) -> Decision:
        p, c = self.predict(emb)
        u = p[0] - self.lam * c[0]
        j = int(u.argmax())
        return Decision(
            model=self.models[j],
            p_correct=dict(zip(self.models, p[0].round(4).tolist(), strict=True)),
            expected_cost_usd=dict(zip(self.models, c[0].tolist(), strict=True)),
            utility=dict(zip(self.models, u.round(6).tolist(), strict=True)),
        )

    def route(self, prompts: list[str]) -> list[str]:
        """Embed prompts with the router's embedder and return the chosen model per prompt."""
        return [self.models[j] for j in self.route_embeddings(embed(prompts, self.embedder))]

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            models=np.array(self.models),
            coef=self.coef.astype(np.float32),
            intercept=self.intercept,
            train_emb=self.train_emb.astype(np.float16),
            train_scores=self.train_scores.astype(np.float16),
            train_costs=self.train_costs.astype(np.float32),
            meta=np.array([self.lam, self.k, self.knn_weight]),
            embedder=np.array(self.embedder),
        )

    @classmethod
    def load(cls, path: str | Path) -> Router:
        z: Any = np.load(path, allow_pickle=False)
        lam, k, kw = z["meta"].tolist()
        return cls(
            models=z["models"].tolist(),
            coef=z["coef"].astype(np.float64),
            intercept=z["intercept"].astype(np.float64),
            train_emb=z["train_emb"].astype(np.float64),
            train_scores=z["train_scores"].astype(np.float64),
            train_costs=z["train_costs"].astype(np.float64),
            lam=lam,
            k=int(k),
            knn_weight=kw,
            embedder=str(z["embedder"]),
        )


def embed(texts: list[str], model: str = DEFAULT_EMBEDDER) -> Array:
    """Embed prompts locally (no API call). Requires `fastembed`."""
    from fastembed import TextEmbedding

    enc = TextEmbedding(model)
    return _normalise(np.array(list(enc.embed(texts)), dtype=np.float64))
