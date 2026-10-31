"""argrouter v4 router: inference only. The weights file is written by fit_router_v4.py.

For each pool model j the router estimates, from the query text alone,
  P_j = P(model j answers correctly)  logistic regression on a bge-small embedding
  K_j = expected cost in $ per 1K queries  exp(ridge on [embedding, log text length] + residual variance / 2)
and picks argmax_j P_j - lam * K_j. The text is embedded as received (last `max_chars` characters); nothing in
here looks at which benchmark or template a query comes from.

    r = RouterV4("router_v4.npz")
    r.pick("What is the capital of France?")  # -> "openai/gpt-6-luna"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RouterV4:
    def __init__(self, weights: str | Path):
        z = np.load(weights)
        self.pool = [str(m) for m in z["pool"]]
        self.lam = float(z["lam"])
        self.emb_model = str(z["emb_model"])
        self.max_chars = int(z["max_chars"])
        self.lr_coef, self.lr_intercept = z["lr_coef"], z["lr_intercept"]
        self.cost_coef, self.cost_intercept, self.cost_s2 = z["cost_coef"], z["cost_intercept"], z["cost_s2"]
        self._embedder = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(self.emb_model)
        return np.array(list(self._embedder.embed([t[-self.max_chars:] for t in texts])))

    def scores(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(P, K), each pool x len(texts)."""
        E = self._embed(texts)
        L = (np.log1p([len(t) for t in texts]) / 10)[:, None]
        P = 1 / (1 + np.exp(-(self.lr_coef @ E.T + self.lr_intercept[:, None])))
        EL = np.hstack([E, L])
        K = np.exp(self.cost_coef @ EL.T + self.cost_intercept[:, None] + self.cost_s2[:, None] / 2)
        return P, K

    def pick_many(self, texts: list[str]) -> list[str]:
        P, K = self.scores(texts)
        return [self.pool[j] for j in (P - self.lam * K).argmax(0)]

    def pick(self, text: str) -> str:
        return self.pick_many([text])[0]
