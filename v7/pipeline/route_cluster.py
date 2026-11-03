"""argrouter cluster router (pre-registration part 9): inference only. The weights file is written by
fit_router_cluster.py.

The query text (last `max_chars` characters, as received) is embedded with `emb_model` through OpenRouter's
embeddings API, L2-normalised and assigned to the nearest of the k-means centroids. Each cluster stores, per pool
model, P = smoothed mean correctness and K = smoothed mean cost ($ per 1K queries) on our calibration set; the query
goes to argmax_j P_j - lam * K_j. Nothing in here looks at which benchmark or template a query comes from.

One embedding call per text, so each pick comes with that call's token count (RouterArena `router_usage`).

    r = RouterCluster("router_cluster.npz", api_key)
    picks, usage = r.pick_many(["What is the capital of France?"])
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import numpy as np

API = "https://openrouter.ai/api/v1/embeddings"


class RouterCluster:
    def __init__(self, weights: str | Path, api_key: str, concurrency: int = 8):
        z = np.load(weights)
        self.pool = [str(m) for m in z["pool"]]
        self.lam = float(z["lam"])
        self.emb_model = str(z["emb_model"])
        self.max_chars = int(z["max_chars"])
        self.centroids = z["centroids"].astype(np.float64)
        self.P, self.K = z["p"], z["k"]
        self.key, self.conc = api_key, concurrency

    async def _embed_all(self, texts: list[str]) -> tuple[np.ndarray, list[int]]:
        out, tokens = [None] * len(texts), [0] * len(texts)
        sem = asyncio.Semaphore(self.conc)

        async def one(client, i):
            async with sem:
                for attempt in range(8):
                    try:
                        r = await client.post(API, timeout=120, headers={"Authorization": f"Bearer {self.key}"},
                                              json={"model": self.emb_model,
                                                    "input": [texts[i][-self.max_chars:] or " "]})
                        d = r.json()
                        out[i] = d["data"][0]["embedding"]
                        tokens[i] = int((d.get("usage") or {}).get("prompt_tokens") or 0)
                        return
                    except Exception:  # retry with backoff
                        if attempt == 7:
                            raise
                        await asyncio.sleep(2 ** attempt)

        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(one(client, i) for i in range(len(texts))))
        return np.array(out, dtype=np.float64), tokens

    def pick_many(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """-> (pool model per text, embedding input tokens per text)."""
        E, tokens = asyncio.run(self._embed_all(texts))
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        d2 = (E ** 2).sum(1)[:, None] - 2 * E @ self.centroids.T + (self.centroids ** 2).sum(1)[None, :]
        cl = d2.argmin(1)
        best = (self.P - self.lam * self.K).argmax(0)  # pool index per cluster
        return [self.pool[j] for j in best[cl]], tokens
