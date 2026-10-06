import numpy as np
import pytest

from thriftllm.router import Router


def _router(lam=50.0):
    """Elle kurulmus kucuk router: iki kume; ucuz model sadece kume 0'da dogru."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(2, 8)) * 3
    cluster = np.repeat([0, 1], 40)
    emb = centers[cluster] + rng.normal(size=(80, 8)) * 0.3
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    scores = np.column_stack([(cluster == 0).astype(float), np.ones(80)])
    costs = np.column_stack([np.full(80, 0.0001), np.full(80, 0.002)])
    return Router(
        models=["cheap", "strong"],
        coef=np.zeros((2, 8)),
        intercept=np.zeros(2),
        train_emb=emb,
        train_scores=scores,
        train_costs=costs,
        lam=lam,
        k=10,
        knn_weight=1.0,
    ), centers


def test_routes_easy_to_cheap_and_hard_to_strong():
    r, centers = _router()
    assert [r.models[j] for j in r.route_embeddings(centers)] == ["cheap", "strong"]


def test_expected_cost_comes_from_neighbours_not_list_price():
    r, centers = _router()
    _, c = r.predict(centers)
    assert c[0, 0] == pytest.approx(0.0001)
    assert c[0, 1] == pytest.approx(0.002)


def test_decide_reports_every_candidate():
    r, centers = _router()
    d = r.decide(centers[0])
    assert d.model == "cheap"
    assert set(d.p_correct) == set(d.expected_cost_usd) == set(d.utility) == {"cheap", "strong"}


def test_save_load_roundtrip(tmp_path):
    r, centers = _router()
    r.save(tmp_path / "router.npz")
    r2 = Router.load(tmp_path / "router.npz")
    assert r2.models == r.models and r2.lam == r.lam and r2.k == r.k
    assert r2.route_embeddings(centers) == r.route_embeddings(centers)
