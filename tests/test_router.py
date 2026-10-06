import numpy as np
import pytest

pytest.importorskip("sklearn")

from thriftllm.router import Router  # noqa: E402


def _data(seed=0, n=400, d=16):
    """Iki kume: 'kolay' kumede ucuz model de dogru, 'zor' kumede sadece pahali model."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(2, d)) * 3
    cluster = rng.integers(0, 2, n)
    emb = centers[cluster] + rng.normal(size=(n, d)) * 0.3
    scores = np.zeros((n, 2))
    scores[:, 0] = (cluster == 0).astype(float)  # ucuz: sadece kolay kume
    scores[:, 1] = 1.0  # pahali: hep dogru
    costs = np.column_stack([np.full(n, 0.0001), np.full(n, 0.002)])
    return emb, scores, costs, centers


def test_routes_easy_to_cheap_and_hard_to_strong():
    emb, scores, costs, centers = _data()
    r = Router.fit(["cheap", "strong"], emb, scores, costs, lam=50, k=20)
    picks = r.route_embeddings(centers)
    assert [r.models[j] for j in picks] == ["cheap", "strong"]


def test_lambda_zero_ignores_cost():
    emb, scores, costs, centers = _data()
    r = Router.fit(["cheap", "strong"], emb, scores, costs, lam=0, k=20)
    # kolay kumede ikisi de dogru; maliyet onemsizse daha yuksek olasilikli olan secilir
    assert r.models[r.route_embeddings(centers[1:])[0]] == "strong"


def test_expected_cost_comes_from_neighbours_not_list_price():
    emb, scores, costs, centers = _data()
    r = Router.fit(["cheap", "strong"], emb, scores, costs, lam=10, k=20)
    _, c = r.predict(centers)
    assert c[0, 0] == pytest.approx(0.0001, rel=1e-6)
    assert c[0, 1] == pytest.approx(0.002, rel=1e-6)


def test_decide_reports_every_candidate():
    emb, scores, costs, centers = _data()
    d = Router.fit(["cheap", "strong"], emb, scores, costs, lam=50, k=20).decide(centers[0])
    assert d.model == "cheap"
    assert set(d.p_correct) == set(d.expected_cost_usd) == set(d.utility) == {"cheap", "strong"}


def test_save_load_roundtrip(tmp_path):
    emb, scores, costs, centers = _data()
    r = Router.fit(["cheap", "strong"], emb, scores, costs, lam=50, k=20)
    path = tmp_path / "router.npz"
    r.save(path)
    r2 = Router.load(path)
    assert r2.models == r.models and r2.lam == r.lam and r2.k == r.k
    assert r2.route_embeddings(centers) == r.route_embeddings(centers)
