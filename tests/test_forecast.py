import pytest

from thriftllm.forecast import OutputForecaster


def test_cold_start_uses_prior_and_says_so():
    f = OutputForecaster(prior_tokens=400)
    fc = f.predict("rag")
    assert fc.source == "prior" and fc.tokens == 400 and fc.samples == 0


def test_prior_is_capped_by_max_tokens():
    assert OutputForecaster(prior_tokens=400).predict("x", max_tokens=150).tokens == 150


def test_switches_to_observed_median_after_min_samples():
    f = OutputForecaster(min_samples=3)
    for n in (100, 120, 500):
        f.observe("rag", n)
    fc = f.predict("rag")
    assert fc.source == "observed" and fc.tokens == 120 and fc.samples == 3


def test_routes_are_independent():
    f = OutputForecaster(min_samples=1)
    f.observe("rag", 80)
    f.observe("gen", 900)
    assert f.predict("rag").tokens == 80
    assert f.predict("gen").tokens == 900


def test_window_forgets_old_observations():
    f = OutputForecaster(window=3, min_samples=1)
    for n in (10, 10, 10, 1000, 1000, 1000):
        f.observe("r", n)
    assert f.predict("r").tokens == 1000


def test_p90_reported_and_capped():
    f = OutputForecaster(min_samples=1)
    for n in range(1, 101):
        f.observe("r", n)
    fc = f.predict("r", max_tokens=50)
    assert fc.p90 == 50 and fc.tokens == 50


@pytest.mark.parametrize("kw", [{"prior_tokens": -1}, {"window": 0}, {"min_samples": 0}])
def test_rejects_invalid_config(kw):
    with pytest.raises(ValueError):
        OutputForecaster(**kw)


def test_rejects_negative_observation():
    with pytest.raises(ValueError):
        OutputForecaster().observe("r", -5)
