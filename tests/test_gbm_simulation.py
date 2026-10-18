"""Deterministic and statistical checks of the GBM engine and parameter estimation."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from risk_engine import (
    estimate_parameters,
    simulate_gbm,
    simulate_gbm_terminal,
    summarize_paths,
)


# ---- Deterministic cases ----------------------------------------------------
def test_zero_volatility_gives_the_exact_deterministic_path():
    mu, horizon = 0.0004, 50
    paths = simulate_gbm(100.0, mu, sigma=0.0, horizon=horizon, n_sims=7, seed=1)
    expected = 100.0 * np.exp(mu * np.arange(horizon + 1))
    assert paths.shape == (horizon + 1, 7)
    for column in paths.T:
        assert np.allclose(column, expected, rtol=1e-12)


def test_zero_volatility_terminal_sampler_is_exact():
    terminal = simulate_gbm_terminal(100.0, 0.05, 0.0, 2.0, 10, seed=1)
    assert np.allclose(terminal, 100.0 * np.exp(0.05 * 2.0))


def test_paths_are_positive_and_start_at_s0():
    paths = simulate_gbm(250.0, 0.0003, 0.03, 100, 2_000, seed=2)
    assert np.all(paths > 0)
    assert np.all(paths[0] == 250.0)


def test_same_seed_reproduces_and_different_seed_differs():
    a = simulate_gbm(100.0, 0.0, 0.01, 30, 100, seed=5)
    b = simulate_gbm(100.0, 0.0, 0.01, 30, 100, seed=5)
    c = simulate_gbm(100.0, 0.0, 0.01, 30, 100, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---- Statistical properties -------------------------------------------------
def test_expected_terminal_price_equals_s0_times_exp_mu_t():
    mu, sigma, horizon, n = 0.0005, 0.02, 63, 200_000
    paths = simulate_gbm(100.0, mu, sigma, horizon, n, seed=3)
    terminal = paths[-1]
    se = terminal.std(ddof=1) / np.sqrt(n)
    assert abs(terminal.mean() - 100.0 * np.exp(mu * horizon)) < 4 * se


def test_log_return_mean_and_variance_match_theory():
    mu, sigma, horizon, n = 0.0005, 0.02, 63, 200_000
    log_ret = np.log(simulate_gbm(100.0, mu, sigma, horizon, n, seed=4)[-1] / 100.0)
    assert log_ret.mean() == pytest.approx((mu - 0.5 * sigma**2) * horizon, abs=0.002)
    assert log_ret.var(ddof=1) == pytest.approx(sigma**2 * horizon, rel=0.02)


def test_path_and_exact_samplers_have_the_same_distribution():
    sigma_daily, horizon = 0.015, 63
    paths = simulate_gbm(100.0, 0.0003, sigma_daily, horizon, 50_000, seed=7)[-1]
    exact = simulate_gbm_terminal(100.0, 0.0003 * 252, sigma_daily * np.sqrt(252),
                                  horizon / 252, 50_000, seed=8)
    assert ks_2samp(paths, exact).pvalue > 0.001


def test_estimate_parameters_recovers_known_drift_and_volatility():
    mu_log, sigma = 0.0004, 0.012
    rng = np.random.default_rng(11)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(mu_log, sigma, 200_000))))
    est_mu, est_sigma, log_returns = estimate_parameters(prices)
    assert est_sigma == pytest.approx(sigma, rel=0.01)
    assert est_mu == pytest.approx(mu_log, abs=5 * sigma / np.sqrt(200_000))
    assert len(log_returns) == len(prices) - 1


def test_summarize_paths_returns_ordered_percentiles():
    paths = simulate_gbm(100.0, 0.0, 0.02, 40, 5_000, seed=9)
    s = summarize_paths(paths, n_samples=25)
    assert s["samples"].shape == (41, 25)
    assert np.all(s["p5"] <= s["p50"]) and np.all(s["p50"] <= s["p95"])
    assert np.array_equal(s["days"], np.arange(41))
