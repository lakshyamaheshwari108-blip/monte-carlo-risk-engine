"""End-to-end checks of the market-risk pipeline (build_market_results)."""
import numpy as np
import pytest

from risk_engine import (
    MODEL_GBM,
    MODEL_HESTON,
    TRADING_DAYS,
    black_scholes,
    build_market_results,
    estimate_parameters,
    risk_metrics,
)


def run(prices, cfg, **overrides):
    cfg = dict(cfg); cfg.update(overrides)
    return build_market_results(prices, cfg)


def test_gbm_pipeline_structure(synthetic_prices, market_cfg):
    res = run(synthetic_prices, market_cfg)
    n = market_cfg["n_sims"]
    assert res["s0"] == pytest.approx(float(synthetic_prices.iloc[-1]))
    assert res["pnl"].shape == res["final_prices"].shape == (n,)
    assert res["heston"] is None and res["vol_summary"] is None and res["gbm_pnl"] is None
    assert res["fan"]["samples"].shape == (market_cfg["horizon"] + 1, 100)
    assert res["metrics"][99]["var"] >= res["metrics"][95]["var"]
    assert res["options"]["ref_name"] == "Black-Scholes"
    assert res["curve"]["strikes"].size == 31


def test_pnl_is_consistent_with_final_prices(synthetic_prices, market_cfg):
    res = run(synthetic_prices, market_cfg)
    expected = market_cfg["investment"] * (res["final_prices"] / res["s0"] - 1.0)
    assert np.allclose(res["pnl"], expected)
    assert res["metrics"][95]["var"] == pytest.approx(risk_metrics(res["pnl"])[95]["var"])


def test_results_are_reproducible_with_the_same_seed(synthetic_prices, market_cfg):
    a = run(synthetic_prices, market_cfg)
    b = run(synthetic_prices, market_cfg)
    c = run(synthetic_prices, market_cfg, seed=43)
    assert np.array_equal(a["pnl"], b["pnl"])
    assert not np.array_equal(a["pnl"], c["pnl"])


def test_gbm_option_prices_agree_with_black_scholes(synthetic_prices, market_cfg):
    res = run(synthetic_prices, market_cfg, n_sims=200_000, horizon=63)
    o = res["options"]
    assert abs(o["call"] - o["bs_call"]) < 4 * o["call_se"]
    assert abs(o["put"] - o["bs_put"]) < 4 * o["put_se"]
    # the same Black-Scholes number must come out of the independent function
    sigma_ann = res["sigma"] * np.sqrt(TRADING_DAYS)
    call, put = black_scholes(res["s0"], o["strike"], 0.04, sigma_ann, 63 / TRADING_DAYS)
    assert o["bs_call"] == pytest.approx(float(call)) and o["bs_put"] == pytest.approx(float(put))


def test_option_pipeline_satisfies_martingale_and_parity(synthetic_prices, market_cfg):
    res = run(synthetic_prices, market_cfg, n_sims=200_000, horizon=63)
    o = res["options"]
    assert abs(o["martingale_ratio"] - 1.0) < 4 * o["martingale_se"]
    parity_tolerance = 4 * np.hypot(o["call_se"], o["put_se"])
    assert abs(o["parity_gap"]) < parity_tolerance


def test_default_strike_is_at_the_money_and_custom_strike_is_used(synthetic_prices, market_cfg):
    atm = run(synthetic_prices, market_cfg)
    assert atm["options"]["strike"] == pytest.approx(atm["s0"])
    custom = run(synthetic_prices, market_cfg, strike=123.0)
    assert custom["options"]["strike"] == 123.0


def test_zero_drift_mode_has_no_expected_gain(synthetic_prices, market_cfg):
    res = run(synthetic_prices, market_cfg, drift_mode="Zero drift (conservative)",
              n_sims=200_000, horizon=63)
    final = res["final_prices"]
    se = final.std(ddof=1) / np.sqrt(final.size)
    assert abs(final.mean() - res["s0"]) < 4 * se


def test_historical_drift_mode_uses_the_estimated_drift(synthetic_prices, market_cfg):
    res = run(synthetic_prices, market_cfg, n_sims=200_000, horizon=63)
    mu_log, sigma, _ = estimate_parameters(synthetic_prices)
    expected = res["s0"] * np.exp((mu_log + 0.5 * sigma**2) * 63)
    se = res["final_prices"].std(ddof=1) / np.sqrt(res["final_prices"].size)
    assert abs(res["final_prices"].mean() - expected) < 4 * se


def test_heston_pipeline_structure_and_reference(synthetic_prices, market_cfg):
    res = run(synthetic_prices, market_cfg, model=MODEL_HESTON, n_sims=100_000)
    assert res["heston"] is not None and res["vol_summary"] is not None
    assert res["gbm_pnl"] is not None and res["gbm_metrics"] is not None
    # calibration ties v0 and theta to the historical variance
    assert res["heston"]["v0"] == pytest.approx(res["var_ann"])
    assert res["heston"]["theta"] == pytest.approx(res["var_ann"])
    o = res["options"]
    assert o["ref_name"] == "Heston (semi-analytic)"
    assert np.isfinite(o["ref_call"]) and np.isfinite(o["ref_put"])
    assert res["curve"]["ref_calls"] is not None and np.all(np.isfinite(res["curve"]["ref_calls"]))
    # Monte Carlo vs the semi-analytic benchmark (allow a small daily-step bias)
    assert abs(o["call"] - o["ref_call"]) < 4 * o["call_se"] + 0.01 * o["ref_call"]
    assert abs(o["put"] - o["ref_put"]) < 4 * o["put_se"] + 0.01 * o["ref_put"]


def test_heston_manual_parameters_are_respected(synthetic_prices, market_cfg):
    res = run(synthetic_prices, market_cfg, model=MODEL_HESTON, calibrate=False,
              theta=0.09, v0=0.06, kappa=1.5, xi=0.5, rho=-0.5, n_sims=20_000)
    assert res["heston"] == {"kappa": 1.5, "xi": 0.5, "rho": -0.5, "theta": 0.09, "v0": 0.06}


def test_heston_gives_fatter_left_tail_than_gbm_when_vol_of_vol_is_high(synthetic_prices, market_cfg):
    res = run(synthetic_prices, market_cfg, model=MODEL_HESTON, calibrate=True,
              xi=0.9, rho=-0.9, kappa=1.0, horizon=63, n_sims=100_000, drift_mode="Zero drift (conservative)")
    assert res["metrics"][99]["cvar"] > res["gbm_metrics"][99]["cvar"]


def test_pipeline_rejects_nonsense_configuration(synthetic_prices, market_cfg):
    with pytest.raises(ValueError):
        run(synthetic_prices, market_cfg, investment=0.0)
    with pytest.raises(ValueError):
        run(synthetic_prices, market_cfg, model=MODEL_HESTON, kappa=-1.0)
    flat = synthetic_prices.copy(); flat[:] = 100.0
    with pytest.raises(ValueError):
        run(flat, market_cfg)
