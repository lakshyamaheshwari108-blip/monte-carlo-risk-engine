"""Invalid inputs must fail loudly with ValueError rather than silently produce nonsense."""
import numpy as np
import pandas as pd
import pytest

from risk_engine import (
    black_scholes,
    cash_stats,
    estimate_parameters,
    heston_call_price,
    option_price_curve,
    price_european_options,
    simulate_cash_flows,
    simulate_gbm,
    simulate_gbm_terminal,
    simulate_heston,
    summarize_paths,
    validate_price_history,
    wilson_interval,
)

BAD_NUMBERS = [float("nan"), float("inf")]


# ---- GBM --------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(s0=0.0), dict(s0=-5.0), dict(s0=float("nan")),
    dict(sigma=-0.01), dict(sigma=float("inf")),
    dict(horizon=0), dict(horizon=-3), dict(horizon=2.5),
    dict(n_sims=0), dict(n_sims=-1), dict(mu=float("nan")),
])
def test_simulate_gbm_rejects_invalid_arguments(kwargs):
    args = dict(s0=100.0, mu=0.0, sigma=0.01, horizon=10, n_sims=10)
    args.update(kwargs)
    with pytest.raises(ValueError):
        simulate_gbm(**args)


@pytest.mark.parametrize("kwargs", [
    dict(years=0.0), dict(years=-1.0), dict(sigma_annual=-0.2), dict(s0=0.0), dict(n_sims=0),
])
def test_simulate_gbm_terminal_rejects_invalid_arguments(kwargs):
    args = dict(s0=100.0, mu_annual=0.03, sigma_annual=0.2, years=0.5, n_sims=10)
    args.update(kwargs)
    with pytest.raises(ValueError):
        simulate_gbm_terminal(**args)


# ---- Heston -----------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(kappa=0.0), dict(kappa=-1.0), dict(theta=0.0), dict(theta=-0.04),
    dict(xi=0.0), dict(xi=-0.3), dict(rho=1.5), dict(rho=-1.01), dict(v0=-0.01),
    dict(s0=0.0), dict(horizon=0), dict(n_sims=0), dict(steps_per_day=0),
    dict(rho=float("nan")),
])
def test_simulate_heston_rejects_invalid_arguments(kwargs):
    args = dict(s0=100.0, mu_annual=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=0.3,
                rho=-0.7, horizon=10, n_sims=10)
    args.update(kwargs)
    with pytest.raises(ValueError):
        simulate_heston(**args)


@pytest.mark.parametrize("kwargs", [
    dict(strike=0.0), dict(years=0.0), dict(v0=0.0), dict(kappa=0.0), dict(theta=-1.0),
    dict(xi=0.0), dict(rho=1.0), dict(rho=-1.0), dict(s0=-1.0),
])
def test_heston_reference_rejects_invalid_arguments(kwargs):
    args = dict(s0=100.0, strike=100.0, r=0.03, years=0.5, v0=0.04, kappa=2.0,
                theta=0.04, xi=0.3, rho=-0.7)
    args.update(kwargs)
    with pytest.raises(ValueError):
        heston_call_price(**args)


# ---- Option pricing ---------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(s0=0.0), dict(strike=0.0), dict(strike=-10.0), dict(sigma=0.0), dict(sigma=-0.2),
    dict(years=0.0), dict(years=-1.0), dict(r=float("nan")),
])
def test_black_scholes_rejects_invalid_arguments(kwargs):
    args = dict(s0=100.0, strike=100.0, r=0.03, sigma=0.2, years=1.0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        black_scholes(**args)


@pytest.mark.parametrize("terminal", [
    np.array([]), np.array([100.0]), np.array([100.0, -1.0]), np.array([100.0, np.nan]),
    np.ones((2, 2)),
])
def test_price_european_options_rejects_bad_terminal_prices(terminal):
    with pytest.raises(ValueError):
        price_european_options(terminal, 100.0, 0.03, 0.5)


@pytest.mark.parametrize("kwargs", [dict(strike=0.0), dict(years=0.0), dict(strike=float("nan"))])
def test_price_european_options_rejects_bad_scalars(kwargs):
    args = dict(terminal=np.array([90.0, 110.0]), strike=100.0, r=0.03, years=0.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        price_european_options(**args)


def test_option_price_curve_rejects_bad_strikes():
    with pytest.raises(ValueError):
        option_price_curve(np.array([100.0, 101.0]), np.array([0.0, 100.0]), 0.03, 0.5)
    with pytest.raises(ValueError):
        option_price_curve(np.array([100.0, 101.0]), np.array([]), 0.03, 0.5)


# ---- Market data ------------------------------------------------------------
def test_estimate_parameters_rejects_bad_prices():
    with pytest.raises(ValueError):
        estimate_parameters(pd.Series([100.0, 101.0]))
    with pytest.raises(ValueError):
        estimate_parameters(pd.Series([100.0, -1.0, 102.0, 103.0]))
    with pytest.raises(ValueError):
        estimate_parameters(pd.Series([100.0, np.nan, 102.0, 103.0]))


def test_validate_price_history_rejects_empty_missing_and_short_data():
    with pytest.raises(ValueError, match="No price data"):
        validate_price_history(pd.DataFrame(), "XYZ")
    with pytest.raises(ValueError, match="No price data"):
        validate_price_history(None, "XYZ")
    with pytest.raises(ValueError, match="No price data"):
        validate_price_history(pd.DataFrame({"Open": [1.0, 2.0]}), "XYZ")
    short = pd.DataFrame({"Close": np.linspace(10, 11, 20)})
    with pytest.raises(ValueError, match="data points"):
        validate_price_history(short, "XYZ")


def test_validate_price_history_cleans_the_series():
    index = pd.bdate_range("2024-01-01", periods=100, tz="America/New_York")
    close = np.linspace(100, 120, 100)
    close[5] = np.nan
    close[7] = -3.0
    close[9] = np.inf
    prices = validate_price_history(pd.DataFrame({"Close": close}, index=index), "XYZ")
    assert len(prices) == 97
    assert prices.index.tz is None
    assert (prices > 0).all() and np.isfinite(prices).all()


def test_summarize_paths_rejects_bad_shapes():
    with pytest.raises(ValueError):
        summarize_paths(np.array([1.0, 2.0, 3.0]))


# ---- Enterprise -------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(revenue=-1.0), dict(fixed=-1.0), dict(var_ratio=-0.1), dict(var_ratio=1.0),
    dict(var_ratio=45.0),            # percent passed by mistake instead of a fraction
    dict(vol=-0.1), dict(opening=-5.0), dict(shock=-1.0), dict(growth=-1.0),
    dict(debt=-1.0), dict(rate=-0.01), dict(months=0), dict(n_paths=0),
    dict(vol=float("nan")),
])
def test_simulate_cash_flows_rejects_invalid_arguments(kwargs):
    args = dict(revenue=100_000.0, fixed=40_000.0, var_ratio=0.45, vol=0.12, opening=40_000.0,
                shock=0.0, growth=0.0, debt=400_000.0, rate=0.06, months=12, n_paths=100)
    args.update(kwargs)
    with pytest.raises(ValueError):
        simulate_cash_flows(**args)


@pytest.mark.parametrize("cash", [
    np.array([1.0, 2.0]),                         # 1-D
    np.array([[100.0, 100.0]]),                   # only the opening row
    np.array([[100.0], [np.nan]]),
])
def test_cash_stats_rejects_invalid_arrays(cash):
    with pytest.raises(ValueError):
        cash_stats(cash)


@pytest.mark.parametrize("successes,n", [(1, 0), (-1, 10), (11, 10)])
def test_wilson_interval_rejects_invalid_counts(successes, n):
    with pytest.raises(ValueError):
        wilson_interval(successes, n)
