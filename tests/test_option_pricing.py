"""
Priority 1 - validate the option-pricing engine.

Under GBM the Black-Scholes formula is an exact analytic benchmark, so the Monte
Carlo pricer can be checked rigorously:

* prices must converge to Black-Scholes, with errors consistent with the Monte
  Carlo standard error (a z-score test) and shrinking like 1/sqrt(N);
* put-call parity must hold (exactly for Black-Scholes, statistically for Monte Carlo);
* the discounted stock price must be a martingale under the risk-neutral measure.
"""
import numpy as np
import pytest

from risk_engine import (
    black_scholes,
    gbm_convergence_study,
    option_price_curve,
    price_european_options,
    simulate_gbm,
    simulate_gbm_terminal,
)

S0, R, SIGMA, YEARS = 100.0, 0.04, 0.20, 0.5


# ---- Black-Scholes itself ---------------------------------------------------
def test_black_scholes_matches_textbook_value():
    # Hull-style benchmark: S=100, K=100, r=5%, sigma=20%, T=1 -> call 10.4506, put 5.5735
    call, put = black_scholes(100.0, 100.0, 0.05, 0.20, 1.0)
    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("strike", [60.0, 90.0, 100.0, 110.0, 150.0])
def test_black_scholes_put_call_parity_is_exact(strike):
    call, put = black_scholes(S0, strike, R, SIGMA, YEARS)
    assert call - put == pytest.approx(S0 - strike * np.exp(-R * YEARS), abs=1e-10)


def test_black_scholes_respects_arbitrage_bounds():
    strikes = np.linspace(50, 150, 41)
    call, put = black_scholes(S0, strikes, R, SIGMA, YEARS)
    disc_k = strikes * np.exp(-R * YEARS)
    assert np.all(call >= np.maximum(S0 - disc_k, 0) - 1e-12)
    assert np.all(call <= S0 + 1e-12)
    assert np.all(put >= np.maximum(disc_k - S0, 0) - 1e-12)
    assert np.all(put <= disc_k + 1e-12)


def test_black_scholes_monotonicity():
    strikes = np.linspace(60, 140, 30)
    call, put = black_scholes(S0, strikes, R, SIGMA, YEARS)
    assert np.all(np.diff(call) < 0)      # calls fall as the strike rises
    assert np.all(np.diff(put) > 0)       # puts rise as the strike rises
    # vega is positive: higher volatility -> higher price
    assert black_scholes(S0, 100, R, 0.30, YEARS)[0] > black_scholes(S0, 100, R, 0.20, YEARS)[0]


# ---- Monte Carlo convergence to Black-Scholes -------------------------------
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_mc_price_within_statistical_error_of_black_scholes(strike):
    n = 400_000
    terminal = simulate_gbm_terminal(S0, R, SIGMA, YEARS, n, seed=123)
    mc = price_european_options(terminal, strike, R, YEARS)
    bs_call, bs_put = black_scholes(S0, strike, R, SIGMA, YEARS)
    # 4 standard errors: a false failure has probability ~6e-5 per assertion
    assert abs(mc["call"] - bs_call) < 4 * mc["call_se"]
    assert abs(mc["put"] - bs_put) < 4 * mc["put_se"]


def test_mc_standard_error_shrinks_like_inverse_sqrt_n():
    small = price_european_options(simulate_gbm_terminal(S0, R, SIGMA, YEARS, 10_000, 1), 100, R, YEARS)
    large = price_european_options(simulate_gbm_terminal(S0, R, SIGMA, YEARS, 160_000, 2), 100, R, YEARS)
    ratio = small["call_se"] / large["call_se"]          # expected sqrt(16) = 4
    assert ratio == pytest.approx(4.0, rel=0.10)


def test_convergence_study_errors_are_consistent_with_standard_errors():
    df = gbm_convergence_study(path_counts=(2_000, 8_000, 32_000, 128_000, 512_000), seed=2024)
    assert (df["call_z"].abs() < 4).all()
    assert (df["put_z"].abs() < 4).all()
    # the standard error itself must fall steadily as paths are added
    assert df["call_se"].is_monotonic_decreasing
    # the biggest sample must be much closer to Black-Scholes than the smallest
    assert abs(df["call_error"].iloc[-1]) < 0.05
    assert df["call_se"].iloc[-1] < 0.1 * df["call_se"].iloc[0]


def test_mc_call_converges_across_replications():
    """Averaging independent replications removes noise: the mean should be unbiased."""
    bs_call = float(black_scholes(S0, 100, R, SIGMA, YEARS)[0])
    errors = []
    for seed in range(20):
        terminal = simulate_gbm_terminal(S0, R, SIGMA, YEARS, 50_000, seed)
        errors.append(price_european_options(terminal, 100, R, YEARS)["call"] - bs_call)
    errors = np.array(errors)
    standard_error_of_mean = errors.std(ddof=1) / np.sqrt(len(errors))
    assert abs(errors.mean()) < 4 * standard_error_of_mean


# ---- Put-call parity under Monte Carlo --------------------------------------
@pytest.mark.parametrize("strike", [85.0, 100.0, 115.0])
def test_mc_put_call_parity(strike):
    n = 400_000
    terminal = simulate_gbm_terminal(S0, R, SIGMA, YEARS, n, seed=77)
    mc = price_european_options(terminal, strike, R, YEARS)
    parity_gap = (mc["call"] - mc["put"]) - (S0 - strike * np.exp(-R * YEARS))
    # call - put = exp(-rT) * (mean(S_T) - K): its error is the error of the mean of S_T
    se = np.exp(-R * YEARS) * terminal.std(ddof=1) / np.sqrt(n)
    assert abs(parity_gap) < 4 * se


def test_parity_holds_exactly_for_the_same_payoff_sample():
    """Identity: max(S-K,0) - max(K-S,0) = S-K, so parity is exact path by path."""
    terminal = simulate_gbm_terminal(S0, R, SIGMA, YEARS, 5_000, seed=5)
    strike = 97.0
    disc = np.exp(-R * YEARS)
    lhs = (np.maximum(terminal - strike, 0) - np.maximum(strike - terminal, 0)) * disc
    assert np.allclose(lhs, (terminal - strike) * disc, atol=1e-12)


# ---- Discounted-stock martingale --------------------------------------------
@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.45])
def test_discounted_terminal_price_is_a_martingale(sigma):
    n = 400_000
    terminal = simulate_gbm_terminal(S0, R, sigma, YEARS, n, seed=99)
    discounted = np.exp(-R * YEARS) * terminal
    se = discounted.std(ddof=1) / np.sqrt(n)
    assert abs(discounted.mean() - S0) < 4 * se


def test_martingale_fails_when_real_world_drift_is_used():
    """Sanity check of the test itself: with drift != r the property must break."""
    n = 400_000
    terminal = simulate_gbm_terminal(S0, 0.12, SIGMA, YEARS, n, seed=99)   # 12% drift
    discounted = np.exp(-R * YEARS) * terminal
    se = discounted.std(ddof=1) / np.sqrt(n)
    assert abs(discounted.mean() - S0) > 10 * se


def test_path_simulator_is_also_a_martingale_under_risk_neutral_drift():
    """The daily-step path simulator must agree with the exact one-step sampler."""
    horizon = 126
    mu_daily = R / 252.0                       # arithmetic drift = r (risk-neutral)
    sigma_daily = SIGMA / np.sqrt(252.0)
    paths = simulate_gbm(S0, mu_daily, sigma_daily, horizon, 100_000, seed=8)
    discounted = np.exp(-R * horizon / 252.0) * paths[-1]
    se = discounted.std(ddof=1) / np.sqrt(paths.shape[1])
    assert abs(discounted.mean() - S0) < 4 * se


# ---- Price curve ------------------------------------------------------------
def test_price_curve_matches_scalar_pricer():
    terminal = simulate_gbm_terminal(S0, R, SIGMA, YEARS, 20_000, seed=3)
    strikes = np.array([90.0, 100.0, 110.0])
    calls, puts = option_price_curve(terminal, strikes, R, YEARS)
    for k, c, p in zip(strikes, calls, puts):
        scalar = price_european_options(terminal, k, R, YEARS)
        assert c == pytest.approx(scalar["call"], rel=1e-12)
        assert p == pytest.approx(scalar["put"], rel=1e-12)
