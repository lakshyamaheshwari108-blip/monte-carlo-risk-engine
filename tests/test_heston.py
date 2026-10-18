"""
Priority 3 - validate the Heston simulator against a trusted reference.

Finite, positive prices do not prove the simulated distribution is right. The checks
below compare the full-truncation Euler Monte Carlo with an independent semi-analytic
Heston price (Fourier inversion of the characteristic function), include parameter
sets that violate the Feller condition, and run a time-step convergence study.

Reference validation
    * collapses to Black-Scholes as vol-of-vol -> 0 (at the theoretically expected rate);
    * agrees with the original 1993 formulation where that formulation is stable;
    * obeys arbitrage bounds, monotonicity and convexity in the strike;
    * agrees with brute-force simulation (including where the 1993 formula breaks down).
"""
import numpy as np
import pytest
from scipy import integrate
from scipy.stats import skew

from risk_engine import (
    black_scholes,
    feller_condition_holds,
    heston_call_price,
    heston_convergence_study,
    heston_put_price,
    price_european_options,
    simulate_heston,
)

S0, K, R = 100.0, 100.0, 0.03

CASES = {
    "feller_ok_negative_rho": dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.7),
    "feller_ok_positive_rho": dict(v0=0.03, kappa=3.0, theta=0.06, xi=0.40, rho=0.3),
    "feller_violated":        dict(v0=0.04, kappa=1.0, theta=0.04, xi=0.60, rho=-0.7),
    "feller_badly_violated":  dict(v0=0.04, kappa=0.5, theta=0.04, xi=1.00, rho=-0.9),
}


def mc_call(params, years, n_paths, seed, steps_per_day=1, strike=K):
    horizon = int(round(years * 252))
    terminal, _, _ = simulate_heston(
        S0, R, params["v0"], params["kappa"], params["theta"], params["xi"], params["rho"],
        horizon, n_paths, seed, keep_paths=False, steps_per_day=steps_per_day)
    return price_european_options(terminal, strike, R, horizon / 252.0)


# ---- The reference price itself ---------------------------------------------
def original_heston_call(s0, strike, r, years, v0, kappa, theta, xi, rho):
    """Independent implementation of Heston's ORIGINAL (1993) formulation."""
    ln_s, ln_k = np.log(s0), np.log(strike)

    def probability(j):
        u = 0.5 if j == 1 else -0.5
        b = kappa - rho * xi if j == 1 else kappa

        def integrand(phi):
            ip = 1j * phi
            rxi = rho * xi * ip
            d = np.sqrt((rxi - b) ** 2 - xi**2 * (2 * u * ip - phi**2))
            g = (b - rxi + d) / (b - rxi - d)
            c = r * ip * years + (kappa * theta / xi**2) * (
                (b - rxi + d) * years - 2 * np.log((1 - g * np.exp(d * years)) / (1 - g)))
            dd = (b - rxi + d) / xi**2 * (1 - np.exp(d * years)) / (1 - g * np.exp(d * years))
            return float(np.real(np.exp(-ip * ln_k) * np.exp(c + dd * v0 + ip * ln_s) / ip))

        return 0.5 + integrate.quad(integrand, 1e-8, 400, limit=800)[0] / np.pi

    return s0 * probability(1) - strike * np.exp(-r * years) * probability(2)


@pytest.mark.parametrize("years", [0.25, 1.0, 3.0])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_reference_collapses_to_black_scholes_when_vol_of_vol_vanishes(years, strike):
    """With rho = 0 and v0 = theta the gap to Black-Scholes must be O(xi^2)."""
    xi = 0.001
    heston = heston_call_price(S0, strike, R, years, 0.04, 3.0, 0.04, xi, 0.0)
    bs = float(black_scholes(S0, strike, R, 0.20, years)[0])
    assert heston == pytest.approx(bs, abs=1e-5)


def test_reference_converges_to_black_scholes_at_the_theoretical_rate():
    def gap(xi):
        return abs(heston_call_price(S0, 120.0, R, 1.0, 0.04, 3.0, 0.04, xi, 0.0)
                   - float(black_scholes(S0, 120.0, R, 0.20, 1.0)[0]))
    # rho = 0  ->  difference ~ xi^2: quadrupling xi should multiply the gap by ~16
    ratio = gap(0.008) / gap(0.002)
    assert ratio == pytest.approx(16.0, rel=0.15)


def test_reference_agrees_with_the_original_1993_formulation_where_it_is_stable():
    for name in ("feller_ok_negative_rho", "feller_ok_positive_rho"):
        p = CASES[name]
        for years in (0.25, 0.5, 1.0):
            a = heston_call_price(S0, K, R, years, **p)
            b = original_heston_call(S0, K, R, years, **p)
            assert a == pytest.approx(b, abs=1e-4)


def test_reference_satisfies_arbitrage_bounds_and_parity():
    p = CASES["feller_ok_negative_rho"]
    for strike in (70.0, 90.0, 100.0, 110.0, 140.0):
        call = heston_call_price(S0, strike, R, 0.5, **p)
        put = heston_put_price(S0, strike, R, 0.5, **p)
        disc_k = strike * np.exp(-R * 0.5)
        assert max(S0 - disc_k, 0) - 1e-8 <= call <= S0 + 1e-8
        assert max(disc_k - S0, 0) - 1e-8 <= put <= disc_k + 1e-8
        assert call - put == pytest.approx(S0 - disc_k, abs=1e-8)


def test_reference_is_decreasing_and_convex_in_the_strike():
    strikes = np.linspace(70, 130, 25)
    calls = np.array([heston_call_price(S0, k, R, 0.5, **CASES["feller_ok_negative_rho"])
                      for k in strikes])
    assert np.all(np.diff(calls) < 0)
    assert np.all(np.diff(calls, 2) > -1e-8)     # convexity (no butterfly arbitrage)


@pytest.mark.parametrize("name", list(CASES))
def test_reference_is_finite_and_sane_for_all_cases(name):
    price = heston_call_price(S0, K, R, 1.0, **CASES[name])
    assert np.isfinite(price) and 0 < price < S0


def test_negative_correlation_creates_a_put_skew_relative_to_black_scholes():
    p = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.8)
    heston_put = heston_put_price(S0, 85.0, R, 0.5, **p)
    bs_put = float(black_scholes(S0, 85.0, R, 0.20, 0.5)[1])
    assert heston_put > 1.15 * bs_put        # out-of-the-money puts are richer


# ---- Monte Carlo against the reference --------------------------------------
@pytest.mark.parametrize("name", ["feller_ok_negative_rho", "feller_ok_positive_rho",
                                  "feller_violated"])
def test_mc_matches_semi_analytic_price_within_statistical_error(name):
    p = CASES[name]
    years = 0.5
    ref = heston_call_price(S0, K, R, years, **p)
    mc = mc_call(p, years, n_paths=200_000, seed=5)
    assert abs(mc["call"] - ref) < 4 * mc["call_se"]


@pytest.mark.parametrize("strike", [85.0, 115.0])
def test_mc_matches_reference_away_from_the_money(strike):
    p = CASES["feller_ok_negative_rho"]
    ref = heston_call_price(S0, strike, R, 0.5, **p)
    mc = mc_call(p, 0.5, n_paths=200_000, seed=6, strike=strike)
    assert abs(mc["call"] - ref) < 4 * mc["call_se"]


def test_badly_violated_feller_case_has_small_but_real_discretisation_bias():
    """
    xi = 1 and rho = -0.9 is a notoriously hard parameter set. Daily Euler steps show a
    statistically significant (but small) positive bias, which refining the step reduces.
    """
    p = CASES["feller_badly_violated"]
    years = 0.25
    ref = heston_call_price(S0, K, R, years, **p)
    coarse = mc_call(p, years, n_paths=150_000, seed=21, steps_per_day=1)
    fine = mc_call(p, years, n_paths=150_000, seed=22, steps_per_day=16)
    err_coarse = coarse["call"] - ref
    err_fine = fine["call"] - ref
    assert err_coarse > 3 * coarse["call_se"]            # the bias is real and detectable...
    assert err_coarse / ref < 0.03                        # ...but below 3% of the price
    assert abs(err_fine) < 0.5 * abs(err_coarse)          # refining the step shrinks it
    assert abs(err_fine) / ref < 0.01


def test_convergence_study_returns_a_consistent_table():
    df = heston_convergence_study(years=0.25, steps_per_day=(1, 4), n_paths=40_000, seed=3,
                                  **CASES["feller_violated"])
    assert list(df["steps_per_day"]) == [1, 4]
    assert df["reference"].nunique() == 1
    assert np.allclose(df["error"], df["mc_call"] - df["reference"])
    assert np.allclose(df["z"], df["error"] / df["call_se"])


# ---- Distributional checks of the simulator ---------------------------------
@pytest.mark.parametrize("name", ["feller_ok_negative_rho", "feller_badly_violated"])
def test_variance_never_negative_and_mean_follows_the_cir_formula(name):
    p = dict(CASES[name]); p["v0"] = 0.09 if name == "feller_ok_negative_rho" else p["v0"]
    years, horizon = 0.5, 126
    _, _, v_paths = simulate_heston(S0, R, p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"],
                                    horizon, 100_000, seed=3, keep_paths=True)
    assert v_paths.min() >= 0.0
    exact_mean = p["theta"] + (p["v0"] - p["theta"]) * np.exp(-p["kappa"] * years)
    assert v_paths[-1].mean() == pytest.approx(exact_mean, rel=0.03)


@pytest.mark.parametrize("name", ["feller_ok_negative_rho", "feller_badly_violated"])
def test_discounted_price_is_a_martingale_under_heston(name):
    p = CASES[name]
    n, years = 200_000, 0.5
    terminal, _, _ = simulate_heston(S0, R, p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"],
                                     126, n, seed=12, keep_paths=False)
    discounted = np.exp(-R * years) * terminal
    se = discounted.std(ddof=1) / np.sqrt(n)
    assert abs(discounted.mean() - S0) < 4 * se


@pytest.mark.parametrize("rho", [-0.7, 0.3, 0.8])
def test_simulated_shocks_have_the_requested_correlation(rho):
    p = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.4)
    _, s_paths, v_paths = simulate_heston(S0, R, p["v0"], p["kappa"], p["theta"], p["xi"], rho,
                                          63, 20_000, seed=4, keep_paths=True)
    d_log_s = np.diff(np.log(s_paths), axis=0).ravel()
    d_v = np.diff(v_paths, axis=0).ravel()
    assert np.corrcoef(d_log_s, d_v)[0, 1] == pytest.approx(rho, abs=0.03)


def test_correlation_sign_controls_the_skew_of_returns():
    def sk(rho):
        term, _, _ = simulate_heston(S0, R, 0.04, 2.0, 0.04, 0.6, rho, 126, 100_000,
                                     seed=1, keep_paths=False)
        return skew(np.log(term / S0))
    assert sk(-0.9) < -0.3
    assert sk(+0.9) > +0.3
    assert abs(sk(0.0)) < 0.15


def test_vol_of_vol_fattens_the_tails():
    def excess_kurtosis(xi):
        term, _, _ = simulate_heston(S0, R, 0.04, 2.0, 0.04, xi, -0.5, 126, 100_000,
                                     seed=2, keep_paths=False)
        r = np.log(term / S0)
        return float(np.mean(((r - r.mean()) / r.std()) ** 4) - 3)
    assert excess_kurtosis(0.8) > excess_kurtosis(0.05) + 0.5


def test_heston_with_negligible_vol_of_vol_reproduces_gbm_prices():
    n = 300_000
    term, _, _ = simulate_heston(S0, R, 0.04, 3.0, 0.04, 1e-4, -0.5, 63, n, seed=9,
                                 keep_paths=False)
    mc = price_european_options(term, K, R, 63 / 252.0)
    bs = float(black_scholes(S0, K, R, 0.20, 63 / 252.0)[0])
    assert abs(mc["call"] - bs) < 4 * mc["call_se"]


def test_same_seed_gives_identical_results_and_new_seed_differs():
    args = (S0, R, 0.04, 2.0, 0.04, 0.3, -0.7, 21, 5_000)
    a, _, _ = simulate_heston(*args, seed=1, keep_paths=False)
    b, _, _ = simulate_heston(*args, seed=1, keep_paths=False)
    c, _, _ = simulate_heston(*args, seed=2, keep_paths=False)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_path_arrays_have_the_expected_shape_and_start_values():
    term, s_paths, v_paths = simulate_heston(S0, R, 0.05, 2.0, 0.04, 0.3, -0.7, 10, 500, seed=1)
    assert s_paths.shape == v_paths.shape == (11, 500)
    assert np.all(s_paths[0] == S0) and np.all(v_paths[0] == 0.05)
    assert np.array_equal(term, s_paths[-1])
    assert np.all(s_paths > 0)


def test_feller_condition_helper():
    assert feller_condition_holds(2.0, 0.04, 0.30) is True
    assert feller_condition_holds(1.0, 0.04, 0.60) is False
    assert feller_condition_holds(0.5, 0.04, 1.00) is False
