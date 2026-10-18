"""
risk_engine.py - numerical core of the Monte Carlo Risk Engine
================================================================

Everything in this file is plain NumPy / SciPy / pandas: there is no Streamlit
and no plotting code. That separation is deliberate, because it lets the
automated tests in ``tests/`` import and verify every calculation directly.

Contents
--------
* input validation helpers
* market-data cleaning
* simulation engines: Geometric Brownian Motion and Heston (full-truncation Euler)
* VaR / CVaR and the helpers that describe them honestly (including the case
  where the lower tail of the P&L distribution is profitable)
* option pricing: Monte Carlo, Black-Scholes and the semi-analytic Heston price
* validation studies (convergence of Monte Carlo prices to analytic prices)
* enterprise cash-flow stress testing with macro scenarios
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
N_SIMULATIONS = 10_000            # Market-risk paths
TRADING_DAYS = 252
MIN_HISTORY_POINTS = 60
MAX_HORIZON_DAYS = 504
ENT_MONTHS = 12                   # Enterprise stress-test horizon

MODEL_GBM = "Geometric Brownian Motion (GBM)"
MODEL_HESTON = "Heston Stochastic Volatility Model"


# ----------------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    """Raise ValueError(message) unless the condition holds."""
    if not condition:
        raise ValueError(message)


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(x))
    except TypeError:
        return False


def _positive_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x >= 1


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a simulated probability (successes out of n)."""
    _require(n > 0, "n must be positive")
    _require(0 <= successes <= n, "successes must be between 0 and n")
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# ----------------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------------
def validate_price_history(hist, ticker):
    """
    Turn a raw yfinance history DataFrame into a clean price Series.
    Raises ValueError with a user-friendly message when the data is unusable.
    """
    if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
        raise ValueError(
            f"No price data found for '{ticker}'. Check the ticker symbol "
            "(e.g. AAPL, MSFT, TSLA, RELIANCE.NS, ^GSPC)."
        )

    prices = hist["Close"].dropna()
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)

    if len(prices) < MIN_HISTORY_POINTS:
        raise ValueError(
            f"Only {len(prices)} data points found for '{ticker}'. "
            f"At least {MIN_HISTORY_POINTS} are needed. Try a longer history window."
        )
    return prices


def estimate_parameters(prices: pd.Series):
    """Mean and standard deviation of daily log returns (plus the series itself)."""
    _require(len(prices) >= 3, "At least 3 prices are needed to estimate parameters")
    values = np.asarray(prices, dtype=float)
    _require(np.all(np.isfinite(values)) and np.all(values > 0),
             "Prices must be finite and strictly positive")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return float(log_returns.mean()), float(log_returns.std(ddof=1)), log_returns


# ----------------------------------------------------------------------------
# Simulation engines
# ----------------------------------------------------------------------------
def simulate_gbm(s0, mu, sigma, horizon, n_sims, seed=None):
    """
    Geometric Brownian Motion, daily steps.

        S(t+1) = S(t) * exp( (mu - 0.5*sigma^2) + sigma * Z ),   Z ~ N(0, 1)

    mu is the *arithmetic* daily drift and sigma the daily volatility.
    Returns an array of shape (horizon + 1, n_sims); row 0 is the start price.
    """
    _require(_finite(s0) and s0 > 0, "s0 must be positive")
    _require(_finite(mu), "mu must be finite")
    _require(_finite(sigma) and sigma >= 0, "sigma must be non-negative")
    _require(_positive_int(horizon), "horizon must be a positive integer")
    _require(_positive_int(n_sims), "n_sims must be a positive integer")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((horizon, n_sims))
    log_paths = np.cumsum((mu - 0.5 * sigma**2) + sigma * z, axis=0)
    paths = np.empty((horizon + 1, n_sims))
    paths[0] = s0
    paths[1:] = s0 * np.exp(log_paths)
    return paths


def simulate_gbm_terminal(s0, mu_annual, sigma_annual, years, n_sims, seed=None):
    """Exact one-step sampling of the GBM terminal price (used for pricing)."""
    _require(_finite(s0) and s0 > 0, "s0 must be positive")
    _require(_finite(mu_annual), "mu_annual must be finite")
    _require(_finite(sigma_annual) and sigma_annual >= 0, "sigma_annual must be non-negative")
    _require(_finite(years) and years > 0, "years must be positive")
    _require(_positive_int(n_sims), "n_sims must be a positive integer")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    drift = (mu_annual - 0.5 * sigma_annual**2) * years
    return s0 * np.exp(drift + sigma_annual * np.sqrt(years) * z)


def simulate_heston(s0, mu_annual, v0, kappa, theta, xi, rho, horizon, n_sims,
                    seed=None, keep_paths=True, steps_per_day=1):
    """
    Heston stochastic volatility model (annual units):

        dS = mu * S dt + sqrt(v) * S dW1
        dv = kappa * (theta - v) dt + xi * sqrt(v) dW2,     corr(dW1, dW2) = rho

    The Brownian motions are correlated through

        Z2 = rho * Z1 + sqrt(1 - rho^2) * Z_independent.

    Discretisation: the "full truncation" Euler scheme (Lord, Koekkoek & van Dijk,
    2010). The variance state may dip below zero, but only max(v, 0) is used in the
    drift and diffusion, so square roots are always real, even when the Feller
    condition 2*kappa*theta > xi^2 is violated.

    horizon        number of trading days
    steps_per_day  Euler sub-steps per trading day (1 = daily steps). Increase it
                   to shrink the discretisation bias (used by the convergence study).

    Returns (terminal_prices, price_paths, variance_paths). The path arrays have
    shape (horizon + 1, n_sims) and are recorded once per trading day; they are
    None when keep_paths is False.
    """
    _require(_finite(s0) and s0 > 0, "s0 must be positive")
    _require(_finite(mu_annual), "mu_annual must be finite")
    _require(_finite(v0) and v0 >= 0, "v0 (initial variance) must be non-negative")
    _require(_finite(kappa) and kappa > 0, "kappa must be positive")
    _require(_finite(theta) and theta > 0, "theta (long-run variance) must be positive")
    _require(_finite(xi) and xi > 0, "xi (volatility of volatility) must be positive")
    _require(_finite(rho) and -1.0 <= rho <= 1.0, "rho must be between -1 and 1")
    _require(_positive_int(horizon), "horizon must be a positive integer")
    _require(_positive_int(n_sims), "n_sims must be a positive integer")
    _require(_positive_int(steps_per_day), "steps_per_day must be a positive integer")

    rng = np.random.default_rng(seed)
    dt = 1.0 / (TRADING_DAYS * steps_per_day)
    sqrt_dt = np.sqrt(dt)
    rho_c = np.sqrt(max(0.0, 1.0 - rho**2))

    log_s = np.full(n_sims, np.log(s0))
    v = np.full(n_sims, float(v0))

    s_paths = v_paths = None
    if keep_paths:
        s_paths = np.empty((horizon + 1, n_sims))
        v_paths = np.empty((horizon + 1, n_sims))
        s_paths[0] = s0
        v_paths[0] = v0

    for day in range(1, horizon + 1):
        for _ in range(steps_per_day):
            z1 = rng.standard_normal(n_sims)
            z2 = rho * z1 + rho_c * rng.standard_normal(n_sims)
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            log_s += (mu_annual - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1
            v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * z2
        if keep_paths:
            s_paths[day] = np.exp(log_s)
            v_paths[day] = np.maximum(v, 0.0)

    return np.exp(log_s), s_paths, v_paths


def feller_condition_holds(kappa, theta, xi):
    """Feller condition 2*kappa*theta > xi^2 (variance stays strictly positive)."""
    return bool(2.0 * kappa * theta > xi**2)


def summarize_paths(paths, n_samples=100):
    """Reduce a big path array to the pieces the charts need (saves memory)."""
    paths = np.asarray(paths)
    _require(paths.ndim == 2 and paths.shape[0] >= 1 and paths.shape[1] >= 1,
             "paths must be a non-empty 2-D array")
    p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=1)
    return {
        "days": np.arange(paths.shape[0]),
        "p5": p5, "p50": p50, "p95": p95,
        "samples": paths[:, :n_samples].copy(),
    }


# ----------------------------------------------------------------------------
# Tail risk: VaR and CVaR
# ----------------------------------------------------------------------------
def risk_metrics(pnl) -> dict:
    """
    Value-at-Risk and CVaR (expected shortfall) from simulated profit & loss.

    Both numbers are expressed as *losses*:

        VaR_c  = -(the (100-c)th percentile of P&L)
        CVaR_c = -(the mean P&L of outcomes at or below that percentile)

    A positive value is a loss. A value of zero or below means that even the
    lower tail of the distribution is break-even or profitable, so there is *no
    loss* at that confidence level. That is mathematically valid, but it must
    never be presented as a loss amount: use ``tail_metric_display`` and
    ``explain_tail_risk`` to describe the result to users.
    """
    pnl = np.asarray(pnl, dtype=float)
    _require(pnl.ndim == 1 and pnl.size >= 1, "pnl must be a non-empty 1-D array")
    _require(np.all(np.isfinite(pnl)), "pnl must contain only finite values")

    out = {}
    for level in (95, 99):
        quantile = float(np.percentile(pnl, 100 - level))
        tail = pnl[pnl <= quantile]
        var = -quantile
        cvar = -float(tail.mean())
        out[level] = {
            "var": float(var),
            "cvar": float(cvar),
            "quantile": quantile,                 # P&L at the tail boundary
            "var_is_loss": bool(var > 0),
            "cvar_is_loss": bool(cvar > 0),
        }
    return out


def tail_metric_display(kind, level, value, investment):
    """
    Describe a VaR or CVaR figure for display.

    kind        "VaR" or "CVaR"
    level       95 or 99
    value       the loss figure from risk_metrics (positive = loss)
    investment  amount invested (used for percentages)

    Returns a dict with ``label``, ``value_text`` and ``delta_text``. When the
    figure is not a loss, the text says so explicitly instead of printing a
    negative "loss".
    """
    _require(kind in ("VaR", "CVaR"), "kind must be 'VaR' or 'CVaR'")
    _require(_finite(value), "value must be finite")
    _require(_finite(investment) and investment > 0, "investment must be positive")
    tail_pct = 100 - level

    if value > 0:
        return {
            "label": f"{level}% {kind}",
            "value_text": f"{value:,.2f}",
            "delta_text": f"-{value / investment:.2%} of investment",
            "is_loss": True,
        }

    gain = -value
    if kind == "VaR":
        delta = (f"worst {tail_pct}% of outcomes still earn at least +{gain:,.2f}"
                 if gain > 0 else f"worst {tail_pct}% of outcomes are at break-even or better")
    else:
        delta = (f"average of the worst {tail_pct}% is a gain of +{gain:,.2f}"
                 if gain > 0 else f"average of the worst {tail_pct}% is break-even")
    return {
        "label": f"{level}% {kind} (no loss)",
        "value_text": "No loss",
        "delta_text": delta,
        "is_loss": False,
    }


def format_tail_value(value, decimals=2):
    """One-line text for tables: a loss amount, or an explicit 'no loss' statement."""
    _require(_finite(value), "value must be finite")
    if value > 0:
        return f"{value:,.{decimals}f}"
    if value == 0:
        return "No loss (break-even)"
    return f"No loss (gain of +{-value:,.{decimals}f})"


def tail_line_label(level, metrics_for_level, kind="VaR"):
    """Chart annotation for a tail marker; avoids calling a gain a 'VaR' or 'CVaR'."""
    value = metrics_for_level["var" if kind == "VaR" else "cvar"]
    if value > 0:
        return f"{level}% {kind}"
    state = "gain" if value < 0 else "break-even"
    if kind == "VaR":
        return f"{100 - level}th pct ({state})"
    return f"Worst {100 - level}% avg ({state})"


def explain_tail_risk(metrics, horizon, investment):
    """Plain-English explanation of the 95% VaR / CVaR that is correct for any sign."""
    m = metrics[95]
    if m["var"] > 0:
        text = (
            f"**Reading this:** with 95% confidence, the simulated loss over {horizon} day(s) "
            f"should not exceed **{m['var']:,.2f}**. If losses do exceed that level, "
            f"the average loss in that worst 5% of scenarios is **{m['cvar']:,.2f}** (CVaR)."
        )
        return text

    gain = -m["var"]
    if m["cvar"] > 0:
        worst = f"an average **loss** of {m['cvar']:,.2f}"
    elif m["cvar"] < 0:
        worst = f"an average **gain** of {-m['cvar']:,.2f}"
    else:
        worst = "an average outcome of break-even"
    boundary = (f"gained at least **{gain:,.2f}**" if gain > 0 else "finished at break-even or better")
    return (
        f"**Reading this:** in 95% of the simulations the investment {boundary} over "
        f"{horizon} day(s), so there is **no loss threshold** at the 95% level and VaR is "
        f"reported as *No loss* (the raw figure is negative because the lower tail is "
        f"profitable). The worst 5% of scenarios still produce {worst}. "
        f"With a longer horizon, a higher confidence level or zero drift, a genuine loss "
        f"figure usually appears."
    )


# ----------------------------------------------------------------------------
# Option pricing
# ----------------------------------------------------------------------------
def black_scholes(s0, strike, r, sigma, years):
    """Closed-form Black-Scholes call and put (works on scalar or array strikes)."""
    strike = np.asarray(strike, dtype=float)
    _require(_finite(s0) and s0 > 0, "s0 must be positive")
    _require(np.all(np.isfinite(strike)) and np.all(strike > 0), "strike must be positive")
    _require(_finite(r), "r must be finite")
    _require(_finite(sigma) and sigma > 0, "sigma must be positive")
    _require(_finite(years) and years > 0, "years must be positive")

    vol_sqrt_t = sigma * np.sqrt(years)
    d1 = (np.log(s0 / strike) + (r + 0.5 * sigma**2) * years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    disc = np.exp(-r * years)
    call = s0 * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    put = strike * disc * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    return call, put


def price_european_options(terminal, strike, r, years):
    """
    Monte Carlo European option prices from simulated terminal stock prices:

        price = exp(-r*T) * mean( payoff(S_T) )

    Returns prices and their Monte Carlo standard errors.
    """
    terminal = np.asarray(terminal, dtype=float)
    _require(terminal.ndim == 1 and terminal.size >= 2, "terminal must be 1-D with >= 2 values")
    _require(np.all(np.isfinite(terminal)) and np.all(terminal >= 0),
             "terminal prices must be finite and non-negative")
    _require(_finite(strike) and strike > 0, "strike must be positive")
    _require(_finite(r), "r must be finite")
    _require(_finite(years) and years > 0, "years must be positive")

    disc = np.exp(-r * years)
    call_pv = np.maximum(terminal - strike, 0.0) * disc
    put_pv = np.maximum(strike - terminal, 0.0) * disc
    n = terminal.size
    return {
        "call": float(call_pv.mean()),
        "call_se": float(call_pv.std(ddof=1) / np.sqrt(n)),
        "put": float(put_pv.mean()),
        "put_se": float(put_pv.std(ddof=1) / np.sqrt(n)),
    }


def option_price_curve(terminal, strikes, r, years):
    """Monte Carlo call/put prices for a whole range of strikes at once."""
    terminal = np.asarray(terminal, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    _require(terminal.ndim == 1 and terminal.size >= 1, "terminal must be non-empty 1-D")
    _require(strikes.ndim == 1 and strikes.size >= 1 and np.all(strikes > 0),
             "strikes must be a non-empty 1-D array of positive values")
    _require(_finite(years) and years > 0, "years must be positive")

    disc = np.exp(-r * years)
    diff = terminal[None, :] - strikes[:, None]
    calls = np.maximum(diff, 0.0).mean(axis=1) * disc
    puts = np.maximum(-diff, 0.0).mean(axis=1) * disc
    return calls, puts


# ---- Heston semi-analytic reference price -----------------------------------
def _heston_probability(j, s0, strike, r, years, v0, kappa, theta, xi, rho, upper):
    """
    The risk-neutral probabilities P_j in Heston's (1993) call formula, computed
    with the numerically stable "little Heston trap" formulation of
    Albrecher, Mayer, Schoutens & Tistaert (2007).
    """
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * xi if j == 1 else kappa
    ln_s, ln_k = np.log(s0), np.log(strike)

    def integrand(phi):
        i_phi = 1j * phi
        rxi = rho * xi * i_phi
        d = np.sqrt((rxi - b) ** 2 - xi**2 * (2.0 * u * i_phi - phi**2))
        g = (b - rxi - d) / (b - rxi + d)
        exp_dt = np.exp(-d * years)
        c = r * i_phi * years + (kappa * theta / xi**2) * (
            (b - rxi - d) * years - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
        dd = (b - rxi - d) / xi**2 * (1.0 - exp_dt) / (1.0 - g * exp_dt)
        f = np.exp(c + dd * v0 + i_phi * ln_s)
        return float(np.real(np.exp(-i_phi * ln_k) * f / i_phi))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=integrate.IntegrationWarning)
        value, _ = integrate.quad(integrand, 1e-8, upper, limit=500)
    return 0.5 + value / np.pi


def heston_call_price(s0, strike, r, years, v0, kappa, theta, xi, rho):
    """
    Semi-analytic European call price under the Heston model.

    Uses one-dimensional numerical integration of the characteristic function,
    so it is exact up to quadrature error (about 1e-8). It serves as the trusted
    reference against which the Monte Carlo simulator is validated.
    """
    _require(_finite(s0) and s0 > 0, "s0 must be positive")
    _require(_finite(strike) and strike > 0, "strike must be positive")
    _require(_finite(r), "r must be finite")
    _require(_finite(years) and years > 0, "years must be positive")
    _require(_finite(v0) and v0 > 0, "v0 must be positive for the semi-analytic price")
    _require(_finite(kappa) and kappa > 0, "kappa must be positive")
    _require(_finite(theta) and theta > 0, "theta must be positive")
    _require(_finite(xi) and xi > 0, "xi must be positive")
    _require(_finite(rho) and -1.0 < rho < 1.0, "rho must be strictly between -1 and 1")

    upper = float(np.clip(15.0 / np.sqrt(max(v0, theta, 1e-4) * years), 100.0, 4000.0))
    args = (s0, strike, r, years, v0, kappa, theta, xi, rho, upper)
    p1 = _heston_probability(1, *args)
    p2 = _heston_probability(2, *args)
    price = s0 * p1 - strike * np.exp(-r * years) * p2
    return float(max(price, 0.0))


def heston_put_price(s0, strike, r, years, v0, kappa, theta, xi, rho):
    """Heston put price from put-call parity."""
    call = heston_call_price(s0, strike, r, years, v0, kappa, theta, xi, rho)
    return float(max(call - s0 + strike * np.exp(-r * years), 0.0))


# ---- Validation studies -----------------------------------------------------
def gbm_convergence_study(s0=100.0, strike=100.0, r=0.04, sigma=0.20, years=0.5,
                          path_counts=(2_000, 8_000, 32_000, 128_000, 512_000), seed=7):
    """
    Compare Monte Carlo GBM option prices with Black-Scholes for growing numbers of
    paths. The error should shrink like 1/sqrt(N) and stay within the Monte Carlo
    standard error.
    """
    bs_call, bs_put = black_scholes(s0, strike, r, sigma, years)
    rows = []
    for i, n in enumerate(path_counts):
        terminal = simulate_gbm_terminal(s0, r, sigma, years, int(n), seed + i)
        mc = price_european_options(terminal, strike, r, years)
        rows.append({
            "paths": int(n),
            "mc_call": mc["call"], "call_se": mc["call_se"], "bs_call": float(bs_call),
            "call_error": mc["call"] - float(bs_call),
            "call_z": (mc["call"] - float(bs_call)) / mc["call_se"],
            "mc_put": mc["put"], "put_se": mc["put_se"], "bs_put": float(bs_put),
            "put_error": mc["put"] - float(bs_put),
            "put_z": (mc["put"] - float(bs_put)) / mc["put_se"],
        })
    return pd.DataFrame(rows)


def heston_convergence_study(s0=100.0, strike=100.0, r=0.04, years=0.5,
                             v0=0.04, kappa=1.0, theta=0.04, xi=0.6, rho=-0.7,
                             steps_per_day=(1, 2, 4, 8), n_paths=100_000, seed=11):
    """
    Compare Monte Carlo Heston call prices (full-truncation Euler) with the
    semi-analytic price while refining the time step. The remaining gap is the
    discretisation bias, which should shrink as steps_per_day grows.
    """
    ref = heston_call_price(s0, strike, r, years, v0, kappa, theta, xi, rho)
    horizon = max(1, int(round(years * TRADING_DAYS)))
    rows = []
    for i, steps in enumerate(steps_per_day):
        terminal, _, _ = simulate_heston(
            s0, r, v0, kappa, theta, xi, rho, horizon, n_paths, seed + i,
            keep_paths=False, steps_per_day=int(steps))
        mc = price_european_options(terminal, strike, r, horizon / TRADING_DAYS)
        rows.append({
            "steps_per_day": int(steps),
            "mc_call": mc["call"], "call_se": mc["call_se"], "reference": ref,
            "error": mc["call"] - ref,
            "z": (mc["call"] - ref) / mc["call_se"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Market-risk pipeline (used by the Streamlit tab)
# ----------------------------------------------------------------------------
def _heston_reference(s0, strike, strikes, r, years, h):
    """Semi-analytic Heston prices; returns NaNs if the quadrature fails."""
    try:
        call = heston_call_price(s0, strike, r, years, h["v0"], h["kappa"],
                                 h["theta"], h["xi"], h["rho"])
        put = call - s0 + strike * np.exp(-r * years)
        curve_calls = np.array([
            heston_call_price(s0, k, r, years, h["v0"], h["kappa"], h["theta"],
                              h["xi"], h["rho"]) for k in strikes])
        curve_puts = curve_calls - s0 + strikes * np.exp(-r * years)
        return {"name": "Heston (semi-analytic)", "call": float(call),
                "put": float(max(put, 0.0)), "curve_calls": curve_calls,
                "curve_puts": np.maximum(curve_puts, 0.0)}
    except Exception:  # never let the benchmark break the dashboard
        nan = float("nan")
        return {"name": "Heston (semi-analytic)", "call": nan, "put": nan,
                "curve_calls": None, "curve_puts": None}


def build_market_results(prices, cfg):
    """Run every simulation the Market tab needs and return light-weight results."""
    s0 = float(prices.iloc[-1])
    m_log, sigma, log_returns = estimate_parameters(prices)
    _require(sigma > 0 and np.isfinite(sigma), "Volatility could not be estimated")
    mu_arith = m_log + 0.5 * sigma**2                 # arithmetic daily drift
    historical_drift = cfg["drift_mode"].startswith("Historical")
    mu_daily = mu_arith if historical_drift else 0.0
    sigma_ann = sigma * np.sqrt(TRADING_DAYS)
    var_ann = sigma**2 * TRADING_DAYS

    horizon = int(cfg["horizon"])
    years = horizon / TRADING_DAYS
    seed = cfg["seed"]
    r = cfg["rf"]
    investment = cfg["investment"]
    n_sims = int(cfg.get("n_sims", N_SIMULATIONS))
    is_heston = cfg["model"] == MODEL_HESTON
    _require(investment > 0, "investment must be positive")

    heston = None
    vol_summary = None
    if is_heston:
        heston = {
            "kappa": cfg["kappa"],
            "xi": cfg["xi"],
            "rho": cfg["rho"],
            "theta": var_ann if cfg["calibrate"] else cfg["theta"],
            "v0": var_ann if cfg["calibrate"] else cfg["v0"],
        }
        terminal, s_paths, v_paths = simulate_heston(
            s0, mu_daily * TRADING_DAYS, heston["v0"], heston["kappa"],
            heston["theta"], heston["xi"], heston["rho"], horizon,
            n_sims, seed, keep_paths=True)
        vol_paths = np.sqrt(v_paths) * 100.0          # annualised vol in %
        vol_summary = summarize_paths(vol_paths, n_samples=60)
        del v_paths, vol_paths
    else:
        s_paths = simulate_gbm(s0, mu_daily, sigma, horizon, n_sims, seed)
        terminal = s_paths[-1]

    fan = summarize_paths(s_paths)
    final_prices = terminal.copy()
    del s_paths

    final_values = investment * final_prices / s0
    pnl = final_values - investment

    # Reference distribution: plain GBM terminal P&L (shows the effect of Heston)
    gbm_pnl = None
    if is_heston:
        gbm_terminal = simulate_gbm_terminal(
            s0, mu_daily * TRADING_DAYS, sigma_ann, years, n_sims,
            None if seed is None else seed + 2)
        gbm_pnl = investment * gbm_terminal / s0 - investment

    # ---- Risk-neutral option pricing ---------------------------------------
    rn_seed = None if seed is None else seed + 1
    if is_heston:
        terminal_rn, _, _ = simulate_heston(
            s0, r, heston["v0"], heston["kappa"], heston["theta"], heston["xi"],
            heston["rho"], horizon, n_sims, rn_seed, keep_paths=False)
    else:
        terminal_rn = simulate_gbm_terminal(s0, r, sigma_ann, years, n_sims, rn_seed)

    strike = cfg["strike"] if cfg["strike"] > 0 else s0
    options = price_european_options(terminal_rn, strike, r, years)
    bs_call, bs_put = black_scholes(s0, strike, r, sigma_ann, years)
    options["bs_call"] = float(bs_call)
    options["bs_put"] = float(bs_put)
    options["strike"] = float(strike)
    options["years"] = years
    options["martingale_ratio"] = float(np.exp(-r * years) * terminal_rn.mean() / s0)
    options["martingale_se"] = float(
        np.exp(-r * years) * terminal_rn.std(ddof=1) / np.sqrt(terminal_rn.size) / s0)
    options["parity_gap"] = float(
        (options["call"] - options["put"]) - (s0 - strike * np.exp(-r * years)))

    strikes = np.linspace(0.7 * s0, 1.3 * s0, 31)
    mc_calls, mc_puts = option_price_curve(terminal_rn, strikes, r, years)
    bs_calls, bs_puts = black_scholes(s0, strikes, r, sigma_ann, years)

    if is_heston:
        reference = _heston_reference(s0, strike, strikes, r, years, heston)
    else:
        reference = {"name": "Black-Scholes", "call": float(bs_call), "put": float(bs_put),
                     "curve_calls": bs_calls, "curve_puts": bs_puts}
    options["ref_name"] = reference["name"]
    options["ref_call"] = reference["call"]
    options["ref_put"] = reference["put"]

    curve = {"strikes": strikes, "mc_calls": mc_calls, "mc_puts": mc_puts,
             "bs_calls": bs_calls, "bs_puts": bs_puts,
             "ref_calls": reference["curve_calls"], "ref_puts": reference["curve_puts"],
             "ref_name": reference["name"]}

    return {
        "s0": s0, "m_log": m_log, "mu_arith": mu_arith, "sigma": sigma,
        "sigma_ann": sigma_ann, "var_ann": var_ann, "log_returns": log_returns,
        "heston": heston, "fan": fan, "vol_summary": vol_summary,
        "final_prices": final_prices, "final_values": final_values, "pnl": pnl,
        "gbm_pnl": gbm_pnl, "metrics": risk_metrics(pnl),
        "gbm_metrics": risk_metrics(gbm_pnl) if gbm_pnl is not None else None,
        "options": options, "curve": curve,
    }


# ----------------------------------------------------------------------------
# Enterprise cash-flow stress test
# ----------------------------------------------------------------------------
# Stylised, illustrative stress parameters (not official forecasts).
# fixed_pct = fixed overheads as a % of baseline monthly revenue.
SCENARIOS = {
    "Baseline (Normal)": {
        "vol": 12.0, "fixed_pct": 40.0, "var_ratio": 45.0,
        "shock": 0.0, "growth": 0.0, "rate": 6.0,
        "desc": "Business as usual: stable demand, normal supplier terms and "
                "borrowing costs.",
    },
    "2008 Financial Crisis": {
        "vol": 30.0, "fixed_pct": 42.0, "var_ratio": 50.0,
        "shock": -25.0, "growth": -1.0, "rate": 8.0,
        "desc": "Demand collapse (-25% revenue shock, then -1% per month), very volatile "
                "sales, tighter supplier terms, wider credit spreads (8% debt cost).",
    },
    "Supply Chain Shock": {
        "vol": 22.0, "fixed_pct": 41.0, "var_ratio": 55.0,
        "shock": -8.0, "growth": 0.0, "rate": 6.5,
        "desc": "Input costs surge (variable cost ratio up to 55%), deliveries are "
                "disrupted (-8% revenue) and sales become erratic.",
    },
    "Rate Spike (+300 bps)": {
        "vol": 20.0, "fixed_pct": 42.0, "var_ratio": 48.0,
        "shock": -8.0, "growth": -0.5, "rate": 9.0,
        "desc": "Policy rates jump 300 basis points: floating-rate debt now costs 9% "
                "(6% + 3%), demand softens (-8% revenue, then -0.5% per month) and "
                "costs creep up.",
    },
}
DEFAULT_SCENARIO = "Baseline (Normal)"


def simulate_cash_flows(revenue, fixed, var_ratio, vol, opening, shock, growth,
                        debt, rate, months=ENT_MONTHS, n_paths=5_000, seed=None):
    """
    Monte Carlo simulation of month-end cash balances.

    Expected revenue:  E[R_t] = revenue * (1 + shock) * (1 + growth)^(t-1)
    Random revenue:    R_t = E[R_t] * exp(vol*Z - 0.5*vol^2),  Z ~ N(0,1), i.i.d. each month
                       (log-normal: never negative, mean equals E[R_t])
    Monthly cash flow: R_t * (1 - var_ratio) - fixed - debt * rate / 12
    Cash balance:      Cash_t = opening + sum of monthly cash flows

    All ratios are fractions (0.45, not 45). Returns an array of shape
    (months + 1, n_paths); row 0 is the opening cash.
    """
    _require(_finite(revenue) and revenue >= 0, "revenue must be non-negative")
    _require(_finite(fixed) and fixed >= 0, "fixed overheads must be non-negative")
    _require(_finite(var_ratio) and 0.0 <= var_ratio < 1.0,
             "variable cost ratio must be a fraction in [0, 1)")
    _require(_finite(vol) and vol >= 0, "volatility must be non-negative")
    _require(_finite(opening) and opening >= 0, "opening cash must be non-negative")
    _require(_finite(shock) and shock > -1.0, "shock must be greater than -100%")
    _require(_finite(growth) and growth > -1.0, "growth must be greater than -100%")
    _require(_finite(debt) and debt >= 0, "debt must be non-negative")
    _require(_finite(rate) and rate >= 0, "interest rate must be non-negative")
    _require(_positive_int(months), "months must be a positive integer")
    _require(_positive_int(n_paths), "n_paths must be a positive integer")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((months, n_paths))
    expected = revenue * (1.0 + shock) * (1.0 + growth) ** np.arange(months)
    revenues = expected[:, None] * np.exp(vol * z - 0.5 * vol**2)
    net = revenues * (1.0 - var_ratio) - fixed - debt * rate / 12.0

    cash = np.empty((months + 1, n_paths))
    cash[0] = opening
    cash[1:] = opening + np.cumsum(net, axis=0)
    return cash


def cash_stats(cash):
    """Insolvency = month-end cash below zero in at least one month."""
    cash = np.asarray(cash, dtype=float)
    _require(cash.ndim == 2 and cash.shape[0] >= 2 and cash.shape[1] >= 1,
             "cash must be a 2-D array with an opening row and at least one month")
    _require(np.all(np.isfinite(cash)), "cash must contain only finite values")

    n = cash.shape[1]
    balances = cash[1:]
    negative = balances < 0
    breached_by_month = np.cumsum(negative, axis=0) > 0
    prob_by_month = breached_by_month.mean(axis=1)

    ever = breached_by_month[-1]
    k = int(ever.sum())
    first_month = np.argmax(negative, axis=0) + 1
    ending = cash[-1]
    peak_gap = np.maximum(0.0, -balances.min(axis=0))
    ci_low, ci_high = wilson_interval(k, n)
    p5 = float(np.percentile(ending, 5))

    return {
        "n": n,
        "insolvent_paths": k,
        "prob": k / n,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prob_by_month": prob_by_month,
        "prob_negative_at_month": negative.mean(axis=1),
        "avg_first_month": float(first_month[ever].mean()) if k else None,
        "ending": ending,
        "mean_end": float(ending.mean()),
        "median_end": float(np.median(ending)),
        "p5_end": p5,
        "es5_end": float(ending[ending <= p5].mean()),
        "buffer_95": float(np.percentile(peak_gap, 95)),
        "median_by_month": np.median(balances, axis=1),
        "p5_by_month": np.percentile(balances, 5, axis=1),
    }


def compare_scenarios(revenue, opening, debt, trials, seed):
    """Run every preset on the user's business size (common random numbers)."""
    rows = []
    for name, p in SCENARIOS.items():
        cash = simulate_cash_flows(
            revenue, revenue * p["fixed_pct"] / 100.0, p["var_ratio"] / 100.0,
            p["vol"] / 100.0, opening, p["shock"] / 100.0, p["growth"] / 100.0,
            debt, p["rate"] / 100.0, n_paths=trials, seed=seed)
        s = cash_stats(cash)
        rows.append({"name": name, "prob": s["prob"], "mean_end": s["mean_end"],
                     "p5_end": s["p5_end"]})
    return rows
