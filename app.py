"""
Monte Carlo Risk Engine  -  Quantitative Finance & Enterprise Risk Suite
========================================================================

Tab 1  Market Asset Risk
       * live market data (yfinance)
       * model switch: Geometric Brownian Motion  |  Heston stochastic volatility
       * 10,000 Monte Carlo paths -> 95% / 99% VaR and CVaR
       * Monte Carlo European call / put pricing (risk-neutral) with a
         Black-Scholes benchmark (SciPy)

Tab 2  Enterprise Cash-Flow Stress Test
       * macro scenario presets that re-set the model parameters
       * 5,000+ Monte Carlo trials over a 12-month horizon
       * ending-cash distribution and insolvency probability
       * one-click Executive PDF Risk Report (fpdf2)

Run locally:  streamlit run app.py
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats as sp_stats
from scipy.stats import norm

try:  # PDF export is optional: the rest of the app works without it
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    FPDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    FPDF = object
    XPos = YPos = None
    FPDF_AVAILABLE = False

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
N_SIMULATIONS = 10_000            # Market tab paths
TRADING_DAYS = 252
MIN_HISTORY_POINTS = 60
MAX_HORIZON_DAYS = 504

ENT_MONTHS = 12                   # Enterprise tab horizon
ENT_TRIAL_OPTIONS = [5_000, 10_000, 25_000, 50_000]

MODEL_GBM = "Geometric Brownian Motion (GBM)"
MODEL_HESTON = "Heston Stochastic Volatility Model"

PLOT_TEMPLATE = "plotly_dark"
ACCENT = "#00d4ff"
RED = "#ff4b6e"
ORANGE = "#ffa62b"
GREEN = "#2ee6a6"
PINK = "#ff00aa"
VIOLET = "#a78bfa"

st.set_page_config(
    page_title="Monte Carlo Risk Engine",
    page_icon="📉",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def style(fig, title, height=450):
    """Apply the common dark look to a Plotly figure."""
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None)


def memo(slot, key, builder):
    """
    Keep exactly one cached result per slot in the session. If the inputs (key)
    have not changed since the last run, the stored result is reused. This keeps
    the app fast and its numbers stable when the page reruns (for example after
    clicking a download button), without piling up memory.
    """
    cached = st.session_state.get(slot)
    if cached is not None and cached["key"] == key:
        return cached["value"]
    value = builder()
    st.session_state[slot] = {"key": key, "value": value}
    return value


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a simulated probability."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# ============================================================================
# TAB 1 - MARKET ASSET RISK
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.Series:
    """Download adjusted closing prices. Raises ValueError on any problem."""
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception as exc:  # network errors, bad symbols, rate limits
        raise ValueError(f"Could not download data for '{ticker}': {exc}") from exc

    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(
            f"No price data found for '{ticker}'. Check the ticker symbol "
            "(e.g. AAPL, MSFT, TSLA, RELIANCE.NS, ^GSPC)."
        )

    prices = hist["Close"].dropna()
    prices = prices[prices > 0]
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)

    if len(prices) < MIN_HISTORY_POINTS:
        raise ValueError(
            f"Only {len(prices)} data points found for '{ticker}'. "
            f"At least {MIN_HISTORY_POINTS} are needed. Try a longer history window."
        )
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def load_name(ticker: str) -> str:
    """Best-effort company name; never fails."""
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def estimate_parameters(prices: pd.Series):
    """Mean and standard deviation of daily log returns (plus the series itself)."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return float(log_returns.mean()), float(log_returns.std(ddof=1)), log_returns


# ---- Simulation engines -----------------------------------------------------
def simulate_gbm(s0, mu, sigma, horizon, n_sims, seed=None):
    """
    Geometric Brownian Motion, daily steps.

        S(t+1) = S(t) * exp( (mu - 0.5*sigma^2) + sigma * Z ),   Z ~ N(0, 1)

    mu is the *arithmetic* daily drift and sigma the daily volatility.
    Returns an array of shape (horizon + 1, n_sims); row 0 is the start price.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((horizon, n_sims))
    log_paths = np.cumsum((mu - 0.5 * sigma**2) + sigma * z, axis=0)
    paths = np.empty((horizon + 1, n_sims))
    paths[0] = s0
    paths[1:] = s0 * np.exp(log_paths)
    return paths


def simulate_gbm_terminal(s0, mu_annual, sigma_annual, years, n_sims, seed=None):
    """Exact one-step sampling of the GBM terminal price (used for pricing)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    drift = (mu_annual - 0.5 * sigma_annual**2) * years
    return s0 * np.exp(drift + sigma_annual * np.sqrt(years) * z)


def simulate_heston(s0, mu_annual, v0, kappa, theta, xi, rho, horizon, n_sims,
                    seed=None, keep_paths=True):
    """
    Heston stochastic volatility model (annual units, daily Euler steps):

        dS = mu * S dt + sqrt(v) * S dW1
        dv = kappa * (theta - v) dt + xi * sqrt(v) dW2,     corr(dW1, dW2) = rho

    The two Brownian motions are correlated through

        Z2 = rho * Z1 + sqrt(1 - rho^2) * Z_independent.

    The "full truncation" Euler scheme (Lord, Koekkoek & van Dijk) is used, so the
    variance can never produce a negative square root, even when the Feller
    condition 2*kappa*theta > xi^2 is violated.

    Returns (terminal_prices, price_paths, variance_paths). The two path arrays
    have shape (horizon + 1, n_sims) when keep_paths is True, otherwise None.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
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

    for t in range(1, horizon + 1):
        z1 = rng.standard_normal(n_sims)
        z2 = rho * z1 + rho_c * rng.standard_normal(n_sims)
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        log_s += (mu_annual - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1
        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * z2
        if keep_paths:
            s_paths[t] = np.exp(log_s)
            v_paths[t] = np.maximum(v, 0.0)

    return np.exp(log_s), s_paths, v_paths


def summarize_paths(paths, n_samples=100):
    """Reduce a big path array to the pieces the charts need (saves memory)."""
    p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=1)
    return {
        "days": np.arange(paths.shape[0]),
        "p5": p5, "p50": p50, "p95": p95,
        "samples": paths[:, :n_samples].copy(),
    }


def risk_metrics(pnl: np.ndarray) -> dict:
    """VaR and CVaR from simulated profit/loss. Results are positive loss numbers."""
    out = {}
    for level in (95, 99):
        var = -np.percentile(pnl, 100 - level)
        tail = pnl[pnl <= -var]
        cvar = -tail.mean() if tail.size else var
        out[level] = {"var": float(var), "cvar": float(cvar)}
    return out


# ---- Option pricing ---------------------------------------------------------
def black_scholes(s0, strike, r, sigma, years):
    """Closed-form Black-Scholes call and put (works on scalar or array strikes)."""
    strike = np.asarray(strike, dtype=float)
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
    disc = np.exp(-r * years)
    diff = terminal[None, :] - strikes[:, None]
    calls = np.maximum(diff, 0.0).mean(axis=1) * disc
    puts = np.maximum(-diff, 0.0).mean(axis=1) * disc
    return calls, puts


def build_market_results(prices, cfg):
    """Run every simulation the Market tab needs and return light-weight results."""
    s0 = float(prices.iloc[-1])
    m_log, sigma, log_returns = estimate_parameters(prices)
    mu_arith = m_log + 0.5 * sigma**2                 # arithmetic daily drift
    historical_drift = cfg["drift_mode"].startswith("Historical")
    mu_daily = mu_arith if historical_drift else 0.0
    sigma_ann = sigma * np.sqrt(TRADING_DAYS)
    var_ann = sigma**2 * TRADING_DAYS

    horizon = cfg["horizon"]
    years = horizon / TRADING_DAYS
    seed = cfg["seed"]
    r = cfg["rf"]
    investment = cfg["investment"]
    is_heston = cfg["model"] == MODEL_HESTON

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
            N_SIMULATIONS, seed, keep_paths=True)
        vol_paths = np.sqrt(v_paths) * 100.0          # annualised vol in %
        vol_summary = summarize_paths(vol_paths, n_samples=60)
        del v_paths, vol_paths
    else:
        s_paths = simulate_gbm(s0, mu_daily, sigma, horizon, N_SIMULATIONS, seed)
        terminal = s_paths[-1]

    fan = summarize_paths(s_paths)
    final_prices = terminal.copy()
    del s_paths

    final_values = investment * final_prices / s0
    pnl = final_values - investment

    # Reference distribution: plain GBM terminal P&L (to show the effect of Heston)
    gbm_pnl = None
    if is_heston:
        gbm_terminal = simulate_gbm_terminal(
            s0, mu_daily * TRADING_DAYS, sigma_ann, years, N_SIMULATIONS,
            None if seed is None else seed + 2)
        gbm_pnl = investment * gbm_terminal / s0 - investment

    # ---- Risk-neutral option pricing ---------------------------------------
    rn_seed = None if seed is None else seed + 1
    if is_heston:
        terminal_rn, _, _ = simulate_heston(
            s0, r, heston["v0"], heston["kappa"], heston["theta"], heston["xi"],
            heston["rho"], horizon, N_SIMULATIONS, rn_seed, keep_paths=False)
    else:
        terminal_rn = simulate_gbm_terminal(s0, r, sigma_ann, years, N_SIMULATIONS, rn_seed)

    strike = cfg["strike"] if cfg["strike"] > 0 else s0
    options = price_european_options(terminal_rn, strike, r, years)
    bs_call, bs_put = black_scholes(s0, strike, r, sigma_ann, years)
    options["bs_call"] = float(bs_call)
    options["bs_put"] = float(bs_put)
    options["strike"] = float(strike)
    options["years"] = years
    options["martingale_ratio"] = float(np.exp(-r * years) * terminal_rn.mean() / s0)
    options["parity_gap"] = float(
        (options["call"] - options["put"]) - (s0 - strike * np.exp(-r * years)))

    strikes = np.linspace(0.7 * s0, 1.3 * s0, 31)
    mc_calls, mc_puts = option_price_curve(terminal_rn, strikes, r, years)
    bs_calls, bs_puts = black_scholes(s0, strikes, r, sigma_ann, years)
    curve = {"strikes": strikes, "mc_calls": mc_calls, "mc_puts": mc_puts,
             "bs_calls": bs_calls, "bs_puts": bs_puts}

    return {
        "s0": s0, "m_log": m_log, "mu_arith": mu_arith, "sigma": sigma,
        "sigma_ann": sigma_ann, "var_ann": var_ann, "log_returns": log_returns,
        "heston": heston, "fan": fan, "vol_summary": vol_summary,
        "final_prices": final_prices, "final_values": final_values, "pnl": pnl,
        "gbm_pnl": gbm_pnl, "metrics": risk_metrics(pnl),
        "gbm_metrics": risk_metrics(gbm_pnl) if gbm_pnl is not None else None,
        "options": options, "curve": curve,
    }


# ---- Market charts ----------------------------------------------------------
def price_history_chart(prices, ticker):
    fig = go.Figure(
        go.Scatter(x=prices.index, y=prices.values, mode="lines",
                   line=dict(color=ACCENT, width=2), name="Close",
                   hovertemplate="%{y:,.2f}")
    )
    fig.update_yaxes(title="Price")
    return style(fig, f"{ticker} – Historical Adjusted Close", 380)


def paths_chart(fan, model_label):
    days = fan["days"]
    fig = go.Figure()
    for i in range(fan["samples"].shape[1]):
        fig.add_trace(go.Scattergl(
            x=days, y=fan["samples"][:, i], mode="lines",
            line=dict(width=1, color="rgba(0, 212, 255, 0.10)"),
            hoverinfo="skip", showlegend=False))

    for key, color, label in (("p5", RED, "5th percentile"),
                              ("p50", "#ffffff", "Median"),
                              ("p95", GREEN, "95th percentile")):
        fig.add_trace(go.Scatter(
            x=days, y=fan[key], mode="lines", line=dict(width=3, color=color),
            name=label, hovertemplate="%{y:,.2f}"))

    fig.update_xaxes(title="Trading days ahead")
    fig.update_yaxes(title="Simulated price")
    return style(fig, f"Simulated Price Paths – {model_label} "
                      f"(100 of {N_SIMULATIONS:,} shown)", 480)


def vol_paths_chart(vol, theta):
    days = vol["days"]
    fig = go.Figure()
    for i in range(vol["samples"].shape[1]):
        fig.add_trace(go.Scattergl(
            x=days, y=vol["samples"][:, i], mode="lines",
            line=dict(width=1, color="rgba(167, 139, 250, 0.15)"),
            hoverinfo="skip", showlegend=False))
    for key, color, label in (("p5", RED, "5th percentile"),
                              ("p50", "#ffffff", "Median"),
                              ("p95", GREEN, "95th percentile")):
        fig.add_trace(go.Scatter(
            x=days, y=vol[key], mode="lines", line=dict(width=3, color=color),
            name=label, hovertemplate="%{y:.1f}%"))
    fig.add_hline(y=float(np.sqrt(theta) * 100), line=dict(color=ORANGE, width=1.5, dash="dash"),
                  annotation_text="Long-run volatility √θ", annotation_font_color=ORANGE,
                  annotation_position="top right")
    fig.update_xaxes(title="Trading days ahead")
    fig.update_yaxes(title="Annualised volatility √v (%)", ticksuffix="%")
    return style(fig, "Heston Stochastic Volatility Paths", 450)


def distribution_chart(pnl, metrics, currency, compare_pnl=None, model_label="Simulated P&L"):
    fig = go.Figure()
    if compare_pnl is not None:
        fig.add_trace(go.Histogram(
            x=compare_pnl, nbinsx=100, name="GBM (constant volatility)",
            marker=dict(color="rgba(255, 166, 43, 0.35)", line=dict(width=0)),
            hovertemplate="P&L: %{x:,.0f}<br>Count: %{y}<extra></extra>"))
    fig.add_trace(go.Histogram(
        x=pnl, nbinsx=100, name=model_label,
        marker=dict(color=ACCENT, line=dict(width=0)), opacity=0.8,
        hovertemplate="P&L: %{x:,.0f}<br>Count: %{y}<extra></extra>"))

    for x, color, label in ((-metrics[95]["var"], ORANGE, "95% VaR"),
                            (-metrics[99]["var"], RED, "99% VaR"),
                            (-metrics[99]["cvar"], PINK, "99% CVaR")):
        fig.add_vline(x=x, line=dict(color=color, width=2, dash="dash"),
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="top")

    fig.update_xaxes(title=f"Profit / Loss ({currency})")
    fig.update_yaxes(title="Number of simulations")
    fig.update_layout(bargap=0.02, hovermode="closest", barmode="overlay")
    return style(fig, "Distribution of Simulated Profit & Loss", 450)


def option_curve_chart(curve, s0, strike, model_label):
    k = curve["strikes"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=curve["mc_calls"], mode="lines", name="Call (Monte Carlo)",
                             line=dict(color=GREEN, width=3), hovertemplate="%{y:,.2f}"))
    fig.add_trace(go.Scatter(x=k, y=curve["bs_calls"], mode="lines", name="Call (Black–Scholes)",
                             line=dict(color=GREEN, width=1.5, dash="dot"), hovertemplate="%{y:,.2f}"))
    fig.add_trace(go.Scatter(x=k, y=curve["mc_puts"], mode="lines", name="Put (Monte Carlo)",
                             line=dict(color=RED, width=3), hovertemplate="%{y:,.2f}"))
    fig.add_trace(go.Scatter(x=k, y=curve["bs_puts"], mode="lines", name="Put (Black–Scholes)",
                             line=dict(color=RED, width=1.5, dash="dot"), hovertemplate="%{y:,.2f}"))
    fig.add_vline(x=s0, line=dict(color="#ffffff", width=1, dash="dot"),
                  annotation_text="Spot", annotation_position="top left")
    fig.add_vline(x=strike, line=dict(color=ACCENT, width=2, dash="dash"),
                  annotation_text="Your strike", annotation_font_color=ACCENT,
                  annotation_position="top right")
    fig.update_xaxes(title="Strike price")
    fig.update_yaxes(title="Option price")
    return style(fig, f"Option Price vs Strike – {model_label}", 450)


def render_market_tab():
    st.caption(
        f"{N_SIMULATIONS:,} Monte Carlo paths · GBM or Heston stochastic volatility · "
        "95% / 99% VaR & CVaR · European option pricing · "
        "Market data from Yahoo Finance via yfinance (may be delayed)"
    )

    with st.form("market_form"):
        r1 = st.columns(4)
        ticker = r1[0].text_input(
            "Ticker symbol", value="AAPL", key="mkt_ticker",
            help="Any Yahoo Finance symbol, e.g. AAPL, TSLA, RELIANCE.NS, ^GSPC, BTC-USD",
        ).strip().upper()
        period = r1[1].selectbox(
            "Historical window", ["1y", "2y", "5y", "10y"], index=2, key="mkt_period")
        investment = r1[2].number_input(
            "Investment amount", min_value=100.0, value=10_000.0, step=1_000.0,
            format="%.2f", key="mkt_investment")
        horizon = r1[3].slider(
            "Horizon (trading days)", 1, MAX_HORIZON_DAYS, 21, key="mkt_horizon",
            help="21 ≈ 1 month, 63 ≈ 1 quarter, 252 ≈ 1 year. "
                 "Also used as the option's time to maturity.")

        r2 = st.columns([2, 2, 1, 1])
        model = r2[0].radio(
            "Price model", [MODEL_GBM, MODEL_HESTON], key="mkt_model",
            help="GBM assumes constant volatility. Heston lets volatility move randomly, "
                 "mean-revert and correlate with price moves (fatter tails, skew).")
        drift_mode = r2[1].radio(
            "Drift assumption (risk simulation)",
            ["Historical average return", "Zero drift (conservative)"],
            key="mkt_drift",
            help="Option pricing always uses the risk-free rate as drift (risk-neutral).")
        seed = int(r2[2].number_input(
            "Random seed", min_value=0, max_value=2_147_483_000, value=42, step=1,
            key="mkt_seed", help="Same seed = same results. Change it to re-sample."))
        r2[3].markdown("&nbsp;", unsafe_allow_html=True)
        r2[3].form_submit_button("🚀 Run", width="stretch")

        with st.expander("Heston model parameters (used when Heston is selected)"):
            h1 = st.columns(4)
            kappa = h1[0].number_input(
                "κ  Mean-reversion speed", min_value=0.01, max_value=20.0, value=2.0,
                step=0.1, format="%.2f", key="mkt_kappa",
                help="How fast variance is pulled back to its long-run level.")
            xi = h1[1].number_input(
                "ξ  Volatility of volatility", min_value=0.01, max_value=3.0, value=0.30,
                step=0.05, format="%.2f", key="mkt_xi",
                help="How violently the variance itself fluctuates.")
            rho = h1[2].number_input(
                "ρ  Price–vol correlation", min_value=-0.99, max_value=0.99, value=-0.70,
                step=0.05, format="%.2f", key="mkt_rho",
                help="Negative for stocks: volatility tends to rise when prices fall.")
            calibrate = h1[3].checkbox(
                "Calibrate v₀ and θ to history", value=True, key="mkt_calibrate",
                help="When ticked, initial and long-run variance are set from the "
                     "historical volatility of the chosen ticker.")
            h2 = st.columns(2)
            theta = h2[0].number_input(
                "θ  Long-run variance", min_value=0.0001, max_value=2.0, value=0.04,
                step=0.005, format="%.4f", key="mkt_theta",
                help="Annualised variance (0.04 = 20% volatility). Ignored if calibrating.")
            v0 = h2[1].number_input(
                "v₀  Initial variance", min_value=0.0001, max_value=2.0, value=0.04,
                step=0.005, format="%.4f", key="mkt_v0",
                help="Annualised variance today. Ignored if calibrating.")

        with st.expander("Option pricing settings"):
            o1 = st.columns(2)
            strike_in = o1[0].number_input(
                "Strike price (0 = at-the-money)", min_value=0.0, value=0.0, step=1.0,
                format="%.2f", key="mkt_strike",
                help="Leave at 0 to use today's price as the strike.")
            rf_pct = o1[1].number_input(
                "Risk-free rate (% per year)", min_value=-2.0, max_value=25.0, value=4.0,
                step=0.25, format="%.2f", key="mkt_rf",
                help="Used for discounting and as the risk-neutral drift.")

    if not ticker:
        st.warning("Please enter a ticker symbol.")
        return

    try:
        with st.spinner(f"Fetching data for {ticker}..."):
            prices = load_prices(ticker, period)
            name = load_name(ticker)
    except ValueError as err:
        st.error(str(err))
        return

    _, sigma_check, _ = estimate_parameters(prices)
    if sigma_check <= 0 or np.isnan(sigma_check):
        st.error("Volatility could not be estimated from this data (prices are flat).")
        return

    cfg = {
        "model": model, "drift_mode": drift_mode, "investment": float(investment),
        "horizon": int(horizon), "seed": seed, "kappa": float(kappa), "xi": float(xi),
        "rho": float(rho), "theta": float(theta), "v0": float(v0),
        "calibrate": bool(calibrate), "strike": float(strike_in), "rf": rf_pct / 100.0,
    }
    key = (ticker, period, str(prices.index[-1]), float(prices.iloc[-1]), len(prices),
           tuple(sorted(cfg.items())))
    with st.spinner("Running Monte Carlo simulations..."):
        res = memo("mkt_results", key, lambda: build_market_results(prices, cfg))

    s0, sigma, sigma_ann = res["s0"], res["sigma"], res["sigma_ann"]
    metrics, options, heston = res["metrics"], res["options"], res["heston"]
    is_heston = model == MODEL_HESTON
    model_short = "Heston" if is_heston else "GBM"

    # ---- Header info --------------------------------------------------------
    st.subheader(f"{name} ({ticker}) · {model_short} model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last price", f"{s0:,.2f}")
    c2.metric("Annualised return (hist.)", f"{res['mu_arith'] * TRADING_DAYS:.2%}")
    c3.metric("Annualised volatility (hist.)", f"{sigma_ann:.2%}")
    c4.metric("Data points used", f"{len(prices):,}")

    if is_heston:
        feller_ok = 2 * heston["kappa"] * heston["theta"] > heston["xi"] ** 2
        st.caption(
            f"Heston parameters in use: κ = {heston['kappa']:.2f}, θ = {heston['theta']:.4f} "
            f"(long-run vol {np.sqrt(heston['theta']):.1%}), ξ = {heston['xi']:.2f}, "
            f"ρ = {heston['rho']:.2f}, v₀ = {heston['v0']:.4f} "
            f"(current vol {np.sqrt(heston['v0']):.1%}). "
            f"Feller condition (2κθ > ξ²): {'satisfied ✅' if feller_ok else 'not satisfied ⚠️'}"
        )
        if not feller_ok:
            st.info(
                "The Feller condition is not satisfied, so the variance can touch zero. "
                "The simulation uses full-truncation Euler, which handles this safely."
            )

    # ---- Risk metrics -------------------------------------------------------
    investment = cfg["investment"]
    st.markdown(f"### Risk over {horizon} trading day(s) on {investment:,.2f}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("95% VaR", f"{metrics[95]['var']:,.2f}",
              f"{-metrics[95]['var'] / investment:.2%} of investment", delta_color="off")
    m2.metric("99% VaR", f"{metrics[99]['var']:,.2f}",
              f"{-metrics[99]['var'] / investment:.2%} of investment", delta_color="off")
    m3.metric("95% CVaR", f"{metrics[95]['cvar']:,.2f}",
              f"{-metrics[95]['cvar'] / investment:.2%} of investment", delta_color="off")
    m4.metric("99% CVaR", f"{metrics[99]['cvar']:,.2f}",
              f"{-metrics[99]['cvar'] / investment:.2%} of investment", delta_color="off")

    st.caption(
        f"**Reading this:** with 95% confidence, the simulated loss over {horizon} day(s) "
        f"should not exceed **{metrics[95]['var']:,.2f}**. If losses do exceed that level, "
        f"the average loss in that worst 5% of scenarios is **{metrics[95]['cvar']:,.2f}** (CVaR)."
    )

    # ---- Option pricing headline -------------------------------------------
    st.markdown(f"### European option prices (strike {options['strike']:,.2f}, "
                f"maturity {horizon} trading days)")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Call price (Monte Carlo)", f"{options['call']:,.4f}",
              f"± {1.96 * options['call_se']:.4f} (95% CI)", delta_color="off")
    o2.metric("Put price (Monte Carlo)", f"{options['put']:,.4f}",
              f"± {1.96 * options['put_se']:.4f} (95% CI)", delta_color="off")
    o3.metric("Call – Black–Scholes", f"{options['bs_call']:,.4f}",
              help="Flat-volatility benchmark using historical volatility.")
    o4.metric("Put – Black–Scholes", f"{options['bs_put']:,.4f}",
              help="Flat-volatility benchmark using historical volatility.")

    # ---- Charts -------------------------------------------------------------
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📈 Simulated paths", "📊 P&L distribution", "🌪️ Volatility paths",
        "💰 Option pricing", "🕰️ Price history", "📋 Details & export",
    ])

    with t1:
        show(paths_chart(res["fan"], model_short))

    with t2:
        show(distribution_chart(
            res["pnl"], metrics, "currency units",
            compare_pnl=res["gbm_pnl"], model_label=f"{model_short} P&L"))
        if is_heston:
            g, h = res["gbm_metrics"], metrics
            cmp_df = pd.DataFrame({
                "Metric": ["95% VaR", "99% VaR", "95% CVaR", "99% CVaR"],
                "GBM": [f"{g[95]['var']:,.2f}", f"{g[99]['var']:,.2f}",
                        f"{g[95]['cvar']:,.2f}", f"{g[99]['cvar']:,.2f}"],
                "Heston": [f"{h[95]['var']:,.2f}", f"{h[99]['var']:,.2f}",
                           f"{h[95]['cvar']:,.2f}", f"{h[99]['cvar']:,.2f}"],
            })
            st.markdown("**Model comparison** (same inputs, same drift)")
            st.table(cmp_df.set_index("Metric"))

    with t3:
        if is_heston:
            show(vol_paths_chart(res["vol_summary"], heston["theta"]))
            st.caption("Each faint line is one simulated volatility path. Volatility "
                       "is pulled back towards its long-run level at speed κ.")
        else:
            st.info("GBM assumes constant volatility, so there are no volatility paths to show. "
                    "Switch the price model to **Heston** to see stochastic volatility.")

    with t4:
        show(option_curve_chart(res["curve"], s0, options["strike"], model_short))
        st.caption(
            "Option prices are computed under the **risk-neutral measure**: the simulation "
            "is re-run with the risk-free rate as drift, then payoffs are averaged and "
            "discounted. Dotted lines are the Black–Scholes benchmark with historical "
            "volatility; under Heston the gap reflects stochastic volatility and skew."
        )

    with t5:
        show(price_history_chart(prices, ticker))

    with t6:
        lr = res["log_returns"].values
        summary = pd.DataFrame({
            "Metric": [
                "Price model", "Starting price", "Expected final price (mean)",
                "Median final price", "Best case (max)", "Worst case (min)",
                "Probability of loss", "Expected P&L", "95% VaR", "99% VaR",
                "95% CVaR", "99% CVaR", "Historical skewness (daily returns)",
                "Historical excess kurtosis (daily returns)",
                "Risk-neutral martingale check (should be ≈ 1)",
                "Put–call parity residual (should be ≈ 0)",
            ],
            "Value": [
                model, f"{s0:,.2f}", f"{res['final_prices'].mean():,.2f}",
                f"{np.median(res['final_prices']):,.2f}",
                f"{res['final_prices'].max():,.2f}", f"{res['final_prices'].min():,.2f}",
                f"{(res['pnl'] < 0).mean():.2%}", f"{res['pnl'].mean():,.2f}",
                f"{metrics[95]['var']:,.2f}", f"{metrics[99]['var']:,.2f}",
                f"{metrics[95]['cvar']:,.2f}", f"{metrics[99]['cvar']:,.2f}",
                f"{sp_stats.skew(lr):.3f}", f"{sp_stats.kurtosis(lr):.3f}",
                f"{options['martingale_ratio']:.4f}", f"{options['parity_gap']:.4f}",
            ],
        })
        st.table(summary.set_index("Metric"))

        export = pd.DataFrame({
            "final_price": res["final_prices"],
            "final_value": res["final_values"],
            "profit_loss": res["pnl"],
        })
        st.download_button(
            "⬇️ Download simulation results (CSV)",
            data=export.to_csv(index_label="simulation").encode("utf-8"),
            file_name=f"{ticker}_{model_short.lower()}_monte_carlo_{horizon}d.csv",
            mime="text/csv", key="mkt_download",
        )

    with st.expander("ℹ️ Methodology & limitations"):
        st.markdown(
            f"""
**GBM.** `S(t+1) = S(t)·exp[(μ − ½σ²) + σ·Z]`, with daily σ = {sigma:.5f} and arithmetic
daily drift μ = {res['mu_arith']:.5f} estimated from the selected history.

**Heston.** Price and variance follow
`dS = μS dt + √v S dW₁` and `dv = κ(θ − v) dt + ξ√v dW₂`, with `corr(dW₁, dW₂) = ρ`.
Correlated shocks are built as `Z₂ = ρ·Z₁ + √(1−ρ²)·Z⊥`. Variance is discretised with
the full-truncation Euler scheme on daily steps.

**VaR / CVaR.** VaR is the loss not exceeded with the stated confidence over the horizon;
CVaR is the average loss in the scenarios worse than VaR.

**Option pricing.** Prices of European options equal the discounted average payoff over
risk-neutral terminal prices (drift = risk-free rate). No dividends are modelled. The
± figures are 95% Monte Carlo confidence intervals.

**Limitations.** Heston parameters are not calibrated to option prices (only v₀ and θ
can be tied to historical variance), jumps are not modelled, and past data does not
guarantee future behaviour. For education and analysis only; **not financial advice**.
"""
        )


# ============================================================================
# TAB 2 - ENTERPRISE CASH-FLOW STRESS TEST
# ============================================================================
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
                "(6% + 3%), demand softens (-8% revenue, then -0.5% per month) and costs creep up.",
    },
}
DEFAULT_SCENARIO = "Baseline (Normal)"


def apply_scenario():
    """Callback: copy the chosen scenario's defaults into the input widgets."""
    p = SCENARIOS[st.session_state["ent_scenario"]]
    st.session_state["ent_vol"] = float(p["vol"])
    st.session_state["ent_var_ratio"] = float(p["var_ratio"])
    st.session_state["ent_shock"] = float(p["shock"])
    st.session_state["ent_growth"] = float(p["growth"])
    st.session_state["ent_rate"] = float(p["rate"])
    revenue = float(st.session_state.get("ent_revenue", 100_000.0))
    st.session_state["ent_fixed"] = float(round(revenue * p["fixed_pct"] / 100.0, 2))


def init_enterprise_state():
    if "ent_initialized" in st.session_state:
        return
    st.session_state["ent_scenario"] = DEFAULT_SCENARIO
    st.session_state["ent_revenue"] = 100_000.0
    st.session_state["ent_opening"] = 40_000.0
    st.session_state["ent_debt"] = 400_000.0
    st.session_state["ent_trials"] = ENT_TRIAL_OPTIONS[0]
    st.session_state["ent_seed"] = 42
    apply_scenario()
    st.session_state["ent_initialized"] = True


def simulate_cash_flows(revenue, fixed, var_ratio, vol, opening, shock, growth,
                        debt, rate, months=ENT_MONTHS, n_paths=5_000, seed=None):
    """
    Monte Carlo simulation of month-end cash balances.

    Expected revenue:  E[R_t] = revenue * (1 + shock) * (1 + growth)^(t-1)
    Random revenue:    R_t = E[R_t] * exp(vol*Z - 0.5*vol^2),  Z ~ N(0,1), i.i.d. each month
                       (log-normal: never negative, mean equals E[R_t])
    Monthly cash flow: R_t * (1 - var_ratio) - fixed - debt * rate / 12
    Cash balance:      Cash_t = opening + sum of monthly cash flows

    Returns an array of shape (months + 1, n_paths); row 0 is the opening cash.
    """
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


# ---- Enterprise charts ------------------------------------------------------
def shortfall_probability_chart(prob_by_month):
    months = np.arange(1, len(prob_by_month) + 1)
    pct = prob_by_month * 100
    fig = go.Figure(go.Scatter(
        x=months, y=pct, mode="lines+markers",
        line=dict(color=RED, width=3), marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(255, 75, 110, 0.18)",
        name="Cumulative probability", hovertemplate="%{y:.2f}%"))
    fig.add_hline(y=5, line=dict(color=ORANGE, width=1.5, dash="dash"),
                  annotation_text="5% tolerance", annotation_font_color=ORANGE,
                  annotation_position="bottom right")
    fig.update_xaxes(title="Month", dtick=1)
    fig.update_yaxes(title="Probability of insolvency by this month (%)",
                     range=[0, max(10.0, float(pct.max()) * 1.15)], ticksuffix="%")
    return style(fig, "Probability of Working Capital Shortfall (cumulative)", 430)


def cash_fan_chart(cash, n_show=50):
    months = np.arange(cash.shape[0])
    fig = go.Figure()
    for i in range(min(n_show, cash.shape[1])):
        fig.add_trace(go.Scatter(
            x=months, y=cash[:, i], mode="lines",
            line=dict(width=1, color="rgba(0, 212, 255, 0.10)"),
            hoverinfo="skip", showlegend=False))

    p = {q: np.percentile(cash, q, axis=1) for q in (5, 25, 50, 75, 95)}
    fig.add_trace(go.Scatter(x=months, y=p[5], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months, y=p[95], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(0, 212, 255, 0.15)",
                             name="5th–95th percentile", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months, y=p[25], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months, y=p[75], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(0, 212, 255, 0.30)",
                             name="25th–75th percentile", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months, y=p[50], mode="lines",
                             line=dict(width=3, color="#ffffff"), name="Median",
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=months, y=p[5], mode="lines",
                             line=dict(width=2, color=RED, dash="dot"),
                             name="5th percentile", hovertemplate="%{y:,.0f}"))
    fig.add_hline(y=0, line=dict(color=RED, width=2, dash="dash"),
                  annotation_text="Cash = 0 (insolvency)", annotation_font_color=RED,
                  annotation_position="bottom right")
    fig.update_xaxes(title="Month", dtick=1)
    fig.update_yaxes(title="Cash balance / working capital")
    return style(fig, f"Simulated Cash Balance (50 of {cash.shape[1]:,} paths shown)", 480)


def ending_cash_chart(stats):
    fig = go.Figure(go.Histogram(
        x=stats["ending"], nbinsx=80, marker=dict(color=ACCENT, line=dict(width=0)),
        opacity=0.85, name="Ending cash",
        hovertemplate="Cash: %{x:,.0f}<br>Paths: %{y}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=RED, width=2, dash="dash"),
                  annotation_text="Cash = 0", annotation_font_color=RED,
                  annotation_position="top")
    fig.add_vline(x=stats["mean_end"], line=dict(color=GREEN, width=2, dash="dot"),
                  annotation_text="Mean", annotation_font_color=GREEN,
                  annotation_position="top right")
    fig.add_vline(x=stats["p5_end"], line=dict(color=ORANGE, width=2, dash="dot"),
                  annotation_text="5th pct", annotation_font_color=ORANGE,
                  annotation_position="top left")
    fig.update_xaxes(title="Cash balance at end of month 12")
    fig.update_yaxes(title="Number of trials")
    fig.update_layout(bargap=0.02, hovermode="closest")
    return style(fig, "Distribution of Month-12 Ending Cash", 420)


def scenario_comparison_chart(rows, selected):
    names = [r["name"] for r in rows]
    probs = [r["prob"] * 100 for r in rows]
    colors = [RED if n == selected else ACCENT for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=probs, marker_color=colors,
        text=[f"{p:.2f}%" for p in probs], textposition="outside",
        hovertemplate="%{x}<br>Insolvency probability: %{y:.2f}%<extra></extra>"))
    fig.update_yaxes(title="Insolvency probability (%)", ticksuffix="%",
                     range=[0, max(10.0, max(probs) * 1.2)])
    fig.update_layout(hovermode="closest", showlegend=False)
    return style(fig, "Insolvency Probability by Macro Scenario "
                      "(your revenue, cash and debt)", 430)


# ---- PDF report -------------------------------------------------------------
NAVY = (15, 23, 42)
TEAL = (0, 150, 190)
GREY = (100, 110, 125)
LIGHT = (241, 245, 249)
RED_RGB = (220, 60, 90)
AMBER_RGB = (235, 150, 20)
GREEN_RGB = (30, 160, 110)
BLUE_RGB = (30, 144, 200)


def pdf_safe(text):
    """The built-in PDF fonts only support Latin-1, so replace anything else."""
    table = {
        "–": "-", "—": "-", "−": "-", "≈": "~", "’": "'", "‘": "'", "“": '"',
        "”": '"', "•": "-", "×": "x", "σ": "sigma", "μ": "mu", "≥": ">=",
        "≤": "<=", "→": "->", "√": "sqrt", "½": "1/2",
    }
    text = str(text)
    for old, new in table.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(
            0, 6,
            f"Monte Carlo Risk Engine  |  Educational and analytical use only  |  "
            f"Page {self.page_no()}",
            align="C")


def _ensure_space(pdf, height):
    if pdf.get_y() + height > pdf.h - pdf.b_margin:
        pdf.add_page()


def _section(pdf, title):
    _ensure_space(pdf, 24)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, pdf_safe(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.6)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.l_margin + pdf.epw, y)
    pdf.ln(3)


def _paragraph(pdf, text, size=9, color=NAVY, height=4.8, bold=False):
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, height, pdf_safe(text), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _kv_table(pdf, rows, widths=(50, 40, 50, 40)):
    for i, row in enumerate(rows):
        _ensure_space(pdf, 8)
        fill = i % 2 == 0
        pdf.set_fill_color(*LIGHT)
        for j, (text, w) in enumerate(zip(row, widths)):
            is_label = j % 2 == 0
            pdf.set_font("Helvetica", "B" if is_label else "", 9)
            pdf.set_text_color(*(GREY if is_label else NAVY))
            last = j == len(row) - 1
            pdf.cell(w, 7, pdf_safe(text), border=0, fill=fill,
                     new_x=XPos.LMARGIN if last else XPos.RIGHT,
                     new_y=YPos.NEXT if last else YPos.TOP)


def _kpi_card(pdf, x, y, w, h, label, value, note, accent):
    pdf.set_fill_color(*LIGHT)
    pdf.set_draw_color(220, 226, 232)
    pdf.set_line_width(0.2)
    pdf.rect(x, y, w, h, style="DF")
    pdf.set_fill_color(*accent)
    pdf.rect(x, y, 2, h, style="F")

    pdf.set_xy(x + 5, y + 3)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(w - 8, 3.6, pdf_safe(label), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(x + 5, y + 12)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*NAVY)
    pdf.cell(w - 8, 9, pdf_safe(value))

    pdf.set_xy(x + 5, y + 22)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*GREY)
    pdf.cell(w - 8, 4, pdf_safe(note))


def _draw_histogram(pdf, data, stats, x, y, w, h, bins=40):
    counts, edges = np.histogram(data, bins=bins)
    lo, hi = float(edges[0]), float(edges[-1])
    span = hi - lo if hi > lo else 1.0
    peak = max(int(counts.max()), 1)
    bar_w = w / bins

    pdf.set_draw_color(180, 186, 195)
    pdf.set_line_width(0.2)
    pdf.line(x, y + h, x + w, y + h)

    for i, c in enumerate(counts):
        bar_h = (h - 2) * c / peak
        mid = 0.5 * (edges[i] + edges[i + 1])
        pdf.set_fill_color(*(RED_RGB if mid < 0 else BLUE_RGB))
        if bar_h > 0:
            pdf.rect(x + i * bar_w, y + h - bar_h, bar_w * 0.92, bar_h, style="F")

    def marker(value, color, dashed=True):
        if value < lo or value > hi:
            return
        mx = x + (value - lo) / span * w
        pdf.set_draw_color(*color)
        pdf.set_line_width(0.5)
        if dashed:
            pdf.set_dash_pattern(dash=1.5, gap=1.2)
        pdf.line(mx, y - 1, mx, y + h)
        pdf.set_dash_pattern()

    marker(0.0, RED_RGB)
    marker(stats["mean_end"], GREEN_RGB)
    marker(stats["p5_end"], AMBER_RGB)

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*GREY)
    low_txt, high_txt = f"{lo:,.0f}", f"{hi:,.0f}"
    pdf.text(x, y + h + 4, low_txt)
    pdf.text(x + w - pdf.get_string_width(high_txt), y + h + 4, high_txt)
    mid_txt = "Ending cash at month 12 (currency units)"
    pdf.text(x + (w - pdf.get_string_width(mid_txt)) / 2, y + h + 4, mid_txt)

    # legend
    ly = y + h + 8
    items = [(RED_RGB, "Cash = 0 (insolvency line)"), (GREEN_RGB, "Mean ending cash"),
             (AMBER_RGB, "5th percentile")]
    lx = x
    for color, label in items:
        pdf.set_fill_color(*color)
        pdf.rect(lx, ly, 3, 3, style="F")
        pdf.text(lx + 4.5, ly + 2.6, pdf_safe(label))
        lx += 8 + pdf.get_string_width(pdf_safe(label)) + 6


def build_pdf_report(ctx):
    """Create the executive PDF and return it as bytes."""
    s = ctx["stats"]
    p = ctx["params"]
    prob_pct = s["prob"] * 100

    if s["prob"] < 0.05:
        rating, rating_color = "LOW", GREEN_RGB
    elif s["prob"] < 0.20:
        rating, rating_color = "MODERATE", AMBER_RGB
    else:
        rating, rating_color = "HIGH", RED_RGB

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_title("Executive Risk Report - Enterprise Cash-Flow Stress Test")
    pdf.set_author("Monte Carlo Risk Engine")
    pdf.add_page()

    # ---- Title band ---------------------------------------------------------
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 34, style="F")
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 34, pdf.w, 1.4, style="F")
    pdf.set_xy(15, 9)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, "Executive Risk Report")
    pdf.set_xy(15, 19)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(190, 205, 220)
    pdf.cell(0, 6, "Enterprise Cash-Flow Stress Test  |  Monte Carlo Simulation")
    pdf.set_xy(15, 26)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 5, f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
                   f"{s['n']:,} trials  |  {ENT_MONTHS}-month horizon")
    pdf.set_y(42)

    # ---- 1. Scenario --------------------------------------------------------
    _section(pdf, "1. Macroeconomic Stress Scenario")
    _paragraph(pdf, ctx["scenario"], size=11, bold=True)
    pdf.ln(1)
    _paragraph(pdf, ctx["description"], size=9, color=GREY)
    pdf.ln(2)

    def money(v):
        return f"{v:,.0f}"

    _kv_table(pdf, [
        ("Monthly revenue (base)", money(p["revenue"]), "Revenue volatility", f"{p['vol']:.1f}%"),
        ("Revenue shock", f"{p['shock']:+.1f}%", "Monthly revenue drift", f"{p['growth']:+.2f}%"),
        ("Variable cost ratio", f"{p['var_ratio']:.1f}%", "Fixed overheads (monthly)", money(p["fixed"])),
        ("Debt outstanding", money(p["debt"]), "Interest rate on debt", f"{p['rate']:.2f}%"),
        ("Opening working capital", money(p["opening"]), "Simulation trials", f"{s['n']:,}"),
    ])

    # ---- 2. Key outputs -----------------------------------------------------
    _section(pdf, "2. Key Quantitative Outputs")
    _ensure_space(pdf, 36)
    top = pdf.get_y() + 1
    gap = 6
    card_w = (pdf.epw - 2 * gap) / 3
    x0 = pdf.l_margin
    _kpi_card(pdf, x0, top, card_w, 30, "MEAN ENDING CASH (MONTH 12)",
              money(s["mean_end"]), f"Median: {money(s['median_end'])}", BLUE_RGB)
    _kpi_card(pdf, x0 + card_w + gap, top, card_w, 30,
              "5TH PERCENTILE CASH (SEVERE DOWNSIDE)", money(s["p5_end"]),
              f"Avg. of worst 5%: {money(s['es5_end'])}", AMBER_RGB)
    _kpi_card(pdf, x0 + 2 * (card_w + gap), top, card_w, 30,
              "INSOLVENCY PROBABILITY (12 MONTHS)", f"{prob_pct:.2f}%",
              f"95% CI: {s['ci_low'] * 100:.2f}% - {s['ci_high'] * 100:.2f}%", rating_color)
    pdf.set_xy(pdf.l_margin, top + 34)

    _paragraph(
        pdf,
        f"Risk rating: {rating}. In {s['insolvent_paths']:,} of {s['n']:,} simulated trials "
        f"({prob_pct:.2f}%) month-end cash fell below zero at least once within "
        f"{ENT_MONTHS} months. The probability of holding negative cash specifically at "
        f"month {ENT_MONTHS} is {s['prob_negative_at_month'][-1] * 100:.2f}%. "
        + (f"Where insolvency occurred, it first appeared on average in month "
           f"{s['avg_first_month']:.1f}. " if s["avg_first_month"] is not None else "")
        + f"Additional opening cash of about {money(s['buffer_95'])} would keep the balance "
          f"non-negative in 95% of trials.",
        size=9)

    # ---- 3. Distribution chart ---------------------------------------------
    _section(pdf, "3. Distribution of Month-12 Ending Cash")
    _ensure_space(pdf, 75)
    chart_y = pdf.get_y() + 4
    _draw_histogram(pdf, s["ending"], s, pdf.l_margin + 2, chart_y, pdf.epw - 4, 52)
    pdf.set_xy(pdf.l_margin, chart_y + 66)

    # ---- 4. Month-by-month --------------------------------------------------
    _section(pdf, "4. Month-by-Month Insolvency Profile")
    widths = (25, 55, 50, 50)
    heads = ("Month", "Cumulative insolvency prob.", "Median cash", "5th percentile cash")
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for j, (head, w) in enumerate(zip(heads, widths)):
        last = j == len(heads) - 1
        pdf.cell(w, 7, head, border=0, fill=True, align="C",
                 new_x=XPos.LMARGIN if last else XPos.RIGHT,
                 new_y=YPos.NEXT if last else YPos.TOP)
    for m in range(ENT_MONTHS):
        _ensure_space(pdf, 8)
        vals = (str(m + 1), f"{s['prob_by_month'][m] * 100:.2f}%",
                money(s["median_by_month"][m]), money(s["p5_by_month"][m]))
        pdf.set_fill_color(*LIGHT)
        pdf.set_text_color(*NAVY)
        pdf.set_font("Helvetica", "", 9)
        for j, (val, w) in enumerate(zip(vals, widths)):
            last = j == len(vals) - 1
            pdf.cell(w, 6.5, val, border=0, fill=(m % 2 == 0), align="C",
                     new_x=XPos.LMARGIN if last else XPos.RIGHT,
                     new_y=YPos.NEXT if last else YPos.TOP)

    # ---- 5. Methodology -----------------------------------------------------
    _section(pdf, "5. Methodology and Limitations")
    _paragraph(
        pdf,
        "Method. Monthly revenue is simulated with a log-normal distribution whose mean equals "
        "expected revenue after the scenario's demand shock and monthly drift, and whose "
        "dispersion is set by revenue volatility. Variable costs are a fixed share of revenue; "
        "fixed overheads and interest on floating-rate debt are paid every month. The cash "
        "balance is opening working capital plus the cumulative monthly cash flow. A trial is "
        "insolvent if its month-end cash balance is below zero in any month.\n\n"
        "Precision. Probabilities are Monte Carlo estimates (the share of trials); the 95% "
        "confidence interval shown is a Wilson score interval.\n\n"
        "Limitations. Monthly revenue shocks are independent, so seasonality, customer "
        "concentration, late payments, credit lines and management responses are not modelled. "
        "Scenario parameters are stylised illustrations, not official forecasts. This report is "
        "for educational and analytical purposes and is not financial advice.",
        size=8.5, color=GREY, height=4.4)

    return bytes(pdf.output())


# ---- Enterprise tab UI ------------------------------------------------------
def render_enterprise_tab():
    init_enterprise_state()

    st.caption(
        "Stress-test a business's working capital under macroeconomic scenarios. "
        "Insolvency = month-end cash balance below zero at any point in 12 months."
    )

    # Scenario selector (outside any form so it updates the inputs instantly)
    st.selectbox(
        "Macro scenario preset", list(SCENARIOS.keys()), key="ent_scenario",
        on_change=apply_scenario,
        help="Selecting a scenario resets revenue volatility, fixed overheads, variable "
             "cost ratio, revenue shock, revenue drift and the interest rate to that "
             "scenario's defaults. You can still edit every value afterwards.")
    scenario = st.session_state["ent_scenario"]
    st.info(f"**{scenario}:** {SCENARIOS[scenario]['desc']}  "
            "_(Stylised, illustrative parameters – not official forecasts.)_")

    c = st.columns(4)
    revenue = c[0].number_input(
        "Monthly Revenue (base)", min_value=0.0, step=5_000.0, format="%.2f",
        key="ent_revenue", help="Expected revenue per month before scenario effects.")
    fixed = c[1].number_input(
        "Fixed Overheads (monthly)", min_value=0.0, step=1_000.0, format="%.2f",
        key="ent_fixed", help="Rent, salaries, utilities and other costs that do not "
                              "change with sales.")
    var_ratio_pct = c[2].number_input(
        "Variable Cost Ratio (% of revenue)", min_value=0.0, max_value=99.0, step=1.0,
        format="%.1f", key="ent_var_ratio",
        help="Share of each unit of revenue spent on materials, commissions, etc.")
    vol_pct = c[3].number_input(
        "Revenue Volatility (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f",
        key="ent_vol", help="Typical month-to-month swing in revenue.")

    d = st.columns(4)
    opening = d[0].number_input(
        "Opening Working Capital", min_value=0.0, step=5_000.0, format="%.2f",
        key="ent_opening", help="Cash on hand at the start.")
    shock_pct = d[1].number_input(
        "Revenue Shock (% level change)", min_value=-90.0, max_value=100.0, step=1.0,
        format="%.1f", key="ent_shock",
        help="Immediate change in the revenue level from month 1 (negative = drop).")
    debt = d[2].number_input(
        "Floating-Rate Debt Outstanding", min_value=0.0, step=10_000.0, format="%.2f",
        key="ent_debt", help="Interest is paid monthly on this balance.")
    rate_pct = d[3].number_input(
        "Interest Rate on Debt (% per year)", min_value=0.0, max_value=50.0, step=0.25,
        format="%.2f", key="ent_rate")

    with st.expander("Simulation settings"):
        s1, s2, s3 = st.columns(3)
        trials = s1.selectbox(
            "Monte Carlo trials", ENT_TRIAL_OPTIONS, key="ent_trials",
            format_func=lambda n: f"{n:,}")
        growth_pct = s2.number_input(
            "Monthly revenue drift (%)", min_value=-20.0, max_value=20.0, step=0.25,
            format="%.2f", key="ent_growth",
            help="Underlying monthly growth (negative = shrinking demand).")
        seed = int(s3.number_input(
            "Random seed", min_value=0, max_value=2_147_483_000, step=1, key="ent_seed",
            help="Same seed = same results. Change it to re-sample."))

    params = {
        "revenue": float(revenue), "fixed": float(fixed),
        "var_ratio": float(var_ratio_pct), "vol": float(vol_pct), "opening": float(opening),
        "shock": float(shock_pct), "growth": float(growth_pct), "debt": float(debt),
        "rate": float(rate_pct),
    }
    key = (tuple(sorted(params.items())), int(trials), seed)

    def run():
        cash_paths = simulate_cash_flows(
            revenue, fixed, var_ratio_pct / 100.0, vol_pct / 100.0, opening,
            shock_pct / 100.0, growth_pct / 100.0, debt, rate_pct / 100.0,
            n_paths=int(trials), seed=seed)
        return cash_paths, cash_stats(cash_paths)

    with st.spinner(f"Running {int(trials):,} trials..."):
        cash, stats = memo("ent_results", key, run)

    expected_net = (revenue * (1 + shock_pct / 100.0) * (1 - var_ratio_pct / 100.0)
                    - fixed - debt * rate_pct / 100.0 / 12.0)
    if expected_net < 0 and growth_pct <= 0:
        st.warning(
            f"Expected monthly cash flow is negative ({expected_net:,.2f}) even before "
            "volatility: the business is burning cash under these assumptions.")

    # ---- Headline -----------------------------------------------------------
    prob = stats["prob"]
    headline = (f"Insolvency probability within 12 months: **{prob:.2%}** "
                f"({stats['insolvent_paths']:,} of {stats['n']:,} trials)")
    if prob < 0.05:
        st.success(f"🟢 Low risk. {headline}")
    elif prob < 0.20:
        st.warning(f"🟠 Moderate risk. {headline}")
    else:
        st.error(f"🔴 High risk. {headline}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Insolvency probability", f"{prob:.2%}",
              f"95% CI {stats['ci_low']:.2%} – {stats['ci_high']:.2%}", delta_color="off")
    m2.metric("Mean ending cash", f"{stats['mean_end']:,.0f}",
              f"Median {stats['median_end']:,.0f}", delta_color="off")
    m3.metric("5th percentile cash (severe downside)", f"{stats['p5_end']:,.0f}",
              f"Worst-5% average {stats['es5_end']:,.0f}", delta_color="off")
    m4.metric("Extra cash for 95% safety", f"{stats['buffer_95']:,.0f}",
              help="Additional opening cash that would keep the balance above zero in "
                   "95% of trials.")

    first = stats["avg_first_month"]
    st.caption(
        (f"Where insolvency occurs, the first shortfall happens on average in month {first:.1f}. "
         if first is not None else "No simulated trial ran out of cash. ")
        + f"Expected monthly cash flow is {expected_net:,.2f}. The percentage is a Monte Carlo "
          f"estimate with standard error ±{np.sqrt(prob * (1 - prob) / stats['n']):.2%}."
    )

    # ---- PDF export ---------------------------------------------------------
    if FPDF_AVAILABLE:
        try:
            pdf_bytes = build_pdf_report({
                "scenario": scenario, "description": SCENARIOS[scenario]["desc"],
                "params": params, "stats": stats,
            })
            st.download_button(
                "📄 Export Executive PDF Risk Report", data=pdf_bytes,
                file_name=f"Executive_Risk_Report_{scenario.split(' (')[0].replace(' ', '_')}.pdf",
                mime="application/pdf", key="ent_pdf_download")
        except Exception as exc:  # never let the report break the dashboard
            st.error(f"The PDF report could not be generated: {exc}")
    else:
        st.warning("PDF export needs the `fpdf2` package. Add `fpdf2` to requirements.txt.")

    # ---- Charts -------------------------------------------------------------
    t1, t2, t3, t4, t5 = st.tabs([
        "⚠️ Insolvency probability", "💵 Cash balance paths", "📊 Ending cash distribution",
        "🔀 Scenario comparison", "📋 Details & export",
    ])

    with t1:
        show(shortfall_probability_chart(stats["prob_by_month"]))
    with t2:
        show(cash_fan_chart(cash))
    with t3:
        show(ending_cash_chart(stats))
    with t4:
        comp_key = (float(revenue), float(opening), float(debt), int(trials), seed)
        rows = memo("ent_comparison", comp_key,
                    lambda: compare_scenarios(revenue, opening, debt, int(trials), seed))
        show(scenario_comparison_chart(rows, scenario))
        comp_df = pd.DataFrame({
            "Scenario": [r["name"] for r in rows],
            "Insolvency probability": [f"{r['prob']:.2%}" for r in rows],
            "Mean ending cash": [f"{r['mean_end']:,.0f}" for r in rows],
            "5th percentile cash": [f"{r['p5_end']:,.0f}" for r in rows],
        })
        st.table(comp_df.set_index("Scenario"))
        st.caption("Each scenario uses its own preset parameters applied to your revenue "
                   "size, opening cash and debt.")
    with t5:
        table = pd.DataFrame({
            "Month": np.arange(1, ENT_MONTHS + 1),
            "Cumulative insolvency probability": [f"{x:.2%}" for x in stats["prob_by_month"]],
            "Cash negative in this month": [f"{x:.2%}" for x in stats["prob_negative_at_month"]],
            "Median cash": [f"{x:,.0f}" for x in stats["median_by_month"]],
            "5th percentile cash": [f"{x:,.0f}" for x in stats["p5_by_month"]],
        })
        st.table(table.set_index("Month"))

        export = pd.DataFrame(
            cash[1:].T, columns=[f"month_{m}" for m in range(1, ENT_MONTHS + 1)])
        st.download_button(
            "⬇️ Download all simulated cash paths (CSV)",
            data=export.to_csv(index_label="trial").encode("utf-8"),
            file_name="enterprise_cash_flow_simulation.csv",
            mime="text/csv", key="ent_csv_download")

    with st.expander("ℹ️ Methodology & limitations"):
        st.markdown(
            f"""
**Revenue.** Each month's revenue is log-normal around its expected value
`revenue × (1 + shock) × (1 + drift)^(t−1)`, so it is never negative and its average equals
that expectation. Shocks are independent from month to month.

**Costs.** Variable costs equal the variable cost ratio times revenue. Fixed overheads and
interest on floating-rate debt (`debt × rate ÷ 12`) are paid every month.

**Insolvency.** A trial is insolvent if its month-end cash balance is below zero in any of
the {ENT_MONTHS} months. The headline probability is the share of trials that are insolvent;
the confidence interval is a Wilson score interval.

**Limitations.** No seasonality, credit lines, tax, customer concentration or management
response is modelled, and the scenario presets are stylised illustrations rather than
forecasts. For education and analysis only; **not financial advice**.
"""
        )


# ============================================================================
# MAIN
# ============================================================================
st.title("📉 Monte Carlo Risk Engine")
st.caption("Quantitative finance and enterprise risk suite")

tab_market, tab_enterprise = st.tabs(
    ["📈 Market Asset Risk", "🏢 Enterprise Cash-Flow Stress Test"]
)

with tab_market:
    render_market_tab()

with tab_enterprise:
    render_enterprise_tab()
