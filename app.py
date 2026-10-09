"""
Monte Carlo Risk Engine
-----------------------
Streamlit app that:
  * pulls live stock history with yfinance
  * runs 10,000 Geometric Brownian Motion (GBM) simulations with NumPy
  * reports 95% VaR, 99% VaR and CVaR (Expected Shortfall)
  * draws interactive dark-mode charts with Plotly

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
N_SIMULATIONS = 10_000
TRADING_DAYS = 252
MIN_HISTORY_POINTS = 60
PLOT_TEMPLATE = "plotly_dark"
ACCENT = "#00d4ff"
RED = "#ff4b6e"
ORANGE = "#ffa62b"
GREEN = "#2ee6a6"

st.set_page_config(
    page_title="Monte Carlo Risk Engine",
    page_icon="📉",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Simulation + risk maths
# ----------------------------------------------------------------------------
def estimate_parameters(prices: pd.Series):
    """Daily log-return mean and standard deviation."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return float(log_returns.mean()), float(log_returns.std(ddof=1)), log_returns


def simulate_gbm(s0, mu, sigma, horizon, n_sims, seed=None):
    """
    Vectorised Geometric Brownian Motion with daily time step (dt = 1 day):

        S(t+1) = S(t) * exp( (mu - 0.5*sigma^2) + sigma * Z ),   Z ~ N(0, 1)

    Here mu and sigma are the *daily* mean and std of log returns.
    Returns an array of shape (horizon + 1, n_sims); row 0 is the start price.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((horizon, n_sims))
    increments = (mu - 0.5 * sigma**2) + sigma * z
    log_paths = np.cumsum(increments, axis=0)
    paths = np.empty((horizon + 1, n_sims))
    paths[0] = s0
    paths[1:] = s0 * np.exp(log_paths)
    return paths


def risk_metrics(pnl: np.ndarray) -> dict:
    """VaR and CVaR from simulated profit/loss. Results are positive loss numbers."""
    out = {}
    for level in (95, 99):
        var = -np.percentile(pnl, 100 - level)
        tail = pnl[pnl <= -var]
        cvar = -tail.mean() if tail.size else var
        out[level] = {"var": float(var), "cvar": float(cvar)}
    return out


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def style(fig, title, height=450):
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


def price_history_chart(prices, ticker):
    fig = go.Figure(
        go.Scatter(x=prices.index, y=prices.values, mode="lines",
                   line=dict(color=ACCENT, width=2), name="Close",
                   hovertemplate="%{y:,.2f}")
    )
    fig.update_yaxes(title="Price")
    return style(fig, f"{ticker} – Historical Adjusted Close", 380)


def paths_chart(paths, horizon, n_show=100):
    days = np.arange(horizon + 1)
    fig = go.Figure()

    for i in range(min(n_show, paths.shape[1])):
        fig.add_trace(go.Scattergl(
            x=days, y=paths[:, i], mode="lines",
            line=dict(width=1, color="rgba(0, 212, 255, 0.10)"),
            hoverinfo="skip", showlegend=False))

    bands = [(5, RED, "5th percentile"), (50, "#ffffff", "Median"),
             (95, GREEN, "95th percentile")]
    for q, color, label in bands:
        fig.add_trace(go.Scatter(
            x=days, y=np.percentile(paths, q, axis=1), mode="lines",
            line=dict(width=3, color=color), name=label,
            hovertemplate="%{y:,.2f}"))

    fig.update_xaxes(title="Trading days ahead")
    fig.update_yaxes(title="Simulated price")
    return style(fig, f"Simulated Price Paths (100 of {paths.shape[1]:,} shown)", 480)


def distribution_chart(pnl, metrics, currency):
    fig = go.Figure(go.Histogram(
        x=pnl, nbinsx=100, marker=dict(color=ACCENT, line=dict(width=0)),
        opacity=0.85, name="Simulated P&L",
        hovertemplate="P&L: %{x:,.0f}<br>Count: %{y}<extra></extra>"))

    lines = [
        (-metrics[95]["var"], ORANGE, "95% VaR"),
        (-metrics[99]["var"], RED, "99% VaR"),
        (-metrics[99]["cvar"], "#ff00aa", "99% CVaR"),
    ]
    for x, color, label in lines:
        fig.add_vline(x=x, line=dict(color=color, width=2, dash="dash"),
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="top")

    fig.update_xaxes(title=f"Profit / Loss ({currency})")
    fig.update_yaxes(title="Number of simulations")
    fig.update_layout(bargap=0.02, hovermode="closest")
    return style(fig, "Distribution of Simulated Profit & Loss", 450)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("📉 Monte Carlo Risk Engine")
st.caption(
    f"Geometric Brownian Motion · {N_SIMULATIONS:,} simulations · "
    "95% / 99% Value-at-Risk & Expected Shortfall (CVaR)"
)

with st.sidebar:
    st.header("⚙️ Settings")
    with st.form("settings"):
        ticker = st.text_input("Ticker symbol", value="AAPL",
                               help="Any Yahoo Finance symbol, e.g. AAPL, TSLA, "
                                    "RELIANCE.NS, ^GSPC, BTC-USD").strip().upper()
        period = st.selectbox("Historical window for estimation",
                              ["1y", "2y", "5y", "10y"], index=2)
        investment = st.number_input("Investment amount", min_value=100.0,
                                     value=10_000.0, step=1_000.0, format="%.2f")
        horizon = st.slider("Time horizon (trading days)", 1, 504, 21,
                            help="21 ≈ 1 month, 63 ≈ 1 quarter, 252 ≈ 1 year")
        drift_mode = st.radio(
            "Drift assumption",
            ["Historical average return", "Zero drift (conservative)"],
            help="Zero drift ignores past average returns and models only volatility.")
        seed_text = st.text_input("Random seed (optional)", value="",
                                  help="Enter a whole number for reproducible results.")
        st.form_submit_button("🚀 Run simulation", width="stretch")

    st.info(f"Number of simulations is fixed at **{N_SIMULATIONS:,}**.")
    st.caption("Market data from Yahoo Finance via yfinance (may be delayed).")

if not ticker:
    st.warning("Please enter a ticker symbol.")
    st.stop()

# Parse seed safely
seed = None
if seed_text.strip():
    try:
        seed = int(seed_text.strip())
    except ValueError:
        st.sidebar.error("Seed must be a whole number. Using a random seed instead.")

try:
    with st.spinner(f"Fetching data for {ticker}..."):
        prices = load_prices(ticker, period)
        name = load_name(ticker)
except ValueError as err:
    st.error(str(err))
    st.stop()

mu, sigma, log_returns = estimate_parameters(prices)
if sigma <= 0 or np.isnan(sigma):
    st.error("Volatility could not be estimated from this data (prices are flat).")
    st.stop()

mu_used = mu if drift_mode.startswith("Historical") else 0.0
s0 = float(prices.iloc[-1])

with st.spinner(f"Running {N_SIMULATIONS:,} simulations..."):
    paths = simulate_gbm(s0, mu_used, sigma, horizon, N_SIMULATIONS, seed)

final_prices = paths[-1]
final_values = investment * final_prices / s0
pnl = final_values - investment
metrics = risk_metrics(pnl)

# ---- Header info ------------------------------------------------------------
st.subheader(f"{name} ({ticker})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last price", f"{s0:,.2f}")
c2.metric("Annualised return (hist.)", f"{mu * TRADING_DAYS:.2%}")
c3.metric("Annualised volatility", f"{sigma * np.sqrt(TRADING_DAYS):.2%}")
c4.metric("Data points used", f"{len(prices):,}")

# ---- Risk metrics ------------------------------------------------------------
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

# ---- Charts ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Simulated paths", "📊 P&L distribution", "🕰️ Price history", "📋 Details & export"]
)

with tab1:
    st.plotly_chart(paths_chart(paths, horizon), width="stretch", theme=None)

with tab2:
    st.plotly_chart(distribution_chart(pnl, metrics, "currency units"),
                    width="stretch", theme=None)

with tab3:
    st.plotly_chart(price_history_chart(prices, ticker), width="stretch", theme=None)

with tab4:
    summary = pd.DataFrame({
        "Metric": [
            "Starting price", "Expected final price (mean)", "Median final price",
            "Best case (max)", "Worst case (min)", "Probability of loss",
            "Expected P&L", "95% VaR", "99% VaR", "95% CVaR", "99% CVaR",
        ],
        "Value": [
            f"{s0:,.2f}", f"{final_prices.mean():,.2f}", f"{np.median(final_prices):,.2f}",
            f"{final_prices.max():,.2f}", f"{final_prices.min():,.2f}",
            f"{(pnl < 0).mean():.2%}", f"{pnl.mean():,.2f}",
            f"{metrics[95]['var']:,.2f}", f"{metrics[99]['var']:,.2f}",
            f"{metrics[95]['cvar']:,.2f}", f"{metrics[99]['cvar']:,.2f}",
        ],
    })
    st.table(summary.set_index("Metric"))

    export = pd.DataFrame({
        "final_price": final_prices,
        "final_value": final_values,
        "profit_loss": pnl,
    })
    st.download_button(
        "⬇️ Download simulation results (CSV)",
        data=export.to_csv(index_label="simulation").encode("utf-8"),
        file_name=f"{ticker}_monte_carlo_{horizon}d.csv",
        mime="text/csv",
    )

with st.expander("ℹ️ Methodology & limitations"):
    st.markdown(
        f"""
**Model.** Daily log returns are assumed to be normally distributed. Prices follow
Geometric Brownian Motion:

`S(t+1) = S(t) · exp[(μ − ½σ²) + σ·Z]`, with `Z ~ N(0,1)`.

μ and σ are estimated from the selected historical window
(μ = {mu:.5f}/day, σ = {sigma:.5f}/day).

**Definitions.**
- **VaR (Value at Risk):** the loss that should not be exceeded with the stated
  confidence over the horizon.
- **CVaR / Expected Shortfall:** the average loss in the scenarios that are worse than VaR.

**Limitations.** GBM assumes constant volatility and normally distributed returns, so it
tends to understate the chance of extreme crashes ("fat tails"). Past data does not
guarantee future behaviour. This tool is for education and analysis only and is
**not financial advice**.
"""
    )
