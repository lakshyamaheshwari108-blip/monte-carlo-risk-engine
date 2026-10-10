"""
Monte Carlo Risk Engine
-----------------------
Tab 1  Market Asset Risk
       * live stock history via yfinance
       * 10,000 Geometric Brownian Motion simulations (NumPy)
       * 95% VaR, 99% VaR and CVaR (Expected Shortfall)

Tab 2  Enterprise Cash-Flow Stress Test
       * 5,000 Monte Carlo paths over a 12-month horizon
       * Probability of Working Capital Shortfall

All charts are interactive Plotly charts in dark mode.

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
N_SIMULATIONS = 10_000          # Market tab
TRADING_DAYS = 252
MIN_HISTORY_POINTS = 60

ENT_PATHS = 5_000               # Enterprise tab
ENT_MONTHS = 12

PLOT_TEMPLATE = "plotly_dark"
ACCENT = "#00d4ff"
RED = "#ff4b6e"
ORANGE = "#ffa62b"
GREEN = "#2ee6a6"
PINK = "#ff00aa"

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


def parse_seed(seed_text):
    """Return an int seed, None (random) or None plus a warning flag."""
    seed_text = (seed_text or "").strip()
    if not seed_text:
        return None, False
    try:
        return int(seed_text), False
    except ValueError:
        return None, True


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None)


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
    """Daily log-return mean and standard deviation."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return float(log_returns.mean()), float(log_returns.std(ddof=1)), log_returns


def simulate_gbm(s0, mu, sigma, horizon, n_sims, seed=None):
    """
    Vectorised Geometric Brownian Motion with daily time step (dt = 1 day):

        S(t+1) = S(t) * exp( (mu - 0.5*sigma^2) + sigma * Z ),   Z ~ N(0, 1)

    mu and sigma are the *daily* mean and std of log returns.
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
        (-metrics[99]["cvar"], PINK, "99% CVaR"),
    ]
    for x, color, label in lines:
        fig.add_vline(x=x, line=dict(color=color, width=2, dash="dash"),
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="top")

    fig.update_xaxes(title=f"Profit / Loss ({currency})")
    fig.update_yaxes(title="Number of simulations")
    fig.update_layout(bargap=0.02, hovermode="closest")
    return style(fig, "Distribution of Simulated Profit & Loss", 450)


def render_market_tab():
    st.caption(
        f"Geometric Brownian Motion · {N_SIMULATIONS:,} simulations · "
        "95% / 99% Value-at-Risk & Expected Shortfall (CVaR) · "
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
            "Horizon (trading days)", 1, 504, 21, key="mkt_horizon",
            help="21 ≈ 1 month, 63 ≈ 1 quarter, 252 ≈ 1 year")

        r2 = st.columns([2, 1, 1])
        drift_mode = r2[0].radio(
            "Drift assumption",
            ["Historical average return", "Zero drift (conservative)"],
            horizontal=True, key="mkt_drift",
            help="Zero drift ignores past average returns and models only volatility.")
        seed_text = r2[1].text_input(
            "Random seed (optional)", value="", key="mkt_seed",
            help="Enter a whole number for reproducible results.")
        r2[2].markdown("&nbsp;", unsafe_allow_html=True)
        r2[2].form_submit_button("🚀 Run simulation", width="stretch")

    if not ticker:
        st.warning("Please enter a ticker symbol.")
        return

    seed, bad_seed = parse_seed(seed_text)
    if bad_seed:
        st.error("Seed must be a whole number. Using a random seed instead.")

    try:
        with st.spinner(f"Fetching data for {ticker}..."):
            prices = load_prices(ticker, period)
            name = load_name(ticker)
    except ValueError as err:
        st.error(str(err))
        return

    mu, sigma, _ = estimate_parameters(prices)
    if sigma <= 0 or np.isnan(sigma):
        st.error("Volatility could not be estimated from this data (prices are flat).")
        return

    mu_used = mu if drift_mode.startswith("Historical") else 0.0
    s0 = float(prices.iloc[-1])

    with st.spinner(f"Running {N_SIMULATIONS:,} simulations..."):
        paths = simulate_gbm(s0, mu_used, sigma, horizon, N_SIMULATIONS, seed)

    final_prices = paths[-1]
    final_values = investment * final_prices / s0
    pnl = final_values - investment
    metrics = risk_metrics(pnl)

    # ---- Header info --------------------------------------------------------
    st.subheader(f"{name} ({ticker})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last price", f"{s0:,.2f}")
    c2.metric("Annualised return (hist.)", f"{mu * TRADING_DAYS:.2%}")
    c3.metric("Annualised volatility", f"{sigma * np.sqrt(TRADING_DAYS):.2%}")
    c4.metric("Data points used", f"{len(prices):,}")

    # ---- Risk metrics -------------------------------------------------------
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

    # ---- Charts -------------------------------------------------------------
    t1, t2, t3, t4 = st.tabs(
        ["📈 Simulated paths", "📊 P&L distribution", "🕰️ Price history", "📋 Details & export"]
    )

    with t1:
        show(paths_chart(paths, horizon))
    with t2:
        show(distribution_chart(pnl, metrics, "currency units"))
    with t3:
        show(price_history_chart(prices, ticker))
    with t4:
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
            key="mkt_download",
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


# ============================================================================
# TAB 2 - ENTERPRISE CASH-FLOW STRESS TEST
# ============================================================================
def simulate_cash_flows(revenue, overhead, vol, opening_cash, growth,
                        months=ENT_MONTHS, n_paths=ENT_PATHS, seed=None):
    """
    Monte Carlo simulation of a business's cash balance.

    Monthly revenue in month t is log-normally distributed around its expected
    value, so it can never go negative and its average equals the expected value:

        E[R_t] = revenue * (1 + growth)^(t-1)
        R_t    = E[R_t] * exp(vol * Z - 0.5 * vol^2),   Z ~ N(0, 1), independent each month

    Overhead is a fixed monthly cost. Cash balance after month t is:

        Cash_t = opening_cash + sum_{k<=t} (R_k - overhead)

    Returns cash with shape (months + 1, n_paths); row 0 is the opening cash.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((months, n_paths))
    expected = revenue * (1.0 + growth) ** np.arange(months)          # shape (months,)
    revenues = expected[:, None] * np.exp(vol * z - 0.5 * vol**2)
    net = revenues - overhead

    cash = np.empty((months + 1, n_paths))
    cash[0] = opening_cash
    cash[1:] = opening_cash + np.cumsum(net, axis=0)
    return cash


def shortfall_stats(cash):
    """Summarise shortfall risk. A shortfall means cash balance below zero."""
    balances = cash[1:]                                   # months 1..12
    negative = balances < 0                               # (months, paths)
    breached_by_month = np.cumsum(negative, axis=0) > 0   # has ever been negative by month t
    prob_by_month = breached_by_month.mean(axis=1)
    prob_negative_at_month = negative.mean(axis=1)

    ever = breached_by_month[-1]
    first_month = np.argmax(negative, axis=0) + 1         # valid only where ever is True
    avg_first = float(first_month[ever].mean()) if ever.any() else None

    peak_gap = np.maximum(0.0, -balances.min(axis=0))     # deepest point below zero
    return {
        "prob_by_month": prob_by_month,
        "prob_negative_at_month": prob_negative_at_month,
        "prob_any": float(prob_by_month[-1]),
        "avg_first_month": avg_first,
        "peak_gap": peak_gap,
        "buffer_95": float(np.percentile(peak_gap, 95)),
        "ending_cash": cash[-1],
    }


def shortfall_probability_chart(prob_by_month):
    months = np.arange(1, len(prob_by_month) + 1)
    pct = prob_by_month * 100
    fig = go.Figure(go.Scatter(
        x=months, y=pct, mode="lines+markers",
        line=dict(color=RED, width=3), marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(255, 75, 110, 0.18)",
        name="Cumulative probability",
        hovertemplate="%{y:.1f}%"))
    fig.add_hline(y=5, line=dict(color=ORANGE, width=1.5, dash="dash"),
                  annotation_text="5% tolerance", annotation_font_color=ORANGE,
                  annotation_position="bottom right")
    fig.update_xaxes(title="Month", dtick=1)
    fig.update_yaxes(title="Probability of shortfall by this month (%)",
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
                  annotation_text="Cash = 0 (shortfall)", annotation_font_color=RED,
                  annotation_position="bottom right")
    fig.update_xaxes(title="Month", dtick=1)
    fig.update_yaxes(title="Cash balance / working capital")
    return style(fig, f"Simulated Cash Balance (50 of {cash.shape[1]:,} paths shown)", 480)


def ending_cash_chart(ending_cash):
    fig = go.Figure(go.Histogram(
        x=ending_cash, nbinsx=80, marker=dict(color=ACCENT, line=dict(width=0)),
        opacity=0.85, name="Ending cash",
        hovertemplate="Cash: %{x:,.0f}<br>Paths: %{y}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=RED, width=2, dash="dash"),
                  annotation_text="Cash = 0", annotation_font_color=RED,
                  annotation_position="top")
    fig.update_xaxes(title="Cash balance at end of month 12")
    fig.update_yaxes(title="Number of paths")
    fig.update_layout(bargap=0.02, hovermode="closest")
    return style(fig, "Distribution of Month-12 Cash Balance", 400)


def render_enterprise_tab():
    st.caption(
        f"{ENT_PATHS:,} Monte Carlo paths · {ENT_MONTHS}-month horizon · "
        "a shortfall occurs when the cash balance (working capital) drops below zero."
    )

    with st.form("enterprise_form"):
        c = st.columns(4)
        revenue = c[0].number_input(
            "Monthly Revenue", min_value=0.0, value=100_000.0, step=5_000.0,
            format="%.2f", key="ent_revenue",
            help="Expected revenue per month.")
        overhead = c[1].number_input(
            "Monthly Overhead Costs", min_value=0.0, value=95_000.0, step=5_000.0,
            format="%.2f", key="ent_overhead",
            help="Fixed costs paid every month (rent, salaries, utilities, etc.).")
        vol_pct = c[2].number_input(
            "Revenue Volatility (%)", min_value=0.0, max_value=100.0, value=15.0,
            step=1.0, format="%.1f", key="ent_vol",
            help="Typical month-to-month swing in revenue, as a percentage of revenue.")
        opening = c[3].number_input(
            "Opening Working Capital", min_value=0.0, value=25_000.0, step=5_000.0,
            format="%.2f", key="ent_opening",
            help="Cash on hand at the start. Needed to judge whether cash runs out.")

        with st.expander("Advanced options"):
            a1, a2 = st.columns(2)
            growth_pct = a1.number_input(
                "Expected monthly revenue growth (%)", min_value=-20.0, max_value=20.0,
                value=0.0, step=0.5, format="%.1f", key="ent_growth",
                help="0 = flat revenue. Negative values model a shrinking business.")
            seed_text = a2.text_input(
                "Random seed (optional)", value="", key="ent_seed",
                help="Enter a whole number for reproducible results.")

        st.form_submit_button("🚀 Run stress test", width="stretch")

    seed, bad_seed = parse_seed(seed_text)
    if bad_seed:
        st.error("Seed must be a whole number. Using a random seed instead.")

    vol = vol_pct / 100.0
    growth = growth_pct / 100.0

    cash = simulate_cash_flows(revenue, overhead, vol, opening, growth, seed=seed)
    stats = shortfall_stats(cash)

    expected_net = revenue - overhead
    if expected_net < 0 and growth_pct <= 0:
        st.warning(
            f"Expected monthly cash flow is negative ({expected_net:,.2f}). "
            "The business is burning cash even before volatility is considered."
        )

    # ---- Headline result ----------------------------------------------------
    prob = stats["prob_any"]
    headline = f"Probability of a Working Capital Shortfall within 12 months: **{prob:.1%}**"
    if prob < 0.05:
        st.success(f"🟢 Low risk. {headline}")
    elif prob < 0.20:
        st.warning(f"🟠 Moderate risk. {headline}")
    else:
        st.error(f"🔴 High risk. {headline}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shortfall probability (12 mo)", f"{prob:.1%}")
    m2.metric("Cash negative at month 12", f"{stats['prob_negative_at_month'][-1]:.1%}")
    m3.metric("Median month-12 cash", f"{np.median(stats['ending_cash']):,.0f}")
    m4.metric("Extra cash for 95% safety", f"{stats['buffer_95']:,.0f}",
              help="Additional opening cash that would keep the balance above zero "
                   "in 95% of simulated paths.")

    first = stats["avg_first_month"]
    detail = (
        f"In paths that run short, the first shortfall happens on average in month {first:.1f}. "
        if first is not None else
        "No simulated path ran out of cash. "
    )
    st.caption(
        f"{detail}Expected monthly cash flow is **{expected_net:,.2f}** "
        f"(revenue {revenue:,.2f} − overhead {overhead:,.2f}) "
        f"with revenue volatility of {vol_pct:.1f}%."
    )

    # ---- Charts -------------------------------------------------------------
    t1, t2, t3, t4 = st.tabs(
        ["⚠️ Shortfall probability", "💵 Cash balance paths",
         "📊 Month-12 distribution", "📋 Details & export"]
    )

    with t1:
        show(shortfall_probability_chart(stats["prob_by_month"]))
    with t2:
        show(cash_fan_chart(cash))
    with t3:
        show(ending_cash_chart(stats["ending_cash"]))
    with t4:
        table = pd.DataFrame({
            "Month": np.arange(1, ENT_MONTHS + 1),
            "Cumulative shortfall probability": [f"{x:.1%}" for x in stats["prob_by_month"]],
            "Cash negative in this month": [f"{x:.1%}" for x in stats["prob_negative_at_month"]],
            "Median cash balance": [f"{x:,.0f}" for x in np.median(cash[1:], axis=1)],
            "5th percentile cash": [f"{x:,.0f}" for x in np.percentile(cash[1:], 5, axis=1)],
        })
        st.table(table.set_index("Month"))

        export = pd.DataFrame(
            cash[1:].T, columns=[f"month_{m}" for m in range(1, ENT_MONTHS + 1)])
        export.insert(0, "peak_funding_gap", stats["peak_gap"])
        st.download_button(
            "⬇️ Download all simulated cash paths (CSV)",
            data=export.to_csv(index_label="path").encode("utf-8"),
            file_name="enterprise_cash_flow_simulation.csv",
            mime="text/csv",
            key="ent_download",
        )

    with st.expander("ℹ️ Methodology & limitations"):
        st.markdown(
            f"""
**Model.** Each month's revenue is drawn from a log-normal distribution whose average is
your expected revenue (grown by the monthly growth rate, if any) and whose spread is set by
the volatility you entered. Revenue can therefore never be negative. Overhead is a fixed
cost. The cash balance is the opening working capital plus the running total of
(revenue − overhead).

**Shortfall.** A path has a *shortfall* in a given month if its cash balance is below zero
at the end of that month. The headline probability is the share of the {ENT_PATHS:,} paths
that dip below zero at least once in the {ENT_MONTHS} months.

**Extra cash for 95% safety.** The extra opening cash that would have kept the balance
non-negative in 95% of paths.

**Limitations.** Monthly revenue shocks are assumed independent, so seasonality, customer
concentration, late payments and one-off shocks are not modelled. Overhead is assumed fixed
and there is no debt, tax or financing. Treat the output as a planning aid, not a forecast.
This tool is for education and analysis only and is **not financial advice**.
"""
        )


# ============================================================================
# MAIN
# ============================================================================
st.title("📉 Monte Carlo Risk Engine")

tab_market, tab_enterprise = st.tabs(
    ["📈 Market Asset Risk", "🏢 Enterprise Cash-Flow Stress Test"]
)

with tab_market:
    render_market_tab()

with tab_enterprise:
    render_enterprise_tab()
