"""
Monte Carlo Risk Engine  -  Quantitative Finance & Enterprise Risk Suite
========================================================================

This file is the Streamlit user interface. All of the mathematics lives in
``risk_engine.py`` (simulation, pricing, risk metrics, stress testing) and the PDF
builder lives in ``report_pdf.py``; both are covered by the automated tests in
``tests/``.

Tab 1  Market Asset Risk          GBM or Heston, VaR / CVaR, option pricing
Tab 2  Enterprise Stress Test     macro scenarios, insolvency probability, PDF report
Tab 3  Model Validation           live convergence studies against analytic prices

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats as sp_stats

from risk_engine import (
    DEFAULT_SCENARIO,
    ENT_MONTHS,
    MAX_HORIZON_DAYS,
    MODEL_GBM,
    MODEL_HESTON,
    N_SIMULATIONS,
    SCENARIOS,
    TRADING_DAYS,
    build_market_results,
    cash_stats,
    compare_scenarios,
    estimate_parameters,
    explain_tail_risk,
    feller_condition_holds,
    format_tail_value,
    gbm_convergence_study,
    heston_convergence_study,
    simulate_cash_flows,
    tail_line_label,
    tail_metric_display,
    validate_price_history,
)

try:  # PDF export is optional: the rest of the app works without it
    from report_pdf import build_pdf_report

    PDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    build_pdf_report = None
    PDF_AVAILABLE = False

ENT_TRIAL_OPTIONS = [5_000, 10_000, 25_000, 50_000]

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

    return validate_price_history(hist, ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_name(ticker: str) -> str:
    """Best-effort company name; never fails."""
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


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

    for x, color, label in (
            (-metrics[95]["var"], ORANGE, tail_line_label(95, metrics[95], "VaR")),
            (-metrics[99]["var"], RED, tail_line_label(99, metrics[99], "VaR")),
            (-metrics[99]["cvar"], PINK, tail_line_label(99, metrics[99], "CVaR"))):
        fig.add_vline(x=x, line=dict(color=color, width=2, dash="dash"),
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="top")

    fig.update_xaxes(title=f"Profit / Loss ({currency})")
    fig.update_yaxes(title="Number of simulations")
    fig.update_layout(bargap=0.02, hovermode="closest", barmode="overlay")
    return style(fig, "Distribution of Simulated Profit & Loss", 450)


def option_curve_chart(curve, s0, strike, model_label):
    k = curve["strikes"]
    ref_name = curve["ref_name"]
    ref_calls = curve["ref_calls"] if curve["ref_calls"] is not None else curve["bs_calls"]
    ref_puts = curve["ref_puts"] if curve["ref_puts"] is not None else curve["bs_puts"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=curve["mc_calls"], mode="lines", name="Call (Monte Carlo)",
                             line=dict(color=GREEN, width=3), hovertemplate="%{y:,.2f}"))
    fig.add_trace(go.Scatter(x=k, y=ref_calls, mode="lines", name=f"Call ({ref_name})",
                             line=dict(color=GREEN, width=1.5, dash="dot"), hovertemplate="%{y:,.2f}"))
    fig.add_trace(go.Scatter(x=k, y=curve["mc_puts"], mode="lines", name="Put (Monte Carlo)",
                             line=dict(color=RED, width=3), hovertemplate="%{y:,.2f}"))
    fig.add_trace(go.Scatter(x=k, y=ref_puts, mode="lines", name=f"Put ({ref_name})",
                             line=dict(color=RED, width=1.5, dash="dot"), hovertemplate="%{y:,.2f}"))
    fig.add_vline(x=s0, line=dict(color="#ffffff", width=1, dash="dot"),
                  annotation_text="Spot", annotation_position="top left")
    fig.add_vline(x=strike, line=dict(color=ACCENT, width=2, dash="dash"),
                  annotation_text="Your strike", annotation_font_color=ACCENT,
                  annotation_position="top right")
    fig.update_xaxes(title="Strike price")
    fig.update_yaxes(title="Option price")
    return style(fig, f"Option Price vs Strike – {model_label}", 450)


def price_text(value):
    """Format an option price, tolerating a missing benchmark."""
    return "n/a" if value is None or not np.isfinite(value) else f"{value:,.4f}"


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
    for col, kind, level in ((m1, "VaR", 95), (m2, "VaR", 99), (m3, "CVaR", 95), (m4, "CVaR", 99)):
        d = tail_metric_display(
            kind, level, metrics[level]["var" if kind == "VaR" else "cvar"], investment)
        col.metric(d["label"], d["value_text"], d["delta_text"], delta_color="off")

    st.caption(explain_tail_risk(metrics, horizon, investment))

    # ---- Option pricing headline -------------------------------------------
    st.markdown(f"### European option prices (strike {options['strike']:,.2f}, "
                f"maturity {horizon} trading days)")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Call price (Monte Carlo)", f"{options['call']:,.4f}",
              f"± {1.96 * options['call_se']:.4f} (95% CI)", delta_color="off")
    o2.metric("Put price (Monte Carlo)", f"{options['put']:,.4f}",
              f"± {1.96 * options['put_se']:.4f} (95% CI)", delta_color="off")
    ref_help = ("Closed-form Black–Scholes price using historical volatility."
                if not is_heston else
                "Semi-analytic Heston price (Fourier inversion): the trusted reference "
                "for the Monte Carlo simulation.")
    o3.metric(f"Call – {options['ref_name']}", price_text(options["ref_call"]), help=ref_help)
    o4.metric(f"Put – {options['ref_name']}", price_text(options["ref_put"]), help=ref_help)

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
            order = ((95, "var"), (99, "var"), (95, "cvar"), (99, "cvar"))
            cmp_df = pd.DataFrame({
                "Metric": ["95% VaR", "99% VaR", "95% CVaR", "99% CVaR"],
                "GBM": [format_tail_value(g[lv][k]) for lv, k in order],
                "Heston": [format_tail_value(h[lv][k]) for lv, k in order],
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
            f"discounted. Dotted lines are the {options['ref_name']} benchmark. "
            "Under Heston, prices differ from Black–Scholes because of stochastic "
            "volatility and skew."
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
                "Risk-neutral martingale check (1 ± 95% CI)",
                "Put–call parity residual (should be ≈ 0)",
                f"Benchmark price: {options['ref_name']} (call / put)",
                "Flat-volatility Black–Scholes (call / put)",
            ],
            "Value": [
                model, f"{s0:,.2f}", f"{res['final_prices'].mean():,.2f}",
                f"{np.median(res['final_prices']):,.2f}",
                f"{res['final_prices'].max():,.2f}", f"{res['final_prices'].min():,.2f}",
                f"{(res['pnl'] < 0).mean():.2%}", f"{res['pnl'].mean():,.2f}",
                format_tail_value(metrics[95]["var"]), format_tail_value(metrics[99]["var"]),
                format_tail_value(metrics[95]["cvar"]), format_tail_value(metrics[99]["cvar"]),
                f"{sp_stats.skew(lr):.3f}", f"{sp_stats.kurtosis(lr):.3f}",
                f"{options['martingale_ratio']:.4f} ± {1.96 * options['martingale_se']:.4f}",
                f"{options['parity_gap']:.4f}",
                f"{price_text(options['ref_call'])} / {price_text(options['ref_put'])}",
                f"{price_text(options['bs_call'])} / {price_text(options['bs_put'])}",
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
CVaR is the average loss in the scenarios worse than VaR. Both are shown as loss amounts.
If even the lower tail of the simulated distribution is profitable (for example a short
horizon with strong positive drift), there is no loss at that confidence level and the app
says **No loss** instead of printing a negative loss.

**Option pricing.** Prices of European options equal the discounted average payoff over
risk-neutral terminal prices (drift = risk-free rate). No dividends are modelled. The
± figures are 95% Monte Carlo confidence intervals. The benchmark is Black–Scholes under
GBM and a semi-analytic Heston price (Fourier inversion of the characteristic function,
"little Heston trap" formulation) under Heston.

**Limitations.** Heston parameters are not calibrated to option prices (only v₀ and θ
can be tied to historical variance), jumps are not modelled, and past data does not
guarantee future behaviour. For education and analysis only; **not financial advice**.
"""
        )


# ============================================================================
# TAB 2 - ENTERPRISE CASH-FLOW STRESS TEST
# ============================================================================
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
    if PDF_AVAILABLE:
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
# TAB 3 - MODEL VALIDATION
# ============================================================================
HESTON_VALIDATION_PARAMS = dict(v0=0.04, kappa=1.0, theta=0.04, xi=0.6, rho=-0.7)


def gbm_convergence_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["paths"], y=df["call_error"].abs(), mode="lines+markers",
        name="|Monte Carlo − Black–Scholes| (call)",
        line=dict(color=ACCENT, width=3), hovertemplate="%{y:.5f}"))
    fig.add_trace(go.Scatter(
        x=df["paths"], y=1.96 * df["call_se"], mode="lines",
        name="95% error band (1.96 × standard error)",
        line=dict(color=ORANGE, width=2, dash="dash"), hovertemplate="%{y:.5f}"))
    fig.update_xaxes(type="log", title="Number of simulated paths")
    fig.update_yaxes(type="log", title="Absolute price error")
    return style(fig, "GBM: Monte Carlo error vs Black–Scholes", 420)


def heston_convergence_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["steps_per_day"], y=df["error"], mode="lines+markers",
        error_y=dict(type="data", array=(1.96 * df["call_se"]).tolist(), visible=True),
        name="Monte Carlo − semi-analytic (call)",
        line=dict(color=VIOLET, width=3), hovertemplate="%{y:+.4f}"))
    fig.add_hline(y=0, line=dict(color="#ffffff", width=1, dash="dot"))
    fig.update_xaxes(type="log", title="Euler time steps per trading day",
                     tickvals=df["steps_per_day"].tolist(),
                     ticktext=[str(s) for s in df["steps_per_day"]])
    fig.update_yaxes(title="Price error (Monte Carlo − reference)")
    return style(fig, "Heston: discretisation error as the time step is refined", 420)


def run_validation_studies():
    gbm = gbm_convergence_study()
    heston = heston_convergence_study(
        years=0.25, steps_per_day=(1, 2, 4, 8), n_paths=60_000, **HESTON_VALIDATION_PARAMS)
    return {"gbm": gbm, "heston": heston}


def render_validation_tab():
    st.caption("Do the pricing engines reproduce known answers? Run the studies below to find out.")
    st.markdown(
        """
1. **GBM vs Black–Scholes.** Black–Scholes is exact under GBM, so Monte Carlo prices must
   converge to it, with errors consistent with the standard error (shrinking like 1/√N).
2. **Heston vs a semi-analytic reference.** The Monte Carlo (full-truncation Euler) is compared
   with the price obtained by Fourier inversion of Heston's characteristic function, using a
   parameter set that **violates the Feller condition**. Refining the time step should reduce
   the discretisation error.
"""
    )
    if st.button("▶ Run validation studies", key="val_run"):
        with st.spinner("Running convergence studies (a few seconds)..."):
            st.session_state["val_results"] = run_validation_studies()

    res = st.session_state.get("val_results")
    if res is None:
        st.info("Press **Run validation studies**. Results appear here.")
    else:
        g, h = res["gbm"], res["heston"]
        st.subheader("1. GBM Monte Carlo vs Black–Scholes")
        if (g["call_z"].abs() < 4).all() and (g["put_z"].abs() < 4).all():
            st.success("✅ Every call and put price is within 4 standard errors of Black–Scholes.")
        else:
            st.error("❌ At least one price is more than 4 standard errors from Black–Scholes.")
        show(gbm_convergence_chart(g))
        g_table = pd.DataFrame({
            "Paths": [f"{n:,}" for n in g["paths"]],
            "MC call": [f"{x:.4f}" for x in g["mc_call"]],
            "± 1.96 SE": [f"{1.96 * x:.4f}" for x in g["call_se"]],
            "Black–Scholes": [f"{x:.4f}" for x in g["bs_call"]],
            "Error": [f"{x:+.4f}" for x in g["call_error"]],
            "z-score": [f"{x:+.2f}" for x in g["call_z"]],
        })
        st.table(g_table.set_index("Paths"))

        st.subheader("2. Heston Monte Carlo vs semi-analytic price")
        p = HESTON_VALIDATION_PARAMS
        st.caption(
            f"κ = {p['kappa']}, θ = {p['theta']}, ξ = {p['xi']}, ρ = {p['rho']}, v₀ = {p['v0']} · "
            f"Feller condition (2κθ > ξ²): "
            f"{'satisfied' if feller_condition_holds(p['kappa'], p['theta'], p['xi']) else 'violated'} · "
            f"reference call price {h['reference'].iloc[0]:.4f}")
        show(heston_convergence_chart(h))
        h_table = pd.DataFrame({
            "Steps per day": [str(s) for s in h["steps_per_day"]],
            "MC call": [f"{x:.4f}" for x in h["mc_call"]],
            "± 1.96 SE": [f"{1.96 * x:.4f}" for x in h["call_se"]],
            "Reference": [f"{x:.4f}" for x in h["reference"]],
            "Error": [f"{x:+.4f}" for x in h["error"]],
            "z-score": [f"{x:+.2f}" for x in h["z"]],
        })
        st.table(h_table.set_index("Steps per day"))
        st.caption("A small positive error with coarse steps is expected: Euler schemes carry a "
                   "discretisation bias that shrinks as the step shrinks. Remaining noise is "
                   "Monte Carlo sampling error.")

    with st.expander("🧪 Automated test suite"):
        st.markdown(
            """
The project ships with an automated `pytest` suite (see the `tests/` folder) that also runs
on every commit through GitHub Actions. It covers:

- **Option pricing:** convergence to Black–Scholes, put–call parity, the discounted-price martingale
- **Heston:** agreement with the semi-analytic reference, Feller-violating parameters, time-step convergence
- **Tail risk:** VaR / CVaR against closed forms, and honest labelling of profitable lower tails
- **Enterprise model:** closed-form cash paths, monotonicity, Wilson intervals, scenario ordering
- **Invalid inputs, PDF generation and an end-to-end app smoke test**

Run it locally with `pip install -r requirements-dev.txt` and then `pytest`.
"""
        )


# ============================================================================
# MAIN
# ============================================================================
st.title("📉 Monte Carlo Risk Engine")
st.caption("Quantitative finance and enterprise risk suite")

tab_market, tab_enterprise, tab_validation = st.tabs(
    ["📈 Market Asset Risk", "🏢 Enterprise Cash-Flow Stress Test", "🧪 Model Validation"]
)

with tab_market:
    render_market_tab()

with tab_enterprise:
    render_enterprise_tab()

with tab_validation:
    render_validation_tab()
