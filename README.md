# 📈 Quantitative Risk & Enterprise Analytics Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://monte-carlo-risk-engine-bwbp7vyngxhxj7s7jtymzk.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, dual-module web application designed to model financial asset price uncertainty using stochastic processes and simulate operational cash-flow stress testing for mid-market business enterprises.

---

## 🚀 Live Demo
Access the fully deployed application here: **[Quantitative Risk & Enterprise Analytics Engine](https://monte-carlo-risk-engine-bwbp7vyngxhxj7s7jtymzk.streamlit.app/)**

---

## 📊 Core Modules & Features

### 1. 📈 Market Asset Risk Engine
* **Real-Time Market Data Integration:** Dynamically fetches live historical equity pricing via the Yahoo Finance API (`yfinance`).
* **Vectorized Stochastic Simulation:** Runs $10,000+$ independent price trajectories using optimized NumPy array operations for instant execution.
* **Institutional Tail-Risk Metrics:** Computes key risk indicators used in portfolio risk management:
  * **Value at Risk (95% & 99% VaR):** Maximum expected loss at specified confidence intervals over a given horizon.
  * **Conditional Value at Risk (CVaR / Expected Tail Loss):** Average loss magnitude exceeding the VaR threshold during tail-risk events.

### 2. 🏢 Enterprise Cash-Flow Stress Tester
* **Operational Cash-Flow Modeling:** Simulates monthly cash reserve trajectories under variable revenue volatility, fixed operating overheads, and variable cost dynamics.
* **Stochastic Liquidity Analysis:** Runs $5,000$ randomized revenue paths over 3-to-24 month forecast horizons to quantify liquidity buffers.
* **Predictive Risk Indicators:** Calculates the exact **Probability of Working Capital Shortfall / Insolvency** to support executive decision-making and risk mitigation.

---

## 📐 Mathematical Framework

### Market Asset Engine: Geometric Brownian Motion (GBM)
Asset price dynamics are modeled using Geometric Brownian Motion governed by Itô's lemma:

$$S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma \sqrt{t} Z\right)$$

Where:
* $S_0$ = Asset spot price
* $\mu$ = Daily expected return (drift) derived from historical log returns
* $\sigma$ = Historical daily volatility
* $t$ = Forecast time horizon in trading days
* $Z$ = Standard normal random variable drawn from $\mathcal{N}(0, 1)$

### Enterprise Engine: Stochastic Liquidity Process
Working capital reserves $C_t$ at month $t$ are modeled as:

$$C_t = C_{t-1} + \left( R_t \cdot (1 - v) - F \right)$$

$$R_t \sim \max\left(0, \mathcal{N}(\mu_R, \sigma_R)\right)$$

Where:
* $C_0$ = Initial working capital cash reserves
* $R_t$ = Stochastic monthly revenue subject to normal distribution volatility $\sigma_R$
* $v$ = Variable cost margin percentage (% of revenue)
* $F$ = Fixed monthly operating overheads

---

## 🛠️ Technical Stack
* **Language:** Python 3.9+
* **Core Libraries:** 
  * `NumPy` & `Pandas` (Vectorized matrix operations and data processing)
  * `SciPy` (Statistical distribution operations)
  * `Plotly` (Interactive dark-mode financial charting)
  * `yfinance` (Real-time market data retrieval)
* **Deployment Platform:** Streamlit Community Cloud

---

## 💻 Local Installation & Usage

To run or modify this application locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/](https://github.com/YOUR-USERNAME/)
