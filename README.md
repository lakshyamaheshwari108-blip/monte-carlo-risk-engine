# 📈 Quantitative Monte Carlo Risk Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://monte-carlo-risk-engine-bwbp7vyngxhxj7s7jtymzk.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive web-based quantitative finance and risk analytics dashboard designed to model asset price uncertainty, simulate stochastic price trajectories, and compute institutional-grade portfolio risk metrics.

---

## 🚀 Live Demo
Access the fully deployed application here: **[Monte Carlo Risk Engine Live App](https://monte-carlo-risk-engine-bwbp7vyngxhxj7s7jtymzk.streamlit.app/)**

---

## 📊 Key Features
* **Real-Time Data Integration:** Dynamically fetches live historical equity and asset pricing data via the Yahoo Finance API (`yfinance`).
* **Vectorized Stochastic Simulation:** Runs $10,000+$ independent price paths using optimized NumPy array operations for lightning-fast execution.
* **Advanced Risk Metrics:** Computes critical tail-risk indicators utilized in institutional portfolio management:
  * **Value at Risk (95% & 99% VaR):** Maximum expected loss at specified confidence levels over a given time horizon.
  * **Conditional Value at Risk (CVaR / Expected Tail Loss):** Average magnitude of losses exceeding the VaR threshold during tail-risk events.
* **Interactive UI & Dark-Mode Visualizations:** Built with Streamlit and Plotly to allow instantaneous sensitivity analysis across sliding time horizons, drift rates, and portfolio sizes.

---

## 📐 Mathematical Framework

The simulation engine models asset price movements using **Geometric Brownian Motion (GBM)**, the standard continuous-time stochastic process governed by Itô's lemma:

$$S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma \sqrt{t} Z\right)$$

Where:
* $S_0$ = Current asset spot price
* $\mu$ = Daily expected return (drift) calculated from historical log returns
* $\sigma$ = Historical daily volatility (standard deviation of returns)
* $t$ = Time horizon in trading days
* $Z$ = Standard normal random variable drawn from $\mathcal{N}(0, 1)$

---

## 🛠️ Technical Stack
* **Language:** Python 3.9+
* **Core Libraries:** 
  * `NumPy` & `Pandas` (Vectorized mathematical operations and data manipulation)
  * `SciPy` (Statistical distributions and probability calculations)
  * `Plotly` (Interactive, publication-grade dark-mode financial charting)
  * `yfinance` (Automated financial market data retrieval)
* **Deployment Platform:** Streamlit Community Cloud

---

## 💻 Local Installation & Usage

If you wish to run or modify this application locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/monte-carlo-risk-engine.git](https://github.com/YOUR-USERNAME/monte-carlo-risk-engine.git)
   cd monte-carlo-risk-engine
