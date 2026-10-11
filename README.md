# 📊 Quantitative Risk & Enterprise Analytics Suite

An institutional-grade web application deployed on Streamlit designed to evaluate financial market tail-risk and corporate cash-flow solvency under macroeconomic stress scenarios. 

🔗 **Live Application:** [monte-carlo-risk-engine-bwbp7vyngxhxj7s7jtymzk.streamlit.app](https://monte-carlo-risk-engine-bwbp7vyngxhxj7s7jtymzk.streamlit.app/)

---

## 🌟 Key Features

### 📈 Module 1: Market Risk Engine & Option Valuation
* **Stochastic Modeling:** Choose between standard **Geometric Brownian Motion (GBM)** and the **Heston Stochastic Volatility Model** (incorporating mean reversion, volatility of volatility, and correlated Brownian paths).
* **Tail Risk Metrics:** Real-time computation of **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR / Expected Shortfall)** at 95% and 99% confidence levels.
* **Monte Carlo Option Pricing:** Dynamic pricing engine for European Call and Put derivatives based on simulated terminal price distributions.

### 🏢 Module 2: Enterprise Stress-Testing & Macro Shocks
* **Macro Environment Filters:** Pre-configured macroeconomic crisis shocks including *2008 Financial Crisis*, *Supply Chain Disruption*, and *Interest Rate Spikes (+300 bps)*.
* **Solvency Analysis:** Multi-thousand trial cash-flow simulations computing expected ending cash reserves and exact **Insolvency Probability (%)**.

### 📄 Module 3: Executive Reporting Pipeline
* **On-Demand PDF Generation:** Built-in PDF compiler generating downloadable executive summary risk reports detailing simulation parameters, downside percentiles, and risk assessments.

---

## 🛠️ Tech Stack & Libraries
* **Frontend & UX:** Streamlit, Custom HTML/CSS
* **Quantitative Math & Simulations:** NumPy, SciPy
* **Data Sources & Visualization:** yFinance, Plotly
* **Document Generation:** FPDF (`fpdf2`)
