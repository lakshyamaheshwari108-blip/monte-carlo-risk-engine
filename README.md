# Monte Carlo Risk Engine

[![tests](https://github.com/lakshyamaheshwari108-blip/monte-carlo-risk-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/lakshyamaheshwari108-blip/monte-carlo-risk-engine/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

An interactive quantitative-finance and enterprise-risk web application built with Python and
Streamlit. It simulates market and business risk with Monte Carlo methods, and **proves its own
numerical accuracy** against analytic benchmarks and an automated test suite.

**Live app:** https://YOUR-APP-NAME.streamlit.app

## What it does

### 1. Market Asset Risk
* Live prices from Yahoo Finance (`yfinance`).
* Two price models: **Geometric Brownian Motion** and the **Heston stochastic volatility model**
  (correlated Brownian motions, mean-reverting variance, full-truncation Euler scheme).
* 10,000-path simulations producing **95% / 99% VaR and CVaR**. Profitable lower tails are
  reported honestly as "No loss" rather than as a negative loss.
* **European call and put pricing** under the risk-neutral measure, compared with
  Black-Scholes (GBM) or a semi-analytic Heston price (Heston).

### 2. Enterprise Cash-Flow Stress Test
* Macro scenarios: Baseline, 2008 Financial Crisis, Supply Chain Shock, Rate Spike (+300 bps),
  each re-setting volatility, overheads, variable-cost ratio, revenue shock, drift and borrowing cost.
* 5,000 to 50,000 Monte Carlo trials over 12 months; insolvency probability with a Wilson
  confidence interval, ending-cash distribution and a scenario comparison.
* One-click **Executive PDF Risk Report**.

### 3. Model Validation
Live convergence studies: Monte Carlo vs Black-Scholes, and Heston Monte Carlo vs a
semi-analytic reference as the time step is refined.

## Validation

| Check | Result |
|---|---|
| GBM Monte Carlo vs Black-Scholes | Error inside the 95% band at every size; shrinks as 1/sqrt(N) |
| z-statistic calibration | 0.25% of replications beyond 3 standard errors (theory 0.27%) |
| Put-call parity, discounted-price martingale | Hold within Monte Carlo error |
| Heston reference | Collapses to Black-Scholes as vol-of-vol tends to 0, at the theoretical rate |
| Heston Monte Carlo, Feller condition satisfied or moderately violated | Matches reference within sampling error |
| Heston Monte Carlo, severely violated Feller condition | Small, documented Euler bias that shrinks with the time step |

Full tables and discussion: [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Project structure

```
app.py               Streamlit user interface and charts
risk_engine.py       Numerical core: simulation, pricing, VaR/CVaR, stress testing (no UI code)
report_pdf.py        Executive PDF report builder (fpdf2)
tests/               Automated pytest suite
scripts/             run_validation.py reproduces the validation tables
docs/                VALIDATION.md
.github/workflows/   Continuous integration: runs the tests on every commit
```

Keeping the mathematics in `risk_engine.py`, free of any UI code, is what makes it directly testable.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers option-pricing convergence, put-call parity and the martingale property;
Heston against a semi-analytic reference (including Feller-violating parameters); tail-risk edge
cases; enterprise insolvency maths; invalid inputs; PDF generation; and an end-to-end headless
run of the whole app.

## Method summary

* **GBM:** `S(t+1) = S(t) exp[(mu - sigma^2/2) + sigma Z]`
* **Heston:** `dS = mu S dt + sqrt(v) S dW1`, `dv = kappa (theta - v) dt + xi sqrt(v) dW2`, `corr(dW1, dW2) = rho`
* **VaR / CVaR:** loss quantile and mean loss beyond it, from simulated profit and loss
* **Option price:** `exp(-rT) * mean(payoff)` over risk-neutral terminal prices
* **Insolvency:** month-end cash below zero in at least one of 12 months

## Limitations

Heston parameters are not calibrated to market option prices; the Euler scheme is first-order
accurate; dividends and jumps are not modelled; scenario presets are stylised illustrations, not
forecasts. For education and analysis only, not financial advice.

## License

MIT
