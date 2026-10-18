"""
End-to-end smoke test: run the real Streamlit app headlessly (no browser, no internet).

Yahoo Finance is replaced with a reproducible synthetic price history, so the whole
interface - all three tabs, the charts, the Heston benchmark and the PDF report - is
exercised exactly as a user would see it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

testing = pytest.importorskip("streamlit.testing.v1")
yf = pytest.importorskip("yfinance")

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def fake_history(self, period="5y", auto_adjust=True, **kwargs):
    rng = np.random.default_rng(0)
    n = 1250
    returns = rng.standard_t(4, n) * 0.012 + 0.0004
    close = 150.0 * np.exp(np.cumsum(returns))
    index = pd.bdate_range("2021-10-01", periods=n, tz="America/New_York")
    return pd.DataFrame({"Close": close}, index=index)


def fake_info(self):
    return {"shortName": "Test Corp"}


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setattr(yf.Ticker, "history", fake_history)
    monkeypatch.setattr(yf.Ticker, "get_info", fake_info)
    at = testing.AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_runs_without_exceptions(app):
    messages = [e.value for e in app.exception]
    assert len(app.exception) == 0, f"App raised: {messages}"


def test_app_shows_metrics_and_no_pdf_or_data_errors(app):
    assert len(app.metric) >= 12
    error_text = " ".join(str(e.value) for e in app.error)
    assert "PDF report could not be generated" not in error_text
    assert "No price data" not in error_text


def test_default_enterprise_scenario_is_baseline(app):
    assert app.selectbox(key="ent_scenario").value == "Baseline (Normal)"
    assert app.number_input(key="ent_vol").value == pytest.approx(12.0)


def test_choosing_a_macro_scenario_updates_the_inputs(app):
    app.selectbox(key="ent_scenario").select("2008 Financial Crisis").run()
    assert len(app.exception) == 0
    assert app.number_input(key="ent_vol").value == pytest.approx(30.0)
    assert app.number_input(key="ent_var_ratio").value == pytest.approx(50.0)
    assert app.number_input(key="ent_shock").value == pytest.approx(-25.0)
    # fixed overheads are 42% of the (unchanged) 100,000 base revenue
    assert app.number_input(key="ent_fixed").value == pytest.approx(42_000.0)
