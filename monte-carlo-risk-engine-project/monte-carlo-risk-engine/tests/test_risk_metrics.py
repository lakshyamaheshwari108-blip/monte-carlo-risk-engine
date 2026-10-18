"""
Priority 2 - tail-risk edge cases.

VaR and CVaR are reported as *losses*. When the lower tail of the P&L distribution is
profitable, the raw numbers are negative. That is mathematically valid but must never
be shown to a user as a loss amount. These tests pin down both the maths and the wording.
"""
import numpy as np
import pytest
from scipy.stats import norm

from risk_engine import (
    explain_tail_risk,
    format_tail_value,
    risk_metrics,
    simulate_gbm,
    tail_line_label,
    tail_metric_display,
)


# ---- Core maths against known distributions ---------------------------------
def test_uniform_distribution_has_known_var_and_cvar():
    pnl = np.linspace(-100.0, 100.0, 200_001)
    m = risk_metrics(pnl)
    assert m[95]["var"] == pytest.approx(90.0, abs=0.01)
    assert m[95]["cvar"] == pytest.approx(95.0, abs=0.01)
    assert m[99]["var"] == pytest.approx(98.0, abs=0.01)
    assert m[99]["cvar"] == pytest.approx(99.0, abs=0.01)


@pytest.mark.parametrize("mu,sigma", [(0.0, 1_000.0), (200.0, 800.0), (-150.0, 500.0)])
def test_normal_distribution_matches_closed_form(mu, sigma):
    rng = np.random.default_rng(10)
    pnl = rng.normal(mu, sigma, 2_000_000)
    m = risk_metrics(pnl)
    for level in (95, 99):
        z = norm.ppf(level / 100)
        var_exact = -mu + sigma * z
        cvar_exact = -mu + sigma * norm.pdf(z) / (1 - level / 100)
        assert m[level]["var"] == pytest.approx(var_exact, abs=0.02 * sigma)
        assert m[level]["cvar"] == pytest.approx(cvar_exact, abs=0.03 * sigma)


def test_cvar_never_below_var_and_99_never_below_95():
    rng = np.random.default_rng(3)
    for scale in (0.5, 1.0, 5.0):
        pnl = rng.standard_t(3, 50_000) * scale * 100 + 20
        m = risk_metrics(pnl)
        assert m[95]["cvar"] >= m[95]["var"]
        assert m[99]["cvar"] >= m[99]["var"]
        assert m[99]["var"] >= m[95]["var"]
        assert m[99]["cvar"] >= m[95]["cvar"]


def test_cvar_is_mean_of_the_tail_by_hand():
    pnl = np.array([-50.0, -30.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    m = risk_metrics(pnl)
    quantile = np.percentile(pnl, 5)
    expected_cvar = -pnl[pnl <= quantile].mean()
    assert m[95]["quantile"] == pytest.approx(quantile)
    assert m[95]["var"] == pytest.approx(-quantile)
    assert m[95]["cvar"] == pytest.approx(expected_cvar)


def test_loss_distribution_gives_positive_loss_flags():
    pnl = np.linspace(-500.0, 300.0, 10_001)
    m = risk_metrics(pnl)
    assert m[95]["var"] > 0 and m[95]["var_is_loss"] is True
    assert m[95]["cvar"] > 0 and m[95]["cvar_is_loss"] is True


# ---- Profitable lower tail --------------------------------------------------
@pytest.fixture()
def profitable_pnl():
    """Every simulated outcome is a gain: the lower tail is profitable."""
    return np.linspace(10.0, 110.0, 10_001)


def test_profitable_tail_gives_negative_raw_var_and_cvar(profitable_pnl):
    m = risk_metrics(profitable_pnl)
    assert m[95]["var"] < 0 and m[95]["var_is_loss"] is False
    assert m[95]["cvar"] < 0 and m[95]["cvar_is_loss"] is False
    assert m[99]["var"] < 0
    assert m[95]["quantile"] > 0


def test_profitable_tail_is_never_labelled_as_a_loss(profitable_pnl):
    m = risk_metrics(profitable_pnl)
    for kind in ("VaR", "CVaR"):
        for level in (95, 99):
            d = tail_metric_display(kind, level, m[level]["var" if kind == "VaR" else "cvar"], 10_000.0)
            assert d["is_loss"] is False
            assert d["value_text"] == "No loss"
            assert "no loss" in d["label"].lower()
            assert "-" not in d["value_text"]
            assert "+" in d["delta_text"] or "break-even" in d["delta_text"]


def test_format_tail_value_handles_every_sign():
    assert format_tail_value(1234.5) == "1,234.50"
    assert format_tail_value(0.0) == "No loss (break-even)"
    text = format_tail_value(-37.25)
    assert text == "No loss (gain of +37.25)"
    assert "-" not in text


def test_explanation_for_profitable_tail_does_not_claim_a_loss(profitable_pnl):
    m = risk_metrics(profitable_pnl)
    text = explain_tail_risk(m, horizon=5, investment=10_000.0)
    assert "no loss threshold" in text.lower()
    assert "should not exceed" not in text
    assert "gained at least" in text


def test_explanation_for_loss_case_states_the_loss_threshold():
    pnl = np.linspace(-500.0, 300.0, 10_001)
    m = risk_metrics(pnl)
    text = explain_tail_risk(m, horizon=21, investment=10_000.0)
    assert "should not exceed" in text
    assert f"{m[95]['var']:,.2f}" in text
    assert f"{m[95]['cvar']:,.2f}" in text
    assert "no loss threshold" not in text.lower()


def test_loss_display_uses_positive_amounts_and_percent_of_investment():
    d = tail_metric_display("VaR", 95, 1_250.0, 10_000.0)
    assert d["label"] == "95% VaR"
    assert d["value_text"] == "1,250.00"
    assert d["is_loss"] is True
    assert "12.50%" in d["delta_text"]


def test_mixed_case_var_gain_but_cvar_loss():
    """5th percentile is a small gain, yet a few extreme losses drag the tail mean negative."""
    pnl = np.concatenate([np.full(94, 50.0), np.full(3, 5.0), np.full(3, -400.0)])
    m = risk_metrics(pnl)
    assert m[95]["var"] < 0               # percentile is a gain
    assert m[95]["cvar"] > 0              # but the average of the worst outcomes is a loss
    text = explain_tail_risk(m, horizon=10, investment=1_000.0)
    assert "average **loss**" in text


def test_break_even_distribution():
    m = risk_metrics(np.zeros(1_000))
    assert m[95]["var"] == 0.0 and m[95]["cvar"] == 0.0
    d = tail_metric_display("VaR", 95, m[95]["var"], 1_000.0)
    assert d["value_text"] == "No loss"
    assert "break-even" in d["delta_text"]
    assert format_tail_value(m[95]["var"]) == "No loss (break-even)"


def test_chart_labels_do_not_call_a_gain_a_var(profitable_pnl):
    m = risk_metrics(profitable_pnl)
    assert tail_line_label(95, m[95], "VaR") == "5th pct (gain)"
    loss = risk_metrics(np.linspace(-500, 300, 1_001))
    assert tail_line_label(95, loss[95], "VaR") == "95% VaR"
    zero = risk_metrics(np.zeros(100))
    assert tail_line_label(95, zero[95], "VaR") == "5th pct (break-even)"


def test_realistic_pipeline_produces_a_profitable_tail():
    """Strong positive drift + tiny volatility over a short horizon: every path gains."""
    paths = simulate_gbm(100.0, mu=0.01, sigma=0.001, horizon=5, n_sims=20_000, seed=1)
    pnl = 10_000.0 * paths[-1] / 100.0 - 10_000.0
    m = risk_metrics(pnl)
    assert m[95]["var"] < 0 and m[99]["var"] < 0
    assert tail_metric_display("VaR", 95, m[95]["var"], 10_000.0)["value_text"] == "No loss"


# ---- Invalid inputs ---------------------------------------------------------
@pytest.mark.parametrize("bad", [
    np.array([]), np.array([1.0, np.nan, 3.0]), np.array([1.0, np.inf]),
    np.ones((3, 3)),
])
def test_risk_metrics_rejects_invalid_pnl(bad):
    with pytest.raises(ValueError):
        risk_metrics(bad)


def test_display_helpers_reject_bad_arguments():
    with pytest.raises(ValueError):
        tail_metric_display("Sharpe", 95, 1.0, 1_000.0)
    with pytest.raises(ValueError):
        tail_metric_display("VaR", 95, float("nan"), 1_000.0)
    with pytest.raises(ValueError):
        tail_metric_display("VaR", 95, 1.0, 0.0)
    with pytest.raises(ValueError):
        format_tail_value(float("inf"))
