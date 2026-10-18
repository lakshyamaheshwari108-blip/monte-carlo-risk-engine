"""Executive PDF report: generation must succeed for every scenario and awkward input."""
import numpy as np
import pytest

pytest.importorskip("fpdf")          # skipped automatically if fpdf2 is not installed

from report_pdf import build_pdf_report, pdf_safe        # noqa: E402
from risk_engine import SCENARIOS, cash_stats, simulate_cash_flows   # noqa: E402


def make_context(name="Baseline (Normal)", revenue=100_000.0, opening=40_000.0, debt=400_000.0,
                 trials=5_000, seed=42, description=None, **overrides):
    """Build the same context dictionary the Streamlit app passes to the report."""
    p = dict(SCENARIOS[name])
    params = {
        "revenue": revenue, "fixed": revenue * p["fixed_pct"] / 100.0,
        "var_ratio": p["var_ratio"], "vol": p["vol"], "opening": opening,
        "shock": p["shock"], "growth": p["growth"], "debt": debt, "rate": p["rate"],
    }
    params.update(overrides)
    cash = simulate_cash_flows(
        params["revenue"], params["fixed"], params["var_ratio"] / 100.0, params["vol"] / 100.0,
        params["opening"], params["shock"] / 100.0, params["growth"] / 100.0,
        params["debt"], params["rate"] / 100.0, n_paths=trials, seed=seed)
    return {"scenario": name, "description": description or p["desc"],
            "params": params, "stats": cash_stats(cash)}


def assert_valid_pdf(data):
    assert isinstance(data, (bytes, bytearray))
    assert bytes(data[:5]) == b"%PDF-"
    assert b"%%EOF" in bytes(data[-32:])
    assert len(data) > 2_000


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_report_is_a_valid_pdf_for_every_scenario(name):
    assert_valid_pdf(build_pdf_report(make_context(name)))


def test_report_works_with_the_largest_trial_count():
    assert_valid_pdf(build_pdf_report(make_context("2008 Financial Crisis", trials=50_000)))


def test_report_handles_a_business_that_never_fails():
    ctx = make_context("Baseline (Normal)", opening=500_000.0)
    assert ctx["stats"]["prob"] == 0.0 and ctx["stats"]["avg_first_month"] is None
    assert_valid_pdf(build_pdf_report(ctx))


def test_report_handles_a_business_that_always_fails():
    ctx = make_context("2008 Financial Crisis", opening=0.0, debt=5_000_000.0)
    assert ctx["stats"]["prob"] > 0.99
    assert_valid_pdf(build_pdf_report(ctx))


def test_report_handles_a_degenerate_zero_variance_distribution():
    ctx = make_context("Baseline (Normal)", vol=0.0)
    assert np.ptp(ctx["stats"]["ending"]) == pytest.approx(0.0, abs=1e-6)
    assert_valid_pdf(build_pdf_report(ctx))


def test_report_handles_very_large_and_very_small_numbers():
    big = make_context("Baseline (Normal)", revenue=5e11, opening=1e11, debt=1e12)
    assert_valid_pdf(build_pdf_report(big))
    tiny = make_context("Baseline (Normal)", revenue=10.0, opening=1.0, debt=0.0)
    assert_valid_pdf(build_pdf_report(tiny))


def test_report_survives_text_outside_latin_1():
    ctx = make_context("Baseline (Normal)",
                       description="Café stress – σ shock ≈ 2008 → “severe” 😀 日本")
    ctx["scenario"] = "Scénario – ünïcode ≈ test"
    assert_valid_pdf(build_pdf_report(ctx))


def test_report_survives_a_very_long_description():
    assert_valid_pdf(build_pdf_report(make_context(description="Long text. " * 400)))


# ---- pdf_safe ---------------------------------------------------------------
def test_pdf_safe_output_is_always_latin_1_encodable():
    for text in ["plain", "Café", "σ ≈ μ → ∞", "😀", "日本語", "—–−", "", "a\nb"]:
        pdf_safe(text).encode("latin-1")


def test_pdf_safe_replaces_common_symbols_with_readable_ascii():
    assert pdf_safe("a – b") == "a - b"
    assert pdf_safe("x ≈ y") == "x ~ y"
    assert pdf_safe("“quoted”") == '"quoted"'
    assert pdf_safe("σ") == "sigma"
    assert pdf_safe(123) == "123"
