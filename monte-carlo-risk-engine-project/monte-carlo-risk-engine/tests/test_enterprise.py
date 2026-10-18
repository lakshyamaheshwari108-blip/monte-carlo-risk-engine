"""Enterprise cash-flow stress test: deterministic closed forms, monotonicity, statistics."""
import numpy as np
import pytest

from risk_engine import (
    DEFAULT_SCENARIO,
    ENT_MONTHS,
    SCENARIOS,
    cash_stats,
    compare_scenarios,
    simulate_cash_flows,
    wilson_interval,
)

BASE = dict(revenue=100_000.0, fixed=40_000.0, var_ratio=0.45, vol=0.12, opening=40_000.0,
            shock=0.0, growth=0.0, debt=400_000.0, rate=0.06)


def sim(n_paths=5_000, seed=42, **overrides):
    args = dict(BASE); args.update(overrides)
    return simulate_cash_flows(n_paths=n_paths, seed=seed, **args)


# ---- Deterministic (zero-volatility) cases ----------------------------------
def test_zero_volatility_cash_path_matches_the_closed_form():
    cash = sim(n_paths=3, vol=0.0, shock=-0.10, growth=-0.01)
    months = np.arange(ENT_MONTHS)
    revenue = 100_000.0 * 0.9 * 0.99**months
    net = revenue * 0.55 - 40_000.0 - 400_000.0 * 0.06 / 12.0
    expected = 40_000.0 + np.concatenate([[0.0], np.cumsum(net)])
    for column in cash.T:
        assert np.allclose(column, expected, rtol=1e-12)


def test_known_insolvency_month_is_found_exactly():
    # net cash flow = -30 per month, opening 100 -> 70, 40, 10, -20 (month 4)
    cash = simulate_cash_flows(revenue=100.0, fixed=130.0, var_ratio=0.0, vol=0.0, opening=100.0,
                               shock=0.0, growth=0.0, debt=0.0, rate=0.0, months=12, n_paths=50, seed=1)
    stats = cash_stats(cash)
    assert stats["prob"] == 1.0
    assert stats["insolvent_paths"] == 50
    assert stats["avg_first_month"] == 4.0
    assert list(stats["prob_by_month"][:5]) == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert stats["mean_end"] == pytest.approx(100.0 - 12 * 30.0)


def test_comfortably_solvent_business_has_zero_insolvency():
    cash = simulate_cash_flows(revenue=100.0, fixed=10.0, var_ratio=0.2, vol=0.0, opening=50.0,
                               shock=0.0, growth=0.0, debt=0.0, rate=0.0, months=12, n_paths=100, seed=1)
    stats = cash_stats(cash)
    assert stats["prob"] == 0.0
    assert stats["avg_first_month"] is None
    assert stats["buffer_95"] == 0.0
    assert stats["ci_low"] == 0.0 and stats["ci_high"] > 0.0


def test_interest_cost_is_debt_times_rate_over_twelve():
    with_debt = sim(n_paths=10, vol=0.0, debt=1_200_000.0, rate=0.12)
    no_debt = sim(n_paths=10, vol=0.0, debt=0.0, rate=0.12)
    monthly_interest = 1_200_000.0 * 0.12 / 12.0
    assert np.allclose(no_debt[1] - with_debt[1], monthly_interest)


# ---- Statistical properties -------------------------------------------------
def test_lognormal_revenue_preserves_the_expected_value_and_stays_positive():
    n = 200_000
    cash = simulate_cash_flows(revenue=100.0, fixed=0.0, var_ratio=0.0, vol=0.30, opening=0.0,
                               shock=-0.2, growth=0.01, debt=0.0, rate=0.0, months=6,
                               n_paths=n, seed=3)
    monthly_revenue = np.diff(cash, axis=0)
    assert np.all(monthly_revenue > 0)
    for t in range(6):
        expected = 100.0 * 0.8 * 1.01**t
        se = monthly_revenue[t].std(ddof=1) / np.sqrt(n)
        assert abs(monthly_revenue[t].mean() - expected) < 4 * se


def test_same_seed_reproduces_exactly():
    assert np.array_equal(sim(seed=7), sim(seed=7))
    assert not np.array_equal(sim(seed=7), sim(seed=8))


def test_insolvency_never_increases_with_more_opening_cash():
    probs = [cash_stats(sim(opening=o))["prob"] for o in (0, 10_000, 25_000, 50_000, 100_000)]
    assert all(a >= b for a, b in zip(probs, probs[1:]))     # exact: same random draws


@pytest.mark.parametrize("name,values", [
    ("debt", (0.0, 300_000.0, 800_000.0, 1_500_000.0)),
    ("rate", (0.0, 0.05, 0.10, 0.20)),
    ("var_ratio", (0.30, 0.45, 0.55, 0.65)),
    ("fixed", (30_000.0, 40_000.0, 50_000.0)),
])
def test_insolvency_never_decreases_when_costs_rise(name, values):
    probs = [cash_stats(sim(opening=60_000.0, **{name: v}))["prob"] for v in values]
    assert all(a <= b for a, b in zip(probs, probs[1:]))


def test_insolvency_rises_with_revenue_volatility():
    # healthy expected cash flow, so only volatility can push the business under
    probs = [cash_stats(sim(n_paths=20_000, opening=15_000.0, fixed=48_000.0, vol=v))["prob"]
             for v in (0.05, 0.15, 0.30, 0.50)]
    assert probs[0] < probs[1] < probs[2] < probs[3]


def test_larger_revenue_shock_means_more_insolvency():
    probs = [cash_stats(sim(shock=s))["prob"] for s in (0.0, -0.10, -0.25)]
    assert probs[0] <= probs[1] <= probs[2] and probs[2] > probs[0]


# ---- cash_stats invariants --------------------------------------------------
def test_cash_stats_internal_consistency():
    stats = cash_stats(sim(n_paths=20_000, opening=20_000.0, vol=0.25))
    assert 0 < stats["prob"] < 1
    assert stats["prob_by_month"][-1] == pytest.approx(stats["prob"])
    assert np.all(np.diff(stats["prob_by_month"]) >= 0)               # cumulative
    assert np.all(stats["prob_negative_at_month"] <= stats["prob_by_month"] + 1e-12)
    assert stats["es5_end"] <= stats["p5_end"] <= stats["median_end"]
    assert stats["ci_low"] <= stats["prob"] <= stats["ci_high"]
    assert stats["insolvent_paths"] == round(stats["prob"] * stats["n"])
    assert 1.0 <= stats["avg_first_month"] <= ENT_MONTHS
    assert len(stats["median_by_month"]) == len(stats["p5_by_month"]) == ENT_MONTHS


def test_buffer_95_really_gets_insolvency_down_to_five_percent():
    base = sim(n_paths=20_000, opening=10_000.0, fixed=48_000.0, vol=0.25)
    stats = cash_stats(base)
    assert stats["prob"] > 0.10 and stats["buffer_95"] > 0
    topped_up = cash_stats(base + stats["buffer_95"])
    assert topped_up["prob"] <= 0.05 + 1e-12


def test_first_insolvency_month_is_the_earliest_negative_month():
    cash = np.array([[100.0, 100.0, 100.0],
                     [50.0, -10.0, 90.0],        # path 2 goes negative in month 1
                     [20.0, 30.0, 80.0],
                     [-5.0, 10.0, 70.0]])        # path 1 goes negative in month 3
    stats = cash_stats(cash)
    assert stats["insolvent_paths"] == 2
    assert stats["prob"] == pytest.approx(2 / 3)
    assert stats["avg_first_month"] == pytest.approx((3 + 1) / 2)
    assert list(stats["prob_by_month"]) == pytest.approx([1 / 3, 1 / 3, 2 / 3])


# ---- Wilson interval --------------------------------------------------------
def test_wilson_interval_known_values():
    low, high = wilson_interval(10, 100)
    assert low == pytest.approx(0.0552, abs=1e-3) and high == pytest.approx(0.1744, abs=1e-3)
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)


def test_wilson_interval_edges_and_shrinkage():
    low0, high0 = wilson_interval(0, 100)
    assert low0 == 0.0 and 0 < high0 < 0.05
    lown, highn = wilson_interval(100, 100)
    assert highn == pytest.approx(1.0) and lown > 0.95
    narrow = wilson_interval(500, 10_000)
    wide = wilson_interval(5, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_has_close_to_nominal_coverage():
    rng = np.random.default_rng(0)
    p_true, n, hits, trials = 0.12, 2_000, 0, 1_000
    for _ in range(trials):
        k = int(rng.binomial(n, p_true))
        low, high = wilson_interval(k, n)
        hits += low <= p_true <= high
    assert 0.93 < hits / trials < 0.97


# ---- Scenario presets -------------------------------------------------------
def test_scenario_presets_are_complete_and_in_valid_ranges():
    assert DEFAULT_SCENARIO in SCENARIOS
    assert len(SCENARIOS) == 4
    for name, p in SCENARIOS.items():
        assert {"vol", "fixed_pct", "var_ratio", "shock", "growth", "rate", "desc"} <= set(p), name
        assert 0 <= p["vol"] <= 100 and 0 <= p["var_ratio"] < 100
        assert -90 <= p["shock"] <= 100 and -20 <= p["growth"] <= 20
        assert 0 <= p["rate"] <= 50 and p["fixed_pct"] >= 0
        assert len(p["desc"]) > 20


def test_rate_spike_preset_is_exactly_300_bps_above_baseline():
    assert SCENARIOS["Rate Spike (+300 bps)"]["rate"] - SCENARIOS["Baseline (Normal)"]["rate"] == 3.0


def test_scenario_severity_ordering_for_the_default_business():
    rows = {r["name"]: r["prob"] for r in compare_scenarios(100_000.0, 40_000.0, 400_000.0, 5_000, 42)}
    assert rows["Baseline (Normal)"] < 0.01
    assert (rows["Baseline (Normal)"] < rows["Rate Spike (+300 bps)"]
            < rows["Supply Chain Shock"] < rows["2008 Financial Crisis"])
    assert rows["2008 Financial Crisis"] > 0.80


def test_compare_scenarios_is_reproducible_and_complete():
    a = compare_scenarios(100_000.0, 40_000.0, 400_000.0, 2_000, 1)
    b = compare_scenarios(100_000.0, 40_000.0, 400_000.0, 2_000, 1)
    assert a == b
    assert [r["name"] for r in a] == list(SCENARIOS)
    assert all({"name", "prob", "mean_end", "p5_end"} <= set(r) for r in a)


def test_trial_counts_up_to_the_largest_option_run():
    stats = cash_stats(sim(n_paths=50_000))
    assert stats["n"] == 50_000
