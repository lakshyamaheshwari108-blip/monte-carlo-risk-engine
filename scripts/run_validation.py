"""
Reproduce the numerical validation tables in docs/VALIDATION.md.

    python scripts/run_validation.py

Everything is seeded, so the output is reproducible. It takes roughly a minute.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_engine import (  # noqa: E402
    black_scholes,
    feller_condition_holds,
    gbm_convergence_study,
    heston_call_price,
    heston_convergence_study,
    price_european_options,
    simulate_gbm_terminal,
    simulate_heston,
)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def section_gbm():
    print("### 1. GBM Monte Carlo vs Black-Scholes\n")
    print("S0 = 100, K = 100, r = 4%, sigma = 20%, T = 0.5 years. Black-Scholes call = "
          f"{float(black_scholes(100, 100, 0.04, 0.2, 0.5)[0]):.4f}.\n")
    df = gbm_convergence_study(path_counts=(2_000, 8_000, 32_000, 128_000, 512_000, 2_048_000), seed=7)
    print(table(["Paths", "MC call", "1.96 x SE", "Error", "z-score"],
                [[f"{int(r.paths):,}", f"{r.mc_call:.4f}", f"{1.96 * r.call_se:.4f}",
                  f"{r.call_error:+.4f}", f"{r.call_z:+.2f}"] for r in df.itertuples()]))
    print()
    bs_call = float(black_scholes(100, 100, 0.04, 0.2, 0.5)[0])
    reps, rows = 400, []
    for n in (2_000, 8_000, 32_000):
        beyond = 0
        for k in range(reps):
            t = simulate_gbm_terminal(100, 0.04, 0.2, 0.5, n, seed=1000 + 7 * k + n)
            m = price_european_options(t, 100, 0.04, 0.5)
            beyond += abs((m["call"] - bs_call) / m["call_se"]) > 3
        rows.append([f"{n:,}", f"{beyond / reps:.2%}"])
    print("Calibration of the z-statistic: share of 400 independent replications with |z| > 3 "
          "(normal theory: 0.27%).\n")
    print(table(["Paths per replication", "Share with abs(z) > 3"], rows))
    print()
    n = 1_000_000
    terminal = simulate_gbm_terminal(100, 0.04, 0.2, 0.5, n, seed=99)
    mc = price_european_options(terminal, 100, 0.04, 0.5)
    disc = np.exp(-0.04 * 0.5) * terminal
    se = disc.std(ddof=1) / np.sqrt(n)
    parity = (mc["call"] - mc["put"]) - (100 - 100 * np.exp(-0.04 * 0.5))
    print(f"\nWith {n:,} paths: discounted mean terminal price = {disc.mean():.4f} "
          f"(spot 100, 95% CI +/- {1.96 * se:.4f}); put-call parity residual = {parity:+.4f} "
          f"(1.96 x SE = {1.96 * se:.4f}).\n")


def section_reference():
    print("### 2. The semi-analytic Heston reference\n")
    worst = 0.0
    for years in (0.25, 1.0, 3.0):
        for strike in (80.0, 100.0, 120.0):
            h = heston_call_price(100, strike, 0.03, years, 0.04, 3.0, 0.04, 0.001, 0.0)
            b = float(black_scholes(100, strike, 0.03, 0.2, years)[0])
            worst = max(worst, abs(h - b))
    print(f"With vol-of-vol xi = 0.001 and rho = 0, the largest gap to Black-Scholes over 9 "
          f"maturity/strike combinations is {worst:.1e}.\n")

    def gap(xi):
        return abs(heston_call_price(100, 120.0, 0.03, 1.0, 0.04, 3.0, 0.04, xi, 0.0)
                   - float(black_scholes(100, 120.0, 0.03, 0.2, 1.0)[0]))
    print(f"Quadrupling xi (0.002 -> 0.008) multiplies the gap by {gap(0.008) / gap(0.002):.1f} "
          "(theory for rho = 0: 16, i.e. error proportional to xi squared).\n")


CASES = {
    "A: Feller holds, rho = -0.7": dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.7),
    "B: Feller holds, rho = +0.3": dict(v0=0.03, kappa=3.0, theta=0.06, xi=0.40, rho=0.3),
    "C: Feller violated": dict(v0=0.04, kappa=1.0, theta=0.04, xi=0.60, rho=-0.7),
    "D: Feller badly violated": dict(v0=0.04, kappa=0.5, theta=0.04, xi=1.00, rho=-0.9),
}


def section_heston_cases():
    print("### 3. Heston Monte Carlo vs semi-analytic price\n")
    print("S0 = 100, K = 100, r = 3%, T = 0.5, 200,000 paths per run.\n")
    rows = []
    for name, p in CASES.items():
        ref = heston_call_price(100, 100, 0.03, 0.5, **p)
        cells = [name, "yes" if feller_condition_holds(p["kappa"], p["theta"], p["xi"]) else "no",
                 f"{ref:.4f}"]
        for steps in (1, 4):
            t, _, _ = simulate_heston(100, 0.03, p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"],
                                      126, 200_000, seed=5, keep_paths=False, steps_per_day=steps)
            mc = price_european_options(t, 100, 0.03, 0.5)
            cells.append(f"{mc['call']:.4f} ({(mc['call'] - ref) / mc['call_se']:+.1f} SE)")
        rows.append(cells)
    print(table(["Parameter set", "Feller holds", "Reference", "MC, 1 step/day", "MC, 4 steps/day"], rows))
    print()


def section_convergence():
    print("### 4. Time-step convergence under severe Feller violation\n")
    print("T = 0.25, 150,000 paths per run.\n")
    for name in ("C: Feller violated", "D: Feller badly violated"):
        df = heston_convergence_study(years=0.25, steps_per_day=(1, 2, 4, 8, 16), n_paths=150_000,
                                      seed=21, r=0.03, **CASES[name])
        print(f"**{name}** (reference call = {df['reference'].iloc[0]:.4f})\n")
        print(table(["Steps/day", "MC call", "1.96 x SE", "Error", "z-score"],
                    [[int(r.steps_per_day), f"{r.mc_call:.4f}", f"{1.96 * r.call_se:.4f}",
                      f"{r.error:+.4f}", f"{r.z:+.2f}"] for r in df.itertuples()]))
        print()


if __name__ == "__main__":
    section_gbm()
    section_reference()
    section_heston_cases()
    section_convergence()
