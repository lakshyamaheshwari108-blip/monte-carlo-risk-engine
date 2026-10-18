"""Shared pytest configuration: make the project root importable."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_prices():
    """A reproducible 5-year daily price history with fat-tailed returns."""
    rng = np.random.default_rng(0)
    n = 1250
    returns = rng.standard_t(4, n) * 0.012 + 0.0004
    prices = 150.0 * np.exp(np.cumsum(returns))
    index = pd.bdate_range("2021-10-01", periods=n)
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture()
def market_cfg():
    """A valid default configuration for build_market_results."""
    return {
        "model": "Geometric Brownian Motion (GBM)",
        "drift_mode": "Historical average return",
        "investment": 10_000.0,
        "horizon": 21,
        "seed": 42,
        "kappa": 2.0, "xi": 0.30, "rho": -0.70,
        "theta": 0.04, "v0": 0.04, "calibrate": True,
        "strike": 0.0, "rf": 0.04,
        "n_sims": 20_000,
    }
