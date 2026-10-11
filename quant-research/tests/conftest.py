import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Five random-walk assets over ~6 years of business days, one trending."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    drift = np.array([0.0006, 0.0, -0.0003, 0.0002, 0.0])
    vol = np.array([0.010, 0.006, 0.012, 0.008, 0.004])
    rets = rng.normal(drift, vol, size=(len(idx), len(drift)))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    df = pd.DataFrame(prices, idx, ["AAA", "BBB", "CCC", "DDD", "EEE"])
    df.iloc[:300, 4] = np.nan  # EEE lists later, like a young ETF
    return df
