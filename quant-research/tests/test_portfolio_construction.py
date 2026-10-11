import numpy as np
import pandas as pd

from src.portfolio.construction import construct, rebalance_dates
from src.strategy import PortfolioConfig


def test_month_end_rebalance_dates(synthetic_prices):
    d = rebalance_dates(synthetic_prices.index, "M")
    assert len(d) == synthetic_prices.index.to_period("M").nunique()
    assert all(d.to_series().dt.is_month_end | (d.month != (d + pd.offsets.BDay()).month))


def test_long_only_caps_and_gross(synthetic_prices):
    fc = pd.DataFrame(1.0, synthetic_prices.index, synthetic_prices.columns)
    fc["CCC"] = -1.0
    cfg = PortfolioConfig(target_vol=0.5, max_weight=0.3, max_gross=1.0, long_only=True, rebalance="M")
    w = construct(fc, synthetic_prices, cfg)
    assert (w >= 0).all().all()
    assert (w["CCC"] == 0).all()
    assert (w <= 0.3 + 1e-12).all().all()
    assert (w.sum(axis=1) <= 1.0 + 1e-12).all()


def test_untradable_asset_gets_zero(synthetic_prices):
    fc = pd.DataFrame(1.0, synthetic_prices.index, synthetic_prices.columns)
    w = construct(fc, synthetic_prices, PortfolioConfig(rebalance="D"))
    assert (w["EEE"].iloc[:300] == 0).all()


def test_inverse_vol_sizing(synthetic_prices):
    """With equal forecasts and no caps binding, weight x vol is equal across assets."""
    p = synthetic_prices[["AAA", "BBB"]]
    fc = pd.DataFrame(1.0, p.index, p.columns)
    cfg = PortfolioConfig(target_vol=0.05, max_weight=1.0, max_gross=10.0, rebalance="D")
    w = construct(fc, p, cfg).iloc[-1]
    vol = p.pct_change().iloc[-63:].std() * np.sqrt(252)
    np.testing.assert_allclose((w * vol).values, (w * vol).iloc[0], rtol=1e-9)
