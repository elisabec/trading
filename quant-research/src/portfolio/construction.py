"""
src/portfolio/construction.py

Portfolio construction: turn forecasts in [-1, 1] into target weights.

Method (deliberately simple, v0)
--------------------------------
1. Each instrument gets a risk budget of target_vol / sqrt(N), where N is the
   number of instruments tradable that day. This is what each would need if
   they were uncorrelated and should sum to the portfolio target.
2. Raw weight = forecast x risk budget / instrument vol (inverse-vol sizing).
3. Long-only: negative weights -> 0 (that capital sits in cash).
4. Cap each weight at max_weight, then scale everything down if gross
   exposure exceeds max_gross. Weights never scale UP, so a quiet market
   leaves cash rather than adding leverage.
5. Keep only rebalance dates; the backtester holds (and lets weights drift)
   in between.

Correlations are ignored on purpose in v0. Add a covariance-aware method
when a strategy needs it, not before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import PortfolioConfig

TRADING_DAYS = 252


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day of each period in ``index`` ("D", "W" or "M")."""
    if freq == "D":
        return index
    if freq not in ("W", "M"):
        raise ValueError(f"rebalance must be 'D', 'W' or 'M', got {freq!r}")
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period(freq)).max().values)


def ex_ante_vol(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Annualised rolling vol of daily returns, known at each date's close."""
    rets = prices.pct_change(fill_method=None)
    return rets.rolling(lookback, min_periods=max(20, lookback // 2)).std() * np.sqrt(TRADING_DAYS)


def construct(forecasts: pd.DataFrame, prices: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    """Target weights on rebalance dates (other dates dropped).

    Returns a DataFrame indexed by rebalance date, one column per instrument.
    Untradable instruments get weight 0.
    """
    forecasts = forecasts.reindex_like(prices).clip(-1.0, 1.0)
    vol = ex_ante_vol(prices, cfg.vol_lookback)

    tradable = forecasts.notna() & vol.notna() & (vol > 0)
    n = tradable.sum(axis=1).replace(0, np.nan)
    budget = cfg.target_vol / np.sqrt(n)

    w = (forecasts * (1.0 / vol)).mul(budget, axis=0)
    w = w.where(tradable, 0.0).fillna(0.0)

    if cfg.long_only:
        w = w.clip(lower=0.0)
    w = w.clip(-cfg.max_weight, cfg.max_weight)

    gross = w.abs().sum(axis=1)
    scale = (cfg.max_gross / gross).where(gross > cfg.max_gross, 1.0)
    w = w.mul(scale, axis=0)

    dates = rebalance_dates(prices.index, cfg.rebalance)
    return w.loc[w.index.intersection(dates)]
