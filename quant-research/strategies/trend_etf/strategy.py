"""
strategies/trend_etf/strategy.py

Cross-asset time-series momentum on liquid ETFs, vol-targeted, long-only.

Alpha model
-----------
For each ETF and each lookback L in {1, 3, 6, 12} months, take the sign of the
trailing L-day total return. The forecast is the average of those signs, so it
lies in [-1, 1]: +1 when every horizon is up, 0 when they disagree evenly.

Sizing, the long-only rule and monthly rebalancing are NOT here; they live in
the shared portfolio layer and are configured below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.strategy import PortfolioConfig, Strategy

UNIVERSE = [
    "SPY",  # US large cap
    "EFA",  # developed ex-US equities
    "EEM",  # emerging-market equities
    "EWZ",  # Brazil
    "ILF",  # Latin America 40
    "IEF",  # 7-10y US Treasuries
    "TLT",  # 20y+ US Treasuries
    "LQD",  # US investment-grade credit
    "TIP",  # US inflation-linked
    "GLD",  # gold
    "DBC",  # broad commodities
    "UUP",  # US dollar index
]


@dataclass
class TrendETF(Strategy):
    name: str = "trend_etf"
    universe: list[str] = field(default_factory=lambda: list(UNIVERSE))
    lookbacks: tuple[int, ...] = (21, 63, 126, 252)
    portfolio: PortfolioConfig = field(default_factory=lambda: PortfolioConfig(
        target_vol=0.10,
        vol_lookback=63,
        max_weight=0.20,
        max_gross=1.0,
        long_only=True,
        rebalance="M",
        cost_bps=5.0,
    ))

    def forecast(self, prices: pd.DataFrame) -> pd.DataFrame:
        signs = [np.sign(prices / prices.shift(L) - 1.0) for L in self.lookbacks]
        fc = sum(signs) / len(signs)
        # only forecast once the longest lookback is available
        return fc.where(prices.shift(max(self.lookbacks)).notna())
