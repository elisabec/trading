"""
src/strategy.py

The contract every strategy fulfils. This is the only thing a new strategy
has to implement; data, portfolio construction, backtest and reporting are
shared engine code.

Pipeline
--------
    prices ──► Strategy.forecast()        alpha model: WHAT do I believe?
           ──► portfolio.construct()       portfolio construction: HOW MUCH?
           ──► backtest.run()              simulation: what would have happened?
           ──► report                      metrics + tear sheet
    (later) ──► execution                   how to trade it: orders at the broker

Rules for forecast()
--------------------
* Input: wide price panel (date x instrument), adjusted closes.
* Output: same shape, values in [-1, 1]. +1 = maximum long conviction,
  -1 = maximum short conviction, 0 = no view. NaN = not tradable.
* The forecast on date t may only use prices up to and including t.
  The backtester then trades it at the close of t, so it earns returns
  from t+1 onwards. tests/test_no_lookahead.py checks this for every strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PortfolioConfig:
    """How forecasts become weights. Lives with each strategy's config."""

    target_vol: float = 0.10        # annualised portfolio vol target
    vol_lookback: int = 63          # days of returns for each asset's vol estimate
    max_weight: float = 0.25        # cap per instrument (absolute weight)
    max_gross: float = 1.0          # cap on sum of |weights| (1.0 = no leverage)
    long_only: bool = True          # IRA-compatible: negative forecasts become 0
    rebalance: str = "M"            # "D" daily, "W" weekly, "M" month-end
    cost_bps: float = 5.0           # one-way cost per unit of turnover, in bps


@dataclass
class Strategy:
    """Base class. Subclasses set name/universe and implement forecast()."""

    name: str
    universe: list[str]
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    def forecast(self, prices: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError
