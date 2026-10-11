"""
src/backtest/portfolio_sim.py

Daily portfolio simulator for weight-based strategies (the counterpart to
the event-driven FX engine in engine.py, which simulates individual trades).

Timing convention
-----------------
* Target weights dated t are decided with information up to the close of t
  and traded AT the close of t.
* They therefore earn returns from t+1 onwards. No lookahead by construction.
* Between rebalance dates, weights drift with prices (no daily rebalancing).
* Uninvested capital (1 - sum of weights) sits in cash at a 0% return in v0.

Costs
-----
cost_t = turnover_t x cost_bps / 10,000, charged on the trade date, where
turnover_t = sum |target weight - drifted weight before trading|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    returns: pd.Series          # daily net portfolio returns
    gross_returns: pd.Series    # before costs
    weights: pd.DataFrame       # end-of-day weights after any trade
    turnover: pd.Series         # one-way turnover on each day (0 if no trade)
    costs: pd.Series            # cost deducted on each day (as a return)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


def simulate(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = 5.0,
) -> SimResult:
    """Run the daily simulation.

    Args:
        target_weights: weights on rebalance dates only (index subset of prices).
        prices:         adjusted close panel covering the full period.
        cost_bps:       one-way transaction cost per unit of turnover.
    """
    cols = list(target_weights.columns)
    prices = prices[cols].sort_index()
    rets = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
    idx = prices.index

    start = target_weights.index.min()
    if pd.isna(start):
        raise ValueError("target_weights is empty")
    targets = target_weights.reindex(idx)
    is_rebal = targets.notna().any(axis=1).to_numpy()
    tgt = targets.fillna(0.0).to_numpy()

    n_days, n_assets = rets.shape
    w = np.zeros(n_assets)               # weights held overnight into day t
    out_w = np.zeros((n_days, n_assets))
    gross = np.zeros(n_days)
    turnover = np.zeros(n_days)
    costs = np.zeros(n_days)
    first = idx.get_loc(start)

    for t in range(first, n_days):
        # 1) earn day-t returns on yesterday's closing weights
        if t > first:
            r = rets[t]
            port = float(w @ r)
            gross[t] = port
            # drift: each position grows with its asset, the whole book with the portfolio
            w = w * (1.0 + r) / (1.0 + port)
        # 2) trade to target at the close of t
        if is_rebal[t]:
            turnover[t] = float(np.abs(tgt[t] - w).sum())
            costs[t] = turnover[t] * cost_bps / 1e4
            w = tgt[t].copy()
        out_w[t] = w

    sl = slice(first, None)
    gross_s = pd.Series(gross[sl], idx[sl], name="gross")
    cost_s = pd.Series(costs[sl], idx[sl], name="costs")
    return SimResult(
        returns=(gross_s - cost_s).rename("net"),
        gross_returns=gross_s,
        weights=pd.DataFrame(out_w[sl], idx[sl], cols),
        turnover=pd.Series(turnover[sl], idx[sl], name="turnover"),
        costs=cost_s,
    )
