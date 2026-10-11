"""
src/report/metrics.py

Performance statistics from a daily return series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def summary(returns: pd.Series, turnover: pd.Series | None = None) -> pd.Series:
    """Headline stats. Annualised with 252 trading days; risk-free rate = 0."""
    r = returns.dropna()
    years = len(r) / TRADING_DAYS
    total = float((1.0 + r).prod())
    cagr = total ** (1.0 / years) - 1.0 if years > 0 and total > 0 else np.nan
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    downside = float(r[r < 0].std() * np.sqrt(TRADING_DAYS))
    mean_ann = float(r.mean() * TRADING_DAYS)
    dd = drawdown(r)
    max_dd = float(dd.min())

    stats = {
        "start": r.index.min().date(),
        "end": r.index.max().date(),
        "years": round(years, 1),
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": mean_ann / vol if vol > 0 else np.nan,
        "sortino": mean_ann / downside if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "pct_positive_months": float((r.groupby(r.index.to_period("M")).apply(
            lambda x: (1 + x).prod() - 1) > 0).mean()),
    }
    if turnover is not None:
        stats["ann_turnover"] = float(turnover.sum() / years) if years > 0 else np.nan
    return pd.Series(stats, name=returns.name or "strategy")


def format_summary(stats: pd.Series) -> str:
    """Markdown table for the README / tear sheet."""
    pct = {"cagr", "ann_vol", "max_drawdown", "pct_positive_months"}
    lines = ["| Metric | Value |", "| --- | --- |"]
    for k, v in stats.items():
        if k in pct:
            v = f"{v:.1%}"
        elif isinstance(v, float):
            v = f"{v:.2f}"
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
