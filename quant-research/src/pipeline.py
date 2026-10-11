"""
src/pipeline.py

Runs one strategy end to end: forecast -> construct -> simulate -> report.
Used by scripts/run_backtest.py and by notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.backtest.portfolio_sim import SimResult, simulate
from src.portfolio.construction import construct
from src.report.metrics import format_summary, summary
from src.report.tearsheet import plot_tearsheet
from src.strategy import Strategy


@dataclass
class BacktestRun:
    strategy: Strategy
    forecasts: pd.DataFrame
    target_weights: pd.DataFrame
    sim: SimResult
    stats: pd.Series


def run(strategy: Strategy, prices: pd.DataFrame, start: str | None = None) -> BacktestRun:
    prices = prices[strategy.universe]
    forecasts = strategy.forecast(prices)
    targets = construct(forecasts, prices, strategy.portfolio)
    targets = targets[targets.abs().sum(axis=1) > 0]  # skip warm-up period with no positions
    if start is not None:
        targets = targets.loc[start:]
    sim = simulate(targets, prices, cost_bps=strategy.portfolio.cost_bps)
    stats = summary(sim.returns.rename(strategy.name), sim.turnover)
    return BacktestRun(strategy, forecasts, targets, sim, stats)


def write_report(result: BacktestRun, prices: pd.DataFrame, out_dir: str | Path,
                 benchmark: str | None = "SPY") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = None
    if benchmark and benchmark in prices:
        bench = prices[benchmark].pct_change(fill_method=None)

    plot_tearsheet(result.sim.returns, result.sim.weights,
                   title=f"{result.strategy.name}: net of {result.strategy.portfolio.cost_bps:g} bps costs",
                   out_path=out_dir / "tearsheet.png", benchmark=bench, benchmark_name=benchmark or "")

    cfg = result.strategy.portfolio
    lines = [
        f"# {result.strategy.name} backtest",
        "",
        format_summary(result.stats),
        "",
        f"Universe: {', '.join(result.strategy.universe)}  ",
        f"Portfolio: target vol {cfg.target_vol:.0%}, max weight {cfg.max_weight:.0%}, "
        f"long-only={cfg.long_only}, rebalance={cfg.rebalance}, costs {cfg.cost_bps:g} bps  ",
        "",
        "![tear sheet](tearsheet.png)",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))
    result.sim.weights.iloc[[-1]].T.rename(columns=lambda _: "weight").to_csv(out_dir / "latest_weights.csv")
    return out_dir
