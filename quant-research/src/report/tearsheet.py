"""
src/report/tearsheet.py

One-page tear sheet: equity curve vs a benchmark, drawdown, rolling vol,
and the weights over time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.report.metrics import TRADING_DAYS, drawdown  # noqa: E402


def plot_tearsheet(
    returns: pd.Series,
    weights: pd.DataFrame,
    title: str,
    out_path: str | Path,
    benchmark: pd.Series | None = None,
    benchmark_name: str = "benchmark",
) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2, 2]})
    ax_eq, ax_dd, ax_vol, ax_w = axes

    (1 + returns).cumprod().plot(ax=ax_eq, lw=1.6, label="strategy (net)")
    if benchmark is not None:
        b = benchmark.reindex(returns.index).fillna(0.0)
        (1 + b).cumprod().plot(ax=ax_eq, lw=1.0, color="0.55", label=benchmark_name)
    ax_eq.set_yscale("log")
    ax_eq.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax_eq.set_ylabel("growth of $1 (log)")
    ax_eq.legend(frameon=False)

    dd = drawdown(returns)
    ax_dd.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.35, lw=0)
    ax_dd.set_ylabel("drawdown")

    roll = returns.rolling(TRADING_DAYS // 4).std() * np.sqrt(TRADING_DAYS)
    roll.plot(ax=ax_vol, lw=1.0, color="#2c3e50")
    ax_vol.set_ylabel("63d vol")

    w = weights.loc[:, weights.abs().sum() > 0]
    ax_w.stackplot(w.index, w.clip(lower=0).T.values, labels=w.columns, alpha=0.85)
    ax_w.set_ylabel("weights")
    ax_w.legend(ncol=6, fontsize=8, frameon=False, loc="upper left")

    for ax in axes:
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
