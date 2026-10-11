"""
Run a strategy backtest end to end.

    cd quant-research
    python scripts/run_backtest.py --strategy trend_etf --refresh   # download + run
    python scripts/run_backtest.py --strategy trend_etf             # rerun from cache

Writes results/<strategy>/{summary.md, tearsheet.png, latest_weights.csv}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import strategies  # noqa: E402
from src import pipeline  # noqa: E402
from src.data import has_panel, load_panel, save_panel  # noqa: E402
from src.data.sources import fetch_yahoo_prices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="trend_etf")
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    ap.add_argument("--start", default=None, help="first trade date, e.g. 2008-01-01")
    args = ap.parse_args()

    strat = strategies.get(args.strategy)
    dataset = f"prices_{strat.name}"

    if args.refresh or not has_panel(dataset):
        print(f"Downloading {len(strat.universe)} tickers ...")
        save_panel(fetch_yahoo_prices(strat.universe), dataset)
    prices = load_panel(dataset, columns=strat.universe)

    result = pipeline.run(strat, prices, start=args.start)
    out = pipeline.write_report(result, prices, ROOT / "results" / strat.name)

    print(result.stats.to_string())
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
