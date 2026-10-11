"""
src/data/sources.py

Loaders that fetch raw data from free sources and return wide panels.

Add a new source here only when a strategy needs it.
"""

from __future__ import annotations

import pandas as pd


def fetch_yahoo_prices(tickers: list[str], start: str = "2000-01-01") -> pd.DataFrame:
    """Daily total-return prices (split- and dividend-adjusted closes) from Yahoo.

    Returns a wide DataFrame (date x ticker). Days before a ticker's inception
    are NaN; downstream code treats NaN as "not tradable yet".
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # keep yfinance optional for tests/CI
        raise ImportError("pip install yfinance to download prices") from exc

    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,   # adjusted for splits and dividends
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {tickers}")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.reindex(columns=tickers)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    return prices.sort_index().dropna(how="all")
