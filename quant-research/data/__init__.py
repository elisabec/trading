"""
data — one way in, whatever the source.

    from data import load
    vix = load("indices", ["VIX"])                 # panel: date x symbol, column 'close'
    tbill = load("rates", ["DTB3"], field="value")
"""

import pandas as pd

from data.store import read_clean


def load(dataset: str, symbols: list[str], field: str = "close", start: str | None = None) -> pd.DataFrame:
    """Read from the local store and return a panel (dates as rows, symbols as columns)."""
    df = read_clean(dataset)
    df = df[df["symbol"].isin(symbols)]
    if start:
        df = df[df["date"] >= start]
    return df.pivot(index="date", columns="symbol", values=field)[symbols]
