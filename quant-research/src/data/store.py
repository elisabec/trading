"""
src/data/store.py

Local Parquet cache for price panels.

Every dataset is a "wide" DataFrame: DatetimeIndex (one row per trading day),
one column per instrument. Strategies and the backtester only ever see this
shape, regardless of where the data came from.

The cache lives in ``quant-research/data/`` (git-ignored). Override with the
``QR_DATA_DIR`` environment variable, e.g. to point at an external drive.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data"


def data_dir() -> Path:
    """Return (and create) the cache directory."""
    path = Path(os.environ.get("QR_DATA_DIR", _DEFAULT_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(name: str) -> Path:
    return data_dir() / f"{name}.parquet"


def save_panel(df: pd.DataFrame, name: str) -> Path:
    """Write a wide price panel to ``<data_dir>/<name>.parquet``."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("panel index must be a DatetimeIndex")
    out = df.sort_index()
    out.index.name = "date"
    path = _path(name)
    out.to_parquet(path)
    return path


def load_panel(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a cached panel. Raises FileNotFoundError if it was never saved."""
    path = _path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached dataset '{name}' in {data_dir()}. "
            "Run the loader first (e.g. scripts/run_backtest.py --refresh)."
        )
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    if columns is not None:
        missing = sorted(set(columns) - set(df.columns))
        if missing:
            raise KeyError(f"'{name}' has no columns {missing}; refresh the cache")
        df = df[columns]
    return df


def has_panel(name: str) -> bool:
    return _path(name).exists()
