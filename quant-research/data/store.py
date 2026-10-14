"""
data/store.py

The local data store. Lives outside the repo, in ~/qr-data
(change it with the environment variable QR_DATA_DIR).

    raw/<source>/<dataset>/<YYYY-MM-DD>.csv   exactly as downloaded, never edited
    clean/<dataset>.parquet                    one standard table per dataset

Clean tables are "long": one row per (date, symbol), value columns after that.
"""

import os
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd


def root() -> Path:
    return Path(os.environ.get("QR_DATA_DIR", Path.home() / "qr-data"))


def save_raw(text: str, source: str, dataset: str) -> Path:
    """Keep today's download exactly as received (the audit trail)."""
    path = root() / "raw" / source / dataset / f"{date.today()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def upsert_clean(df: pd.DataFrame, dataset: str) -> Path:
    """Merge new rows into clean/<dataset>.parquet. New values win on (date, symbol)."""
    path = root() / "clean" / f"{dataset}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df])
    df = (df.drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True))
    df.to_parquet(path, index=False)
    return path


def read_clean(dataset: str) -> pd.DataFrame:
    path = root() / "clean" / f"{dataset}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No dataset '{dataset}' yet. Run: python -m data.update")
    return pd.read_parquet(path)


def query(sql: str) -> pd.DataFrame:
    """SQL over all clean tables. Each table is available under its dataset name.

    Example: query("SELECT symbol, max(date) FROM indices GROUP BY symbol")
    """
    con = duckdb.connect()
    for path in (root() / "clean").glob("*.parquet"):
        con.execute(f"CREATE VIEW {path.stem} AS SELECT * FROM read_parquet('{path}')")
    return con.execute(sql).df()
