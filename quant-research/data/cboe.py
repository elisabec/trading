"""
data/cboe.py

Free daily index history from Cboe, e.g. PUT (PutWrite index) and VIX.
Each file is the full history (small), so every update re-downloads it.

File formats:
    PUT: DATE,PUT                     (date as MM/DD/YYYY)
    VIX: DATE,OPEN,HIGH,LOW,CLOSE
"""

import io
import urllib.request

import pandas as pd

URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"


def download(symbol: str) -> str:
    req = urllib.request.Request(URL.format(symbol=symbol), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def parse(text: str, symbol: str) -> pd.DataFrame:
    """Raw CSV text -> clean rows: date, symbol, open, high, low, close."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "CLOSE" not in df.columns:              # single-value files like PUT
        df = df.rename(columns={symbol: "CLOSE"})
    out = pd.DataFrame({
        "date": pd.to_datetime(df["DATE"], format="%m/%d/%Y"),
        "symbol": symbol,
    })
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(df[col.upper()], errors="coerce") if col.upper() in df else float("nan")
    return out.dropna(subset=["close"])
