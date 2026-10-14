"""
data/fred.py

Free macro and rates series from FRED, e.g. DTB3 (3-month T-bill, % per year).
No API key needed for this download link. Full history each time (small).
"""

import io
import urllib.request

import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def download(series: str) -> str:
    req = urllib.request.Request(URL.format(series=series), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def parse(text: str, series: str) -> pd.DataFrame:
    """Raw CSV text -> clean rows: date, symbol, value. Missing days ('.') dropped."""
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]                  # 'observation_date' (or 'DATE' in older files)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "symbol": series,
        "value": pd.to_numeric(df[series], errors="coerce"),
    })
    return out.dropna(subset=["value"])
