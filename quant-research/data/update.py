"""
data/update.py — refresh the local store. Run once a day (or whenever):

    cd quant-research
    python -m data.update

Add a symbol: put it in the lists below.
"""

from data import cboe, fred, store

CBOE = ["PUT", "VIX"]       # -> clean/indices.parquet
FRED = ["DTB3"]             # -> clean/rates.parquet


def main() -> None:
    for symbol in CBOE:
        text = cboe.download(symbol)
        store.save_raw(text, "cboe", symbol)
        store.upsert_clean(cboe.parse(text, symbol), "indices")
        print(f"cboe  {symbol}: ok")

    for series in FRED:
        text = fred.download(series)
        store.save_raw(text, "fred", series)
        store.upsert_clean(fred.parse(text, series), "rates")
        print(f"fred  {series}: ok")

    print(store.query("SELECT symbol, min(date) AS first, max(date) AS last, count(*) AS rows "
                      "FROM indices GROUP BY symbol "
                      "UNION ALL SELECT symbol, min(date), max(date), count(*) FROM rates GROUP BY symbol"))


if __name__ == "__main__":
    main()
