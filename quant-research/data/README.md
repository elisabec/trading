# Data

Get raw data and turn it into clean tables in a local store. Research code only ever reads from the store.

**Store** (outside the repo, `~/qr-data`, change with `QR_DATA_DIR`):

```
raw/<source>/<dataset>/<date>.csv   exactly as downloaded (audit trail)
clean/indices.parquet               date | symbol | open | high | low | close
clean/rates.parquet                 date | symbol | value
```

**Files**

| File | Job |
|---|---|
| `store.py` | save raw, merge into clean, read, SQL via DuckDB |
| `cboe.py` | Cboe index history (PUT, VIX) |
| `fred.py` | FRED series (DTB3 T-bill rate) |
| `update.py` | refresh everything: `python -m data.update` |
| `__init__.py` | `load()`: one way in, returns a panel |

**Use**

```python
from data import load
from data.store import query

vix = load("indices", ["VIX"])                       # date x symbol
query("SELECT symbol, max(date) FROM indices GROUP BY symbol")
```

New source: add a `<source>.py` with `download()` and `parse()`, then list its symbols in `update.py`.
