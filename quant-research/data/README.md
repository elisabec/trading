# Data

Get raw data and turn it into clean panels (pandas DataFrame: date x asset).

One loader per source (e.g. `yahoo.py`, `fred.py`, `edgar.py`). Cleaning lives here too: missing values, outliers, resampling. Downloaded files go to `data/cache/` (git-ignored).
