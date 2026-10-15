# Backtest

Simulate what would have happened: target weights + prices -> daily returns, costs, metrics, tear sheet.

Grid searches (many runs over a parameter grid) start here; every run is logged in `experiments/`. Stress tests come from `risk/`.

Contains: `fx_event_engine.py` (intraday, trade-by-trade engine from the FX research).
