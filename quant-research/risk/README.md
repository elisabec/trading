# Risk

A shared toolbox, like `features/`: not a pipeline step itself, but used by three of them.

| Task | What it answers | Used in step |
|---|---|---|
| **Risk model** (factor exposures, covariance; the Barra idea, built from public factors) | What am I really exposed to: equity market, volatility, rates? How much is just beta? | 6 Portfolio: sizing, limits, vol cap |
| **Stress tests** (historical and hypothetical scenarios) | What happens in Feb 2018, March 2020, 2022? What if SPX falls 20% and VIX jumps to 60? | 7 Backtest: before trusting a strategy |
| **Risk and P&L attribution** | Where did this month's P&L come from: premium, market move, hedge? | 9 Monitoring: live |

Factors come from free data (Ken French library, SPX, VIX changes, rates) via `data/`.
