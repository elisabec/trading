# quant-research

> **Disclaimer:** All work in this repository is my own. It was developed independently on personal time and does not reflect the views, proprietary methods, or intellectual property of any employer, past or present.

A personal quantitative research platform for developing, backtesting, and iterating on systematic trading strategies.

The philosophy follows a disciplined pipeline:
**Data → Features → Signal → Backtest → Risk Control → Iterate**

---

## Repository Structure
```
quant-research/
├── src/                        # Reusable engine — signal-agnostic
│   ├── strategy.py             # The contract: a strategy = forecast() + PortfolioConfig
│   ├── pipeline.py             # forecast → construct → simulate → report
│   ├── data/                   # Loaders (Yahoo) + local Parquet cache
│   ├── portfolio/              # Portfolio construction: vol targeting, caps, long-only, rebalancing
│   ├── backtest/
│   │   ├── portfolio_sim.py    # Daily weight-based simulator (drift, turnover, costs)
│   │   └── engine.py           # Event-driven intraday engine (FX research)
│   ├── report/                 # Metrics + one-page tear sheet
│   └── features/               # GARCH, DCC, EMA, rolling vol
│
├── strategies/                 # Alpha models — one folder per strategy
│   └── trend_etf/              # Cross-asset ETF trend, vol-targeted, long-only
├── research/fx/                # FX DCC-GARCH mean-reversion research
├── scripts/run_backtest.py     # CLI: download data, run, write report
└── tests/                      # Engine correctness + no-lookahead checks
```

### How the layers split

| Layer | Question it answers | Where |
|---|---|---|
| Alpha model | What do I believe? Forecast in [-1, 1] per instrument | `strategies/<name>/` |
| Portfolio construction | How much? Sizing, risk budget, caps, long-only, rebalance schedule | `src/portfolio/` |
| Simulation | What would have happened? Drift, turnover, costs, no lookahead | `src/backtest/portfolio_sim.py` |
| Reporting | How good is it? Sharpe, drawdown, turnover, tear sheet | `src/report/` |
| Execution *(planned)* | How do I trade it? Target weights → orders at the broker | `src/execution/` |

A new strategy only implements `forecast()` and chooses a `PortfolioConfig`; everything else is shared.

### Quick start

```bash
cd quant-research
pip install -r requirements.txt
python scripts/run_backtest.py --strategy trend_etf --refresh   # downloads prices, runs, writes results/trend_etf/
pytest -q                                                       # engine tests on synthetic data
```

---

## Research

### FX — DCC-GARCH Mean Reversion Signal
`research/fx/`

A systematic intraday mean-reversion signal across correlated FX pairs using dynamic conditional correlations. The signal detects high-co-movement regimes and trades temporary dislocations that tend to mean-revert within minutes.

See [`research/fx/README.md`](research/fx/README.md) for full methodology and results.

---

### Systematic — Cross-Asset ETF Trend
`strategies/trend_etf/`

Time-series momentum on 12 liquid ETFs (equities, Treasuries, credit, inflation-linked, gold, commodities, USD). Forecast = average sign of 1/3/6/12-month returns; inverse-vol sizing to a 10% portfolio vol target; long-only, monthly rebalance, 5 bps costs.

---

## Tech Stack

`Python` · `NumPy` · `pandas` · `arch` (GARCH) · `scikit-learn` · `matplotlib`

---

## Setup

```bash
git clone <repo>
cd quant-research
pip install -r requirements.txt
```