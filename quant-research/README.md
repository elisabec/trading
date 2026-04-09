# quant-research

> **Disclaimer:** All work in this repository is my own. It was developed independently on personal time and does not reflect the views, proprietary methods, or intellectual property of any employer, past or present.

A personal quantitative research platform for developing, backtesting, and iterating on systematic trading strategies.

The philosophy follows a disciplined pipeline:
**Data → Features → Signal → Backtest → Risk Control → Iterate**

---

## Repository Structure
```
quant-research/
├── src/                            # Reusable research library
│   ├── features/
│   │   ├── correlation.py          # GARCH volatility, DCC dynamic correlation
│   │   └── technical.py            # EMA, returns, rolling volatility
│   └── backtest/
│       └── engine.py               # Trade simulation, markout analysis, Sharpe, PnL
│
└── research/                       # Signal-specific experiments
└── fx/                             # FX mean-reversion signal
├── dcc_signal.py                   # Signal implementation
└── README.md                       # Methodology and results
```
---

## Research

### FX — DCC-GARCH Mean Reversion Signal
`research/fx/`

A systematic intraday mean-reversion signal across correlated FX pairs using dynamic conditional correlations. The signal detects high-co-movement regimes and trades temporary dislocations that tend to mean-revert within minutes.

See [`research/fx/README.md`](research/fx/README.md) for full methodology and results.

---

## `src` Library

Signal-agnostic and designed to be reused across strategies.

| Module | Purpose |
|---|---|
| `src/features/correlation.py` | Rolling correlation, GARCH(1,1) volatility, DCC dynamic correlation, PCA regime detection |
| `src/features/technical.py` | EMA, log returns, rolling volatility, lag features |
| `src/backtest/engine.py` | Trade simulation, entry/exit logic, trailing stops, markout stats, Sharpe ratio, daily PnL |

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