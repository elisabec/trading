# quant-research

> **Disclaimer:** All work in this repository is my own. It was developed independently on personal time and does not reflect the views, proprietary methods, or intellectual property of any employer, past or present.

A personal quantitative research platform for developing, backtesting, and iterating on systematic trading strategies.

The philosophy follows a disciplined pipeline:
**Data → Features → Signal → Evaluate → Portfolio → Backtest → Iterate**

---

## Architecture

Every idea, whether a trend rule, a value factor, an LLM score on filings or a macro regime, becomes **one signal file**. Everything downstream is shared.

```
DATA ──► FEATURES ──► SIGNALS ──► EVALUATION ──► PORTFOLIO ──► BACKTEST ──► EXECUTION ──► MONITORING
```

The common currency between layers is a **panel**: a pandas DataFrame with dates as rows and assets as columns.

```
quant-research/
├── data/          raw → clean panels (one loader per source)
├── features/      reusable ingredients: returns, vol, z-scores, correlations
├── signals/       ONE FILE PER IDEA → score panel        (template: _template.py)
├── evaluation/    does the signal predict? IC, forward returns, decay
├── portfolio/     scores → target weights: sizing, vol target, caps, long-only
├── backtest/      weights → returns, costs, tear sheet
├── strategies/    recipes: universe + signals + settings (template: _template.yaml)
├── experiments/   log of every run: parameters, period, results
├── execution/     weights → orders at Interactive Brokers
├── monitoring/    live P&L, positions vs targets, alerts
└── research/      notebooks and scratch work
```

**Signal vs strategy:** a signal is one idea's logic (`signals/trend.py`). A strategy is a recipe that uses one or more signals on a universe with portfolio settings (`strategies/trend_etf.yaml`).

---

## Research

### FX — DCC-GARCH Mean Reversion Signal
`research/fx/`

A systematic intraday mean-reversion signal across correlated FX pairs using dynamic conditional correlations. The signal detects high-co-movement regimes and trades temporary dislocations that tend to mean-revert within minutes.

See [`research/fx/README.md`](research/fx/README.md) for full methodology and results.

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