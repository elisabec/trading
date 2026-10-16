# quant-research

> **Disclaimer:** All work in this repository is my own. It was developed independently on personal time and does not reflect the views, proprietary methods, or intellectual property of any employer, past or present.

A personal quantitative research platform for developing, backtesting, and iterating on systematic trading strategies.

The philosophy follows a disciplined pipeline:
**Data → Features → Signals → Evaluation → Portfolio & risk → Backtest → Execution → Monitoring**

---

## Architecture

Every idea, whether a trend rule, a value factor, an LLM score on filings or a macro regime, becomes **one signal file**. Everything downstream is shared. Signals never know where the data came from.

### The pipeline, step by step, and where each step lives

| # | Pipeline step | Folder | Question it answers |
|---|---|---|---|
| 1 | Data sources | `data/` (one file per source) | Where does it come from? |
| 2 | Data handling | `data/` (`store.py`, `update.py`) | Stored, cleaned, one way in |
| 3 | Features | `features/` | What do I measure? |
| 4 | Signals | `signals/` | What do I believe? |
| 5 | Signal evaluation | `evaluation/` | Is the belief any good? (IC, forward returns) |
| 6 | Portfolio construction & risk | `portfolio/` (+ `risk/`) | How much of each, within which limits? |
| 7 | Backtest & stress test | `backtest/` (+ `risk/`) | What would have happened, also in crises? |
| 8 | Execution | `execution/` | How do I trade it? |
| 9 | Monitoring & attribution | `monitoring/` (+ `risk/`) | How is it doing live, and why? |

**Beside the pipeline**

| Folder | Role |
|---|---|
| `strategies/` | Recipes (yaml): universe + which signals + settings. Wires steps 1–9 for one strategy |
| `risk/` | Shared risk toolbox: risk model (factor exposures, covariance), stress tests, P&L attribution. Used by steps 6, 7, 9 |
| `experiments/` | Log of every run: parameters, period, results |
| `research/` | Notebooks and scratch work |

The common currency between steps is a **panel**: a pandas DataFrame with dates as rows and assets as columns.

**Signal vs strategy:** a signal is one idea's logic (`signals/trend.py`). A strategy is a recipe that uses one or more signals on a universe with portfolio settings (`strategies/trend_etf.yaml`). Templates: `signals/_template.py`, `strategies/_template.yaml`.

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