# quant-research: Project Brief

For Claude Code. Read before working in `quant-research/`.

## Purpose

Build a systematic strategy and run it as a **timestamped forward-test track record**. Process discipline matters as much as performance.

## How I work

- Small steps: each task finishable in ~25 minutes, with one visible result (a commit, a chart, a working cell)
- Ugly v0 first: end-to-end before anything is polished
- Write kill criteria (Sharpe, drawdown) before looking at results
- Keep code simple enough that I follow every line. Explain choices, build piece by piece
- Work on branches and open pull requests; never push to `main`
- Branch names: `setup/…`, `feat/…`, `fix/…`, `docs/…`; PR titles start the same way (`feat: …`)

## Architecture

Every idea becomes one signal file; everything downstream is shared. Folders (each has a README):

`data/` → `features/` → `signals/` → `evaluation/` → `portfolio/` → `backtest/` → `execution/` → `monitoring/`, plus `strategies/` (recipes, yaml), `experiments/` (run log), `research/` (notebooks).

- Panels are pandas DataFrames: dates as rows, assets as columns
- New idea: copy `signals/_template.py`; new strategy: copy `strategies/_template.yaml`
- Evaluate a signal (IC, forward returns) before any backtest
- Data lives outside the repo in `~/qr-data` (env `QR_DATA_DIR`): `raw/` as downloaded, `clean/` standard tables. Research code reads only from the local store, never from the internet

## Project 1: Volatility Premium

**Idea:** Harvest the equity volatility risk premium.
- Sell SPX puts monthly, fully collateralized with T-bills
- Spend a fixed fraction of the premium on out-of-the-money VIX calls as a tail hedge
- Roll once a month

**Rules, not judgment.** Parameters to test:
- Put strike: fixed delta (10 / 15 / 20 / 25)
- Hedge budget: 0% / 10% / 20% of premium
- Sizing: notional = collateral (no leverage)

**Tools:** QuantConnect for options data, backtests and paper deployment (connected to my IBKR paper account). Keep the code in this repo, in the folders above (index and rates data via `data/`, the strategy code under `strategies/` and `signals/`). The FX code (`backtest/fx_event_engine.py`, `research/fx/`) is not part of this project.

**Why paper only:** one SPX put is several hundred thousand dollars of notional; fully collateralized, it cannot be sized into a small account.

**Confidentiality:** Public sources and my own design only. Don't name any fund or firm in code, docs or commits; call it the "volatility premium" strategy.

## Data

| Need | Source |
|---|---|
| Benchmark | Cboe PutWrite Index (PUT) + VIX index, free daily history |
| SPX + VIX options | QuantConnect (from ~2012; verify VIX options coverage) |
| T-bill rates | FRED |

Use realistic bid-ask spreads; paper fills for options are optimistic.

## Research rules

- Small grid only (deltas x hedge budgets above)
- Choose parameters on an earlier period, validate on a later untouched one
- Pick from a stable region, not the single best result
- Always show 2018, 2020 and 2022, plus max drawdown, Sharpe, turnover

## Track record rules

- **Sandbox** (QuantConnect's built-in paper brokerage): experiments, can be deleted
- **Official record:** frozen rules, git tag with date (e.g. `volprem-v1.0-2026-11-01`), deployed to IBKR paper. Any rule change = new tag + new record
- Monthly log: return, drawdown, anything unusual

## Compliance and IP

- Paper trading only for now. No real money until JPMorgan (until Mar 6, 2027) and BlackRock (from Mar 8, 2027) rules are confirmed; BlackRock's short-term trading rules may keep this strategy paper-only
- Nothing from JPMorgan in this repo
- Repo is **public**: never commit credentials, API keys or licensed data; keep final strategy parameters private. Don't touch the other showcase projects

## Steps

1. Download Cboe PUT + VIX history, plot them (first look at the premium and crashes)
2. Set up QuantConnect + LEAN CLI linked to this repo; deploy a dummy (buy-and-hold SPY) to QuantConnect paper
3. Backtest one simple version (20-delta puts, no hedge) end to end
4. Add the hedge, then run the small grid
5. Pick, freeze, tag, deploy to IBKR paper: official record starts
