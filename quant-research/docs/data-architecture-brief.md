# Data Architecture Brief: quant-research

For Claude Code. Read together with `quant-research/CLAUDE.md`. This brief records the decisions made so far and the next steps for the data layer.

## Goal

A data platform for one person that lasts: free to run, no lock-in, fast enough on a laptop, and able to grow toward intraday data and live trading later. Every pipeline step reads data through one function and never knows which source it came from.

## Current state (Oct 2026)

- `main` has the folder structure (pipeline steps 1–9 mapped to folders; see `quant-research/README.md`).
- **PR #14 `feat/data-store`** (open, targets `main`) is the first version of the data layer: Cboe PUT + VIX and FRED DTB3 loaders, raw/clean store in Parquet, SQL via DuckDB, `python -m data.update`, and `data.load()`. Tested on sample files only; the first live download still has to run.
- Project 1 is the **volatility premium** strategy (always use that name). Options data, backtests and paper trading run in QuantConnect. The local data store serves research: Cboe indices, FRED rates, later ETFs and macro.

## Decisions

1. **Storage = Parquet files + DuckDB.** Free, no server, pandas and Arrow native. Parquet is an open format that DuckDB, kdb (KDB-X), QuestDB, Spark, C++ and Java can all read, so the data outlives any tool choice.
2. **Location = local SSD, `~/qr-data`** (env `QR_DATA_DIR`). Never iCloud or Dropbox: sync tools evict files to cloud-only placeholders and can corrupt database files. Back up with Time Machine or a periodic copy to an external drive or cloud bucket.
3. **Layers:** `raw/` (exactly as downloaded, never edited, the audit trail) → `clean/` (one standard long table per dataset). Larger datasets later use Hive-style partitions, e.g. `clean/options_eod/underlying=SPX/year=2026/…parquet`.
4. **One way in:** `data.load(dataset, symbols, field, start, as_of=None)`. It returns a pandas panel (dates as rows, symbols as columns). `as_of` (to add) guarantees nothing after that timestamp is returned, so the same strategy code works for backtests (`as_of` = simulated time) and later live (`as_of` = now).
5. **Not now:** a real-time tier (SQLite or similar), KDB-X, paid data. Add them only when a strategy needs intraday decisions or tick data. For the volatility premium strategy, QuantConnect already handles live data and state, and the broker is the source of truth for positions.

## The five layers

| Layer | Job | Status |
|---|---|---|
| 1. Sources | One module per source with `download()` and `parse()`. Free first: Cboe, FRED, Yahoo, SEC EDGAR, Ken French. Paid later: options EOD, intraday | Cboe, FRED in PR #14 |
| 2. Jobs | `python -m data.update` refreshes everything; download only what's new (small full-history files are simply replaced) | Manual command in PR #14 |
| 3. Handling | Raw → clean: parse, validate (no duplicate dates, no gaps, no impossible values), standard schema, symbol list in `data/symbols.csv` | Parse only; validation and symbol list to add |
| 4. Storage | Parquet + DuckDB as above; `store.query(sql)` for ad-hoc SQL | In PR #14 |
| 5. Access | `data.load()` returns panels; add `as_of` | In PR #14, `as_of` to add |

## Scheduling and monitoring of jobs

- **Phase 1 (now):** run `python -m data.update` by hand, or with `cron`/`launchd` on the Mac. Every run checks data freshness (latest date per dataset) and prints it.
- **Phase 2 (once the official paper record starts):** GitHub Actions scheduled workflow (free for public repos, run history and failure emails in the browser), with the Parquet store in a cloud bucket instead of the laptop. Alternative: a small cloud VM with cron.
- Orchestrators (Airflow, Dagster, Prefect) are overkill until there are many dependent jobs.

## Backtest, stress test, live

- **Backtest on real history** already includes real crises; always report 2018, 2020 and 2022.
- **Stress tests and simulation** live in `risk/`: replay historical crisis windows, apply hypothetical shocks (e.g. SPX −20% with VIX at 60), later resampled or Monte Carlo paths. Folder exists; no code yet.
- **Same code for backtest and live:** strategies only ask for data "as of t". QuantConnect's engine works this way for the volatility premium project; our own engine should follow the same rule via `as_of`.

## Next steps (small PRs, one at a time)

1. Review and merge PR #14; run `python -m data.update` locally and check the printed table.
2. `docs:` README and template examples still mention the ETF trend idea; switch them to the volatility premium strategy.
3. `research/volprem/01_first_look`: plot PUT vs VIX history, mark 2008, 2018, 2020, 2022.
4. `feat:` add `as_of` to `data.load()` and a simple validation step in `data/`.
5. `feat:` scheduled update job (cron on the Mac first).

## Working style

Small steps (~25 minutes, one visible result), code simple enough to follow every line, explain choices, work on branches, never push to `main`. Claude only commits on the branch (no push, no PR); I review and push myself.
