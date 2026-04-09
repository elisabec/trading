"""
src/backtest/engine.py

Generic backtest runner + statistics. Signal-agnostic.

Flow per bar
------------
1. Engine injects lagged EMA into the row (so strategy stays stateless)
2. strategy.next_action(index, row, status) → actions dict
3. For each instrument action:
     - HOLD: update trailing PnL stats in status, no trade recorded
     - RI/RR: pass to Portfolio.record_trade() for sizing + PnL attribution
     - Update status (cooldown, risk_reduced, ri_price etc.)
4. At end: compute stats and cum_pnl from trades DataFrame

Usage
-----
    from src.backtest.engine import run_backtest
    from research.fx.dcc_signal import DCCStrategy

    strategy = DCCStrategy(instruments=INSTRUMENTS)
    strategy.initialize(config)

    trades, stats, cum_pnl = run_backtest(
        df            = prices,          # pre-processed, reset_index, trading hours only
        strategy      = strategy,
        instruments   = INSTRUMENTS,
        start_datetime= START_DATE,
        end_datetime  = END_DATE,
        base_notional = 1_000_000,       # scaled by abs(signal_strength)
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signal.base import FXStrategy, PortfolioStatus
from src.backtest.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    strategy: FXStrategy,
    instruments: list[str],
    start_datetime,
    end_datetime,
    base_notional: float | None = None,
    cooldown_bars: int = 5,
    markout_offsets: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Run a full backtest. Returns (trades, stats, cum_pnl).

    Args:
        df:              Pre-processed prices DataFrame. Must be integer-indexed
                         (reset_index(drop=True)) and filtered to trading hours.
        strategy:        Initialized FXStrategy subclass.
        instruments:     Instrument names, e.g. ['Brazil', 'Columbia', 'Chile'].
        start_datetime:  Backtest start (for stats date range).
        end_datetime:    Backtest end.
        base_notional:   Max USD per trade, scaled by abs(signal_strength).
                         If None, reads from strategy.base_notional (set via config).
                         e.g. 1M base × 1.37 normalized signal → capped or scaled.
        cooldown_bars:   Minimum bars between entries per instrument.
        markout_offsets: Forward/backward markout horizons in bars.

    Returns:
        trades:   DataFrame, one row per RI/RR trade.
        stats:    pd.Series — Sharpe, PnL, avg signal strength, markouts, etc.
        cum_pnl:  pd.Series — daily cumulative PnL for plotting.
    """
    markout_offsets = markout_offsets or [10, 20, 30]

    # base_notional lives in strategy config — engine reads it via getattr
    # so the notebook only needs one config dict, not split kwargs.
    resolved_base_notional = (
        base_notional
        if base_notional is not None
        else getattr(strategy, "base_notional", 1_000_000)
    )

    status    = PortfolioStatus.initialize(instruments, cooldown=cooldown_bars)
    portfolio = Portfolio(instruments, resolved_base_notional, markout_offsets)

    ema_lookback            = getattr(strategy, "ema_lookback", 5)
    return_period_direction = getattr(strategy, "return_period_direction", 5)

    for index, row in df.iterrows():
        row = row.copy()

        # Force-close at last bar and exit loop
        if index == len(df) - 1:
            _force_close(index, row, instruments, status, portfolio, df, base_notional)
            break

        # Inject lagged EMA for direction calculation (keeps strategy stateless)
        _inject_lagged_ema(df, index, row, instruments, ema_lookback, return_period_direction)

        actions = strategy.next_action(index, row, status)

        for instr, action in actions.items():
            if action is None:
                continue

            trade_type      = action["type"]
            signal_strength = action.get("signal_strength", 0.0)

            # ── HOLD: update trailing stats, no trade ─────────────────
            if trade_type == "HOLD":
                status.trailing_pnl[instr] = action.get("_trailing_pnl", status.trailing_pnl[instr])
                status.max_pnl[instr]      = action.get("_max_pnl",      status.max_pnl[instr])
                status.last_signal_strength[instr] = signal_strength
                continue

            # ── RI or RR: record trade via portfolio ───────────────────
            side  = action["side"]
            price = action["price"]

            quantity = portfolio.record_trade(
                instr=instr, index=index, row=row,
                side=side, signal_strength=signal_strength,
                trade_type=trade_type, price=price,
                status=status, df=df,
            )

            # ── Update portfolio status ────────────────────────────────
            status.last_signal_strength[instr] = signal_strength

            if trade_type == "RI":
                status.risk_reduced[instr]             = False
                status.ri_price[instr]                 = price
                status.ri_index[instr]                 = index + cooldown_bars
                status.trailing_pnl[instr]             = 0.0
                status.max_pnl[instr]                  = 0.0

            elif trade_type == "RR":
                status.risk_reduced[instr]             = True
                status.ri_index[instr]                 = index + cooldown_bars
                status.trailing_pnl[instr]             = 0.0
                status.max_pnl[instr]                  = 0.0
                status.holding_period_tracker[instr]   = np.nan
                status.cum_usd_position[instr]         = 0.0
                status.cum_instr_position_ccy[instr]   = 0.0

    trades  = portfolio.to_dataframe()
    stats   = compute_stats(trades, instruments, start_datetime, end_datetime)
    cum_pnl = _compute_cum_pnl(trades, start_datetime, end_datetime)

    return trades, stats, cum_pnl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_lagged_ema(
    df: pd.DataFrame,
    index: int,
    row: pd.Series,
    instruments: list[str],
    ema_lookback: int,
    return_period_direction: int,
) -> None:
    """Mutate row in-place: add '_ema_prev_<instr>' for each instrument."""
    lag = index - return_period_direction
    for instr in instruments:
        col = f"{instr}{ema_lookback}EMA"
        row[f"_ema_prev_{instr}"] = df[col].loc[lag] if lag >= 0 and col in df.columns else np.nan


def _force_close(
    index: int,
    row: pd.Series,
    instruments: list[str],
    status: PortfolioStatus,
    portfolio: Portfolio,
    df: pd.DataFrame,
    base_notional: float,
) -> None:
    """Close any open positions at end of backtest at best bid/ask."""
    for instr in instruments:
        if status.cum_usd_position[instr] != 0:
            cum_pos   = status.cum_usd_position[instr]
            side      = -1 if cum_pos > 0 else 1
            price_col = f"{instr}BestBid" if cum_pos > 0 else f"{instr}BestAsk"
            price     = row.get(price_col, np.nan)
            if not np.isnan(price):
                # Use last known signal strength for quantity (or 1.0 if unavailable)
                sig = status.last_signal_strength.get(instr, 1.0) or 1.0
                portfolio.record_trade(
                    instr=instr, index=index, row=row,
                    side=side, signal_strength=sig,
                    trade_type="RR", price=price,
                    status=status, df=df,
                )


def get_markout_value(df: pd.DataFrame, column: str, index: int, offset: int) -> float:
    """Safe point-in-time lookup with integer offset. Returns np.nan on miss."""
    try:
        return df[column].loc[index + offset]
    except KeyError:
        return np.nan


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(
    trades: pd.DataFrame,
    instruments: list[str],
    start_datetime,
    end_datetime,
) -> pd.Series:
    """Summary statistics from a completed trades DataFrame."""
    if len(trades) == 0:
        return pd.Series(dtype="int")

    trades = trades.sort_values(by="Time").reset_index(drop=True)
    trades["Date_UTC"] = pd.to_datetime(trades["Time"]).dt.date

    all_dates    = pd.date_range(start_datetime, end_datetime).date
    all_dates_df = pd.DataFrame(all_dates, columns=["Date_UTC"])

    cum_pnl = (
        trades.groupby("Date_UTC")["Total PnL"]
        .last()
        .reindex(all_dates_df["Date_UTC"], method="ffill")
        .fillna(0)
    )
    daily_pnl = cum_pnl.diff().fillna(0)

    def _safe(col: str) -> float:
        return float(np.nanmean(trades[col])) if col in trades.columns else np.nan

    return pd.Series({
        "Total Pnl": trades["Total PnL"].iloc[-1],
        **{
            f"Total Pnl {i}": trades[f"Total PnL {i}"].iloc[-1]
            if f"Total PnL {i}" in trades.columns else np.nan
            for i in instruments
        },
        "Daily PnL Avg":        daily_pnl.mean(),
        "Daily PnL Std":        daily_pnl.std(),
        "Sharpe Ratio p.a.":    daily_pnl.mean() / daily_pnl.std() * np.sqrt(252),
        "Total number trades":  len(trades),
        "Avg holding period":   trades["Holding Period"].dropna().mean(),
        "Max Daily Loss":       daily_pnl.min(),
        "Avg Signal Strength":  _safe("Signal_Strength"),  # new — conviction tracking
        "Markout_m30": _safe("Markout_m30"),
        "Markout_m20": _safe("Markout_m20"),
        "Markout_m10": _safe("Markout_m10"),
        "Markout_p10": _safe("Markout_p10"),
        "Markout_p20": _safe("Markout_p20"),
        "Markout_p30": _safe("Markout_p30"),
    }).round(4)


def _compute_cum_pnl(trades: pd.DataFrame, start_datetime, end_datetime) -> pd.Series:
    """Daily cumulative PnL series for plotting."""
    if len(trades) == 0:
        return pd.Series(dtype=float)
    trades = trades.copy()
    trades["Date_UTC"] = pd.to_datetime(trades["Time"]).dt.date
    all_dates = pd.date_range(start_datetime, end_datetime).date
    return (
        trades.groupby("Date_UTC")["Total PnL"]
        .last()
        .reindex(pd.DataFrame(all_dates, columns=["Date_UTC"])["Date_UTC"], method="ffill")
        .fillna(0)
    )
