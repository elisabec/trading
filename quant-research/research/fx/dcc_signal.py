"""
src/signal/latam_dcc.py

FX mean-reversion signal using DCC-GARCH dynamic correlations.

Signal output
-------------
next_action() returns signal_strength = raw DCC correlation value, which is
already on [-1, 1] and reflects how strongly the three pairs are co-moving.
The Portfolio uses abs(signal_strength) to scale position size.

Entry logic (RI — Risk Inducing)
---------------------------------
  - Signal is triggered when abs(BrazilSignalStrength) > corr_threshold
  - Direction is determined by recent EMA move (momentum or mean-reversion)
  - Cooldown: minimum cooldown_bars between entries

Exit logic (RR — Risk Reducing) — signal-specific, lives here
--------------------------------------------------------------
  - Trailing stop: close when PnL retraces below trailing_stop_retrace
    fraction of its peak (e.g. 0.8 = close if PnL drops to 80% of max)
  - Also closes at end of backtest (forced by engine)

Config keys
-----------
    ema_lookback              int   EMA window in bars (default 5)
    corr_threshold            float Minimum abs(signal) to trigger (default 0.6)
    return_period_direction   int   Bars back to determine direction (default 5)
    momentum_signal           bool  True=momentum, False=mean-reversion (default False)
    cooldown_bars             int   Min bars between entries (default 5)
    trailing_stop_retrace     float Stop fraction of peak PnL (default 0.8)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signal.base import FXStrategy, PortfolioStatus


class DCCStrategy(FXStrategy):
    """DCC-correlation mean-reversion signal for FX.

    Instruments are set at construction and fully configurable — the signal
    logic is instrument-agnostic provided the prices DataFrame has the
    expected column naming convention (see exploration.ipynb).

    Example
    -------
    >>> config = {
    ...     "ema_lookback": 5,
    ...     "corr_threshold": 0.6,
    ...     "return_period_direction": 5,
    ...     "momentum_signal": False,
    ...     "cooldown_bars": 5,
    ...     "trailing_stop_retrace": 0.8,
    ... }
    >>> strategy = DCCStrategy(instruments=["Brazil", "Columbia", "Chile"])
    >>> strategy.initialize(config)
    >>> # Engine calls: actions = strategy.next_action(index, row, status)
    """

    def initialize(self, config: dict) -> None:
        self.ema_lookback               = config.get("ema_lookback", 5)
        self.corr_threshold             = config.get("corr_threshold", 0.6)
        self.return_period_direction    = config.get("return_period_direction", 5)
        self.momentum_signal            = config.get("momentum_signal", False)
        self.cooldown_bars              = config.get("cooldown_bars", 5)
        self.trailing_stop_retrace      = config.get("trailing_stop_retrace", 0.8)
        # base_notional lives here so all strategy params are in one config dict.
        # The engine reads it via getattr(strategy, "base_notional", 1_000_000).
        self.base_notional              = config.get("base_notional", 1_000_000)

    def next_action(
        self,
        index: int,
        row: pd.Series,
        status: PortfolioStatus,
    ) -> dict[str, dict | None]:
        """Core signal logic — called once per bar by the engine.

        Checks RI (entry) and RR (exit/trailing stop) conditions per instrument.

        Args:
            index:  Current bar index.
            row:    Prices row with all precomputed features. Engine injects
                    '_ema_prev_<instr>' for direction calculation.
            status: Live portfolio state (read-only in the signal).

        Returns:
            Dict keyed by instrument. None = no action.
            Action dict keys: signal_strength, side, type.
        """
        actions: dict[str, dict | None] = {instr: None for instr in self.instruments}

        for instr in self.instruments:

            # Column name conventions (matching exploration.ipynb)
            ema_col           = f"{instr}{self.ema_lookback}EMA"
            spread_col        = f"{instr}Spread"
            sig_strength_col  = f"{instr}SignalStrength"
            sig_triggered_col = f"{instr}SignalTriggered"
            best_ask_col      = f"{instr}BestAsk"
            best_bid_col      = f"{instr}BestBid"

            # Normalize DCC score by threshold so that:
            #   abs(signal_strength) == 1.0  ↔  exactly at trigger threshold
            #   abs(signal_strength) >  1.0  ↔  high conviction (above threshold)
            #   abs(signal_strength) <  1.0  ↔  weak signal (below threshold, no trade)
            # Portfolio uses abs(signal_strength) for sizing, so stronger signals
            # deploy proportionally more notional.
            raw_strength = float(row.get(sig_strength_col, 0.0))
            signal_strength = raw_strength / self.corr_threshold if self.corr_threshold != 0 else raw_strength

            # ------------------------------------------------------------------
            # RISK INDUCING (RI): open a new position
            # ------------------------------------------------------------------
            if status.risk_reduced[instr] and index >= status.ri_index[instr]:

                signal_triggered = bool(row.get(sig_triggered_col, False))
                if not signal_triggered:
                    continue  # signal too weak, skip

                # Direction from recent EMA move (engine pre-injects lagged EMA)
                ema_now  = row.get(ema_col, np.nan)
                ema_prev = row.get(f"_ema_prev_{instr}", np.nan)
                spread   = row.get(spread_col, np.nan)

                if (
                    np.isnan(ema_now) or np.isnan(ema_prev)
                    or np.isnan(spread) or spread == 0
                ):
                    continue  # can't determine direction

                direction_return = 1000 * (ema_now - ema_prev) / spread

                if self.momentum_signal:
                    side = 1 if direction_return > 0 else -1
                else:
                    # Mean reversion: fade the move
                    side = 1 if direction_return <= 0 else -1

                price = row.get(best_ask_col if side == 1 else best_bid_col, np.nan)
                if np.isnan(price):
                    continue

                actions[instr] = {
                    "signal_strength": signal_strength,  # raw DCC value
                    "side": side,
                    "type": "RI",
                    "price": price,
                }

            # ------------------------------------------------------------------
            # RISK REDUCING (RR): trailing stop on open position
            # ------------------------------------------------------------------
            elif not status.risk_reduced[instr]:

                cum_pos   = status.cum_usd_position[instr]
                side_mult = -1 if cum_pos > 0 else 1
                rr_col    = best_bid_col if cum_pos > 0 else best_ask_col
                rr_price  = row.get(rr_col, np.nan)

                if np.isnan(rr_price):
                    continue

                # Update trailing PnL (engine reads back from status after action)
                ri_p      = status.ri_price[instr]
                trailing  = (rr_price - ri_p) * side_mult * abs(cum_pos)
                max_pnl   = max(trailing, status.max_pnl[instr])

                # Trailing stop trigger: PnL below retrace fraction of peak
                retrace_breached = (
                    max_pnl <= 0
                    or trailing / max_pnl <= self.trailing_stop_retrace
                )

                if retrace_breached:
                    actions[instr] = {
                        "signal_strength": signal_strength,  # pass through for attribution
                        "side": side_mult,
                        "type": "RR",
                        "price": rr_price,
                        # Pass updated trailing stats so engine can store them
                        "_trailing_pnl": trailing,
                        "_max_pnl": max_pnl,
                    }
                else:
                    # Not closing yet — but update trailing stats in status
                    # (engine does this after calling next_action)
                    actions[instr] = {
                        "signal_strength": signal_strength,
                        "side": None,   # no trade
                        "type": "HOLD",
                        "_trailing_pnl": trailing,
                        "_max_pnl": max_pnl,
                    }

        return actions
