"""
src/features/technical.py

Technical feature engineering utilities.

All functions operate on pandas Series/DataFrames and return
pandas objects with the same index — safe to assign back into
a prices DataFrame.
"""

import numpy as np
import pandas as pd


def get_ema(series: pd.Series, past_period: int = 10) -> pd.Series:
    """Exponential moving average (EMA) of a price series.

    NaN values are dropped before computing EWM to avoid propagation,
    so the returned series may have a shorter index — re-index if needed.

    Args:
        series: Raw price series.
        past_period: EWM span parameter (approx. half-life = span/2).

    Returns:
        EMA series (same index as series after dropping NaNs).
    """
    series_dropped = series.dropna()
    return series_dropped.ewm(span=past_period, adjust=False).mean()


def get_return(prices: pd.Series, spread: pd.Series, shiftmin: int = 1) -> pd.Series:
    """Spread-normalised price return over a given lookback.

    Return is scaled by 1000 so that 1 unit ≈ 0.1% move relative to spread.

    Args:
        prices: EMA mid-price series.
        spread: Bid-ask spread series (same index).
        shiftmin: Lookback in observations (default 1 = 1-period return).

    Returns:
        Spread-normalised return series.
    """
    shifted_prices = prices.shift(shiftmin)
    return 1000 * (prices - shifted_prices) / spread


def get_side(
    prices: pd.Series,
    spread: pd.Series,
    return_period_direction: int = 1,
    momentum_signal: bool = True,
) -> np.ndarray:
    """Determine trade direction (+1 buy / -1 sell) from recent price move.

    Args:
        prices: EMA price series.
        spread: Bid-ask spread series.
        return_period_direction: Lookback window (in observations) to compute
                                  the directional return.
        momentum_signal: If True, follow momentum (buy after up-move).
                         If False, fade (buy after down-move = mean reversion).

    Returns:
        Numpy array of +1 / -1 values.
    """
    r = 1000 * (prices - prices.shift(return_period_direction)) / spread
    if momentum_signal:
        return np.where(r > 0, 1, -1)
    else:
        return np.where(r <= 0, 1, -1)


def compute_markouts(
    prices: pd.DataFrame,
    instruments: list[str],
    offsets: list[int] | None = None,
) -> pd.DataFrame:
    """Compute forward and backward markout returns for each instrument.

    Markout at offset +N = (price[t+N] / price[t] - 1) * 1e4  (in bps)
    Markout at offset -N = (price[t-N] / price[t] - 1) * 1e4  (in bps)

    Args:
        prices: DataFrame containing raw price columns named by instrument.
        instruments: List of instrument names, e.g. ['Brazil', 'Columbia', 'Chile'].
        offsets: Absolute offset values to compute, e.g. [10, 20, 30].
                 Both +N and -N will be computed. Defaults to [10, 20, 30].

    Returns:
        The input DataFrame with additional markout columns added in-place.
    """
    if offsets is None:
        offsets = [10, 20, 30]

    for col in instruments:
        for n in offsets:
            shifted_fwd = prices[col].shift(-n)   # future price
            shifted_bwd = prices[col].shift(n)    # past price
            prices[f"{col}_+{n}Markout"] = (shifted_fwd / prices[col] - 1) * 1e4
            prices[f"{col}_-{n}Markout"] = (shifted_bwd / prices[col] - 1) * 1e4

    return prices
