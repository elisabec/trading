"""Every registered strategy must not peek at the future.

We perturb all prices after a cut-off date and check that forecasts and
target weights up to that date are unchanged.
"""

import numpy as np
import pandas as pd
import pytest

import strategies
from src.portfolio.construction import construct


@pytest.mark.parametrize("name", sorted(strategies.REGISTRY))
def test_forecast_and_weights_use_only_past_data(name, synthetic_prices):
    strat = strategies.get(name)
    cols = strat.universe[: synthetic_prices.shape[1]]
    prices = synthetic_prices.set_axis(cols, axis=1)
    strat.universe = cols

    cut = prices.index[900]
    shocked = prices.copy()
    shocked.loc[prices.index > cut] *= np.exp(np.random.default_rng(1).normal(0, 0.2, shocked.loc[prices.index > cut].shape))

    fc_a = strat.forecast(prices).loc[:cut]
    fc_b = strat.forecast(shocked).loc[:cut]
    pd.testing.assert_frame_equal(fc_a, fc_b)

    w_a = construct(strat.forecast(prices), prices, strat.portfolio).loc[:cut]
    w_b = construct(strat.forecast(shocked), shocked, strat.portfolio).loc[:cut]
    pd.testing.assert_frame_equal(w_a, w_b)
