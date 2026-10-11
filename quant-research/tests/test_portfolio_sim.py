import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio_sim import simulate


def test_fully_invested_single_asset_matches_asset_return(synthetic_prices):
    p = synthetic_prices[["AAA"]]
    targets = pd.DataFrame({"AAA": [1.0]}, index=[p.index[0]])
    res = simulate(targets, p, cost_bps=0.0)
    asset = p["AAA"].pct_change().iloc[1:]
    np.testing.assert_allclose(res.returns.iloc[1:].values, asset.values, rtol=1e-12)


def test_costs_charged_on_turnover(synthetic_prices):
    p = synthetic_prices[["AAA", "BBB"]]
    d0 = p.index[0]
    targets = pd.DataFrame({"AAA": [0.5], "BBB": [0.5]}, index=[d0])
    res = simulate(targets, p, cost_bps=10.0)
    assert res.turnover.loc[d0] == pytest.approx(1.0)
    assert res.costs.loc[d0] == pytest.approx(1.0 * 10 / 1e4)


def test_weights_drift_between_rebalances(synthetic_prices):
    p = synthetic_prices[["AAA", "BBB"]]
    targets = pd.DataFrame({"AAA": [0.5], "BBB": [0.5]}, index=[p.index[0]])
    res = simulate(targets, p, cost_bps=0.0)
    last = res.weights.iloc[-1]
    growth = p.iloc[-1] / p.iloc[0]
    expected = 0.5 * growth / (0.5 * growth).sum()
    np.testing.assert_allclose(last.values, expected.values, rtol=1e-9)


def test_trade_on_close_earns_from_next_day(synthetic_prices):
    """A position decided on day t must not earn day t's return."""
    p = synthetic_prices[["AAA"]]
    t = p.index[10]
    targets = pd.DataFrame({"AAA": [0.0, 1.0]}, index=[p.index[0], t])
    res = simulate(targets, p, cost_bps=0.0)
    assert res.gross_returns.loc[t] == 0.0
    nxt = p.index[11]
    assert res.gross_returns.loc[nxt] == pytest.approx(p["AAA"].pct_change().loc[nxt])


def test_cash_earns_nothing(synthetic_prices):
    p = synthetic_prices[["AAA"]]
    targets = pd.DataFrame({"AAA": [0.0]}, index=[p.index[0]])
    res = simulate(targets, p, cost_bps=0.0)
    assert (res.returns == 0).all()
