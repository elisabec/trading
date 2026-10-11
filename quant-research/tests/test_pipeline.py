import strategies
from src import pipeline


def test_end_to_end_on_synthetic_data(synthetic_prices, tmp_path):
    strat = strategies.get("trend_etf")
    cols = strat.universe[: synthetic_prices.shape[1]]
    prices = synthetic_prices.set_axis(cols, axis=1)
    strat.universe = cols

    result = pipeline.run(strat, prices)
    assert result.sim.returns.notna().all()
    assert (result.sim.weights.sum(axis=1) <= 1.0 + 1e-9).all()
    for key in ("cagr", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"):
        assert key in result.stats

    out = pipeline.write_report(result, prices, tmp_path)
    assert (out / "tearsheet.png").exists()
    assert (out / "summary.md").exists()
    assert (out / "latest_weights.csv").exists()
