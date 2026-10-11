"""
Strategy registry. Each strategy lives in its own folder and implements
src.strategy.Strategy.forecast(). Register it here to run it from the CLI.
"""

from strategies.trend_etf import TrendETF

REGISTRY = {
    "trend_etf": TrendETF,
}


def get(name: str):
    try:
        return REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(REGISTRY)}") from exc
