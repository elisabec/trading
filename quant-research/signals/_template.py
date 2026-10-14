"""
signals/_template.py  —  copy this file to start a new idea.

    cp signals/_template.py signals/my_idea.py

Then fill in the three parts below. Nothing else in the repo has to change.
"""

import pandas as pd

from signals.base import Signal


class MyIdea(Signal):
    # 1) Name: used in strategy recipes and the experiment log
    name = "my_idea"

    # 2) Parameters: the knobs you will want to vary
    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    # 3) Logic: data in, score panel (date x asset) out
    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data["prices"]
        score = prices / prices.shift(self.lookback) - 1  # example: past return
        return score
