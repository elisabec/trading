"""
signals/base.py

The parent class every signal inherits from (like an abstract class in Java).

A signal turns data into a score panel:
    rows    = dates
    columns = assets
    values  = score (higher = more bullish). NaN = no view.

Rule: the score on date t may only use data known at the close of t.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Signal(ABC):
    name: str = "unnamed"

    @abstractmethod
    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a score panel (date x asset).

        data: named panels, e.g. {"prices": ..., "volume": ...}.
              Each panel is a pandas DataFrame, date x asset.
        """
