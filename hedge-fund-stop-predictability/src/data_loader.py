"""
Data ingestion and fund-universe filtering.

Loads raw CSV/Excel inputs, standardises column names and date formats,
applies eligibility filters (currency, minimum track record, strategy),
and constructs AUM-weighted peer-group benchmark returns.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CURRENCY,
    CURRENT_DATE,
    DATA_DIR,
    FIGURES_DIR,
    MIN_TRACK_RECORD_MONTHS,
    START_DATE,
    STRATEGIES,
)

logger = logging.getLogger(__name__)


# ── Raw ingestion ────────────────────────────────────────────────────────────

def convert_excel_to_csv() -> None:
    """Read Excel files linked to the analytics system and persist as CSV."""
    mapping = {
        "input_gdxp.xlsx":      "gdxp.csv",
        "input_aum.xlsx":       "aum.csv",
        "input_funds.xlsx":     "funds.csv",
        "input_returns.xlsx":   "returns.csv",
        "input_returnspg.xlsx": "returnspg.csv",
    }
    for xlsx, csv in mapping.items():
        df = pd.read_excel(DATA_DIR / xlsx)
        df.to_csv(DATA_DIR / csv, index=False)
    logger.info("Converted Excel inputs → CSV.")


def _rename_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw string date column headers to ``datetime.date`` objects."""
    months = (
        pd.date_range(
            start=df.columns[1],
            end=pd.to_datetime(df.columns[-1]) + pd.offsets.MonthEnd(),
            freq="MS",
        )
        .to_series()
        .apply(lambda x: x.date())
    )
    cols = ["fund"]
    cols[1:] = months
    df.columns = cols
    return df


def load_raw_data(
    start_date=START_DATE,
    current_date=CURRENT_DATE,
) -> dict:
    """
    Load the five canonical CSV files and return a dict of DataFrames
    keyed by ``gdxp``, ``funds``, ``aum``, ``returns``, ``returnspg``.
    """
    # Global/macro data
    gdxp = pd.read_csv(
        DATA_DIR / "gdxp.csv",
        usecols=["date", "returns_stock", "returns_treasury",
                 "returns_commodity", "Serie 118894"],
        parse_dates=["date"],
    )
    gdxp["date"] = gdxp["date"].apply(lambda x: x.date())
    gdxp.set_index("date", inplace=True)
    gdxp.columns = ["rf", "returns_stock", "returns_treasury", "returns_commodity"]
    gdxp["rf"] = gdxp["rf"] / 100

    # Fund metadata
    funds = pd.read_csv(
        DATA_DIR / "funds.csv",
        usecols=["fund", "tr_start", "tr_stop", "months", "pg",
                 "tr_pg_start", "tr_pg_stop", "currency", "strategy", "date_added"],
        parse_dates=["tr_start", "tr_stop", "tr_pg_start", "tr_pg_stop", "date_added"],
    )
    funds.loc[funds.strategy == "CTA/Managed Futures", "strategy"] = "CTA_Managed Futures"

    # Panel data (wide format)
    aum       = _rename_month_columns(pd.read_csv(DATA_DIR / "aum.csv"))
    returns   = _rename_month_columns(pd.read_csv(DATA_DIR / "returns.csv"))
    returnspg = _rename_month_columns(pd.read_csv(DATA_DIR / "returnspg.csv"))

    # Trim to sample window
    month_cols = (
        pd.date_range(
            start=start_date,
            end=current_date + pd.offsets.MonthEnd(),
            freq="MS",
        )
        .to_series()
        .apply(lambda x: x.date())
    )
    keep = ["fund"] + list(month_cols)
    returns   = returns[keep]
    aum       = aum[keep]
    returnspg = returnspg[keep]

    logger.info(
        "Loaded raw data: %d funds, %d return months.",
        len(funds),
        len(keep) - 1,
    )
    return {
        "gdxp": gdxp,
        "funds": funds,
        "aum": aum,
        "returns": returns,
        "returnspg": returnspg,
    }


# ── Fund filtering ───────────────────────────────────────────────────────────

def filter_fund_universe(
    funds: pd.DataFrame,
    returns: pd.DataFrame,
    currency: str = CURRENCY,
    min_months: int = MIN_TRACK_RECORD_MONTHS,
    strategies: list = STRATEGIES,
) -> pd.DataFrame:
    """
    Apply eligibility filters and de-duplicate near-identical share classes.

    Persists the filtered universe to ``DATA_DIR / consideredFunds.csv``.
    """
    eligible = funds[
        funds.strategy.isin(strategies)
        & (funds.currency == currency)
        & (funds.months >= min_months)
    ].reset_index(drop=True)

    # De-duplicate share classes with similar names (first 10 chars)
    drop_idx = []
    for i in range(len(eligible) - 1):
        name_a = eligible.fund.iloc[i][:10]
        name_b = eligible.fund.iloc[i + 1][:10]
        if name_a == name_b:
            len_a = returns[returns.fund == eligible.fund.iloc[i]].squeeze().dropna().shape[0]
            len_b = returns[returns.fund == eligible.fund.iloc[i + 1]].squeeze().dropna().shape[0]
            drop_idx.append(i if len_a <= len_b else i + 1)

    result = eligible.drop(eligible.index[drop_idx]).reset_index(drop=True)
    result.to_csv(DATA_DIR / "consideredFunds.csv", index=False)
    logger.info("Filtered universe: %d funds.", len(result))
    return result


def load_filtered_funds() -> pd.DataFrame:
    """Load the previously persisted filtered fund universe."""
    date_cols = ["tr_start", "date_added", "tr_stop", "tr_pg_start", "tr_pg_stop"]
    funds = pd.read_csv(
        DATA_DIR / "consideredFunds.csv",
        usecols=["fund", "tr_start", "date_added", "tr_stop", "months",
                 "pg", "tr_pg_start", "tr_pg_stop", "currency", "strategy"],
        parse_dates=date_cols,
    )
    funds["months"] = funds["months"].astype(int)
    for c in date_cols:
        funds[c] = funds[c].apply(lambda x: x.date())
    return funds


# ── Peer-group benchmarks ────────────────────────────────────────────────────

def compute_peer_group_indices(
    funds: pd.DataFrame,
    returns: pd.DataFrame,
    aum: pd.DataFrame,
    strategies: list = STRATEGIES,
    save_plots: bool = True,
) -> pd.DataFrame:
    """
    Compute AUM-weighted strategy benchmark returns and (optionally) save
    NAV/return charts per strategy.
    """
    returnspg_self = pd.DataFrame(columns=returns.columns)

    for strategy in strategies:
        selection = funds[funds.strategy == strategy].fund
        r = returns[returns.fund.isin(selection)].reset_index(drop=True).sort_values("fund")
        a = aum[aum.fund.isin(selection)].reset_index(drop=True).sort_values("fund")

        if not (r.fund.values == a.fund.values).all():
            raise ValueError(f"Return/AUM fund mismatch for strategy '{strategy}'.")

        weighted_returns = r.iloc[:, 1:] * a.iloc[:, 1:]
        total_aum = a.iloc[:, 1:].sum()
        benchmark = weighted_returns.sum() / total_aum

        if save_plots:
            nav = np.cumprod(1 + benchmark)
            for series, label in [(benchmark, "Return"), (nav, "NAV")]:
                series.plot(title=f"{label} — {strategy}", color="black")
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / f"strategy_{label}_{strategy}.png", dpi=150)
                plt.close()

        row = [strategy] + list(benchmark)
        returnspg_self.loc[len(returnspg_self)] = row

    returnspg_self.to_csv(DATA_DIR / "returnspg_self.csv", index=False)
    logger.info("Computed peer-group indices for %d strategies.", len(strategies))
    return returnspg_self


def load_peer_group_returns() -> pd.DataFrame:
    """Load pre-computed strategy benchmark returns."""
    returnspg = pd.read_csv(DATA_DIR / "returnspg_self.csv")
    return _rename_month_columns(returnspg)
