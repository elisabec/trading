"""
Feature engineering for hedge-fund predictability analysis.

Constructs a panel dataset where each row is a *(fund, month)* observation
enriched with backward/forward-looking performance metrics, drawdown
indicators, volatility ratios, empirical CDFs, and the three target
variables (died, performance, outperformance).
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import shift

from config import (
    CURRENT_DATE,
    FORECAST_MONTHS,
    LOOKBACK_MONTHS,
    SIMILAR_RETURN_TOL,
    START_DATE,
    VOLA_WINDOW_LONG,
    VOLA_WINDOW_SHORT,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Date helpers
# ═══════════════════════════════════════════════════════════════════════════════

def month_diff(a, b):
    return 12 * (b.year - a.year) + (b.month - a.month)


def add_months(source_date, n_months):
    month = source_date.month - 1 + n_months
    year  = source_date.year + month // 12
    month = month % 12 + 1
    return datetime(year, month, source_date.day).date()


def subtract_months(source_date, n_months):
    month = source_date.month - 1 - n_months
    year  = source_date.year + month // 12
    month = month % 12 + 1
    return datetime(year, month, source_date.day).date()


def next_month_start(source_date):
    month = source_date.month
    year  = source_date.year + month // 12
    month = month % 12 + 1
    return datetime(year, month, 1).date()


# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level metric functions
# ═══════════════════════════════════════════════════════════════════════════════

def empirical_cdf(r: np.ndarray, p: float) -> float:
    """Proportion of sorted non-NaN values in *r* that are ≤ *p*."""
    s = np.sort(r[~np.isnan(r)])
    if np.isnan(p) or len(s) <= 1:
        return np.nan
    return (s <= p).sum() / len(s)


def drawdown(r: np.ndarray):
    """Return (current drawdown, max drawdown) from a return series."""
    if len(r) <= 1 or np.isnan(r).any():
        return np.nan, np.nan
    nav = np.cumprod(1 + r)
    dd_series = 1 - nav / np.maximum.accumulate(nav)
    return dd_series[-1], np.max(dd_series)


def max_dd_phase(r) -> list:
    """Binary indicator: 1 if currently in the worst drawdown phase."""
    if len(r) <= 1:
        return [np.nan] * len(r)
    nav = np.cumprod(1 + r)
    trough = nav[0]
    phase = [np.nan]
    for i in range(len(nav) - 1):
        v = nav[i + 1]
        if np.isnan(v):
            phase.append(np.nan)
        elif v > trough:
            phase.append(0)
            trough = min(nav[: i + 1])
        else:
            phase.append(1)
    return phase


def annualised_vol(r: np.ndarray) -> float:
    n = len(r)
    if n <= 1 or np.isnan(r).any():
        return np.nan
    return np.std(r) * np.sqrt(n * 12 / (n - 1))


def _too_similar(r) -> bool:
    """True if consecutive returns are suspiciously identical (≥ 2 times)."""
    r = r[~np.isnan(r)]
    if len(r) <= 1:
        return False
    diffs = np.abs(np.diff(r.values if hasattr(r, "values") else r))
    return (diffs < SIMILAR_RETURN_TOL).sum() >= 2


# ═══════════════════════════════════════════════════════════════════════════════
#  Rolling-window feature builders
# ═══════════════════════════════════════════════════════════════════════════════

def geometric_mean_return(r: np.ndarray, b: int, f: int):
    """Backward (b-month) and forward (f-month) geometric mean returns."""
    n = len(r)

    # Backward
    if n >= b:
        pad = np.full(b - 1, np.nan)
        back = np.array([stats.gmean(1 + r[i - b : i]) - 1 for i in range(b, n + 1)])
        back = np.concatenate([pad, back])
    else:
        back = np.full(n, np.nan)

    # Forward
    if n >= f:
        fwd = np.array([stats.gmean(1 + r[i : i + f]) - 1 for i in range(1, n - f + 1)])
        tail = np.array([stats.gmean(1 + r[i : n]) - 1 for i in range(n - f + 1, n - 2)])
        fwd = np.concatenate([fwd, tail, np.full(3, np.nan)])
    else:
        fwd = np.full(n, np.nan)

    return back, fwd


def sharpe_ratio_ts(r, b: int, f: int, dates, rf_series):
    """Backward / forward Sharpe-ratio time series."""
    gret = geometric_mean_return(r, b, f)
    ann_back, ann_fwd = np.power(gret[0] + 1, 12) - 1, np.power(gret[1] + 1, 12) - 1
    rf = rf_series.loc[dates].values
    n = len(r)

    def _rolling_vol(series, window, n_total):
        if n_total >= window and not _too_similar(series):
            pad = np.full(window - 1, np.nan)
            vols = np.array([annualised_vol(series[i - window : i]) for i in range(window, n_total + 1)])
            return np.concatenate([pad, vols])
        return np.full(n_total, np.nan)

    def _forward_vol(series, window, n_total):
        if n_total >= window and not _too_similar(series):
            fwd = np.array([annualised_vol(series[i : i + window]) for i in range(1, n_total - window + 1)])
            tail = np.array([annualised_vol(series[i : n_total]) for i in range(n_total - window + 1, n_total - 2)])
            return np.concatenate([fwd, tail, np.full(3, np.nan)])
        return np.full(n_total, np.nan)

    vol_back = _rolling_vol(r, b, n)
    vol_fwd  = _forward_vol(r, f, n)

    sr_back = (ann_back - shift(rf, b)) / vol_back
    sr_fwd  = (ann_fwd - rf) / vol_fwd
    return sr_back, sr_fwd


def excess_return_ts(r, b: int, f: int, dates, rf_series):
    gret = geometric_mean_return(r, b, f)
    ann_back, ann_fwd = np.power(gret[0] + 1, 12) - 1, np.power(gret[1] + 1, 12) - 1
    rf = rf_series.loc[dates].values
    return ann_back - shift(rf, b), ann_fwd - rf


def sharpe_product_ts(r, b: int, f: int, dates, rf_series):
    """SR numerator × volatility (used in outperformance edge-case logic)."""
    gret = geometric_mean_return(r, b, f)
    ann_back, ann_fwd = np.power(gret[0] + 1, 12) - 1, np.power(gret[1] + 1, 12) - 1
    rf = rf_series.loc[dates].values
    n = len(r)

    def _rolling_vol(series, window, n_total):
        if n_total >= window and not _too_similar(series):
            pad = np.full(window - 1, np.nan)
            vols = np.array([annualised_vol(series[i - window : i]) for i in range(window, n_total + 1)])
            return np.concatenate([pad, vols])
        return np.full(n_total, np.nan)

    def _forward_vol(series, window, n_total):
        if n_total >= window and not _too_similar(series):
            fwd = np.array([annualised_vol(series[i : i + window]) for i in range(1, n_total - window + 1)])
            tail = np.array([annualised_vol(series[i : n_total]) for i in range(n_total - window + 1, n_total - 2)])
            return np.concatenate([fwd, tail, np.full(3, np.nan)])
        return np.full(n_total, np.nan)

    vol_back = _rolling_vol(r, b, n)
    vol_fwd  = _forward_vol(r, f, n)
    return (ann_back - shift(rf, b)) * vol_back, (ann_fwd - rf) * vol_fwd


def vol_time_series(r, b: int, f: int):
    n = len(r)

    def _rolling_vol(series, window, n_total):
        if n_total >= window and not _too_similar(series):
            pad = np.full(window - 1, np.nan)
            vols = np.array([annualised_vol(series[i - window : i]) for i in range(window, n_total + 1)])
            return np.concatenate([pad, vols])
        return np.full(n_total, np.nan)

    def _forward_vol(series, window, n_total):
        if n_total >= window and not _too_similar(series):
            fwd = np.array([annualised_vol(series[i : i + window]) for i in range(1, n_total - window + 1)])
            tail = np.array([annualised_vol(series[i : n_total]) for i in range(n_total - window + 1, n_total - 2)])
            return np.concatenate([fwd, tail, np.full(3, np.nan)])
        return np.full(n_total, np.nan)

    return _rolling_vol(r, b, n), _forward_vol(r, f, n)


def risk_adjusted_value(r, b: int, f: int):
    gret = geometric_mean_return(r, b, f)
    ann_back, ann_fwd = np.power(gret[0] + 1, 12) - 1, np.power(gret[1] + 1, 12) - 1
    vol_back, vol_fwd = vol_time_series(r, b, f)
    return ann_back / vol_back, ann_fwd / vol_fwd


def aum_features(aum_series, b: int):
    """Rolling mean AUM and AUM change over the lookback window."""
    n = len(aum_series)
    if n <= b:
        return np.full(n, np.nan), np.full(n, np.nan)

    aum_series = aum_series.reset_index(drop=True)
    pad = np.full(b - 1, np.nan)
    means, changes = [], []

    for i in range(b, n + 1):
        window = aum_series[i - b : i]
        non_null = window.dropna()
        means.append(non_null.mean() if len(non_null) >= (b - 2) else np.nan)

        start_val, end_val = aum_series.iloc[i - b], aum_series.iloc[i - 1]
        if start_val == 0 or end_val == 0:
            changes.append(0.0)
        else:
            changes.append(end_val / start_val - 1)

    return np.concatenate([pad, means]), np.concatenate([pad, changes])


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Raw panel construction (per-fund, per-month)
# ═══════════════════════════════════════════════════════════════════════════════

def build_raw_panel(
    funds: pd.DataFrame,
    returns: pd.DataFrame,
    returnspg: pd.DataFrame,
    aum: pd.DataFrame,
    gdxp: pd.DataFrame,
    start_date=START_DATE,
    current_date=CURRENT_DATE,
    vola_short: int = VOLA_WINDOW_SHORT,
    vola_long: int = VOLA_WINDOW_LONG,
) -> pd.DataFrame:
    """
    Iterate over funds and build the base panel with monthly observations
    including returns, drawdowns, volatility ratios, and ECDFs.
    """
    panels = []

    for idx in range(len(funds)):
        fund = funds.iloc[idx]
        start = max(start_date, next_month_start(fund.date_added))
        end   = min(current_date, fund.tr_stop)
        n_months = month_diff(start, end) + 1

        if n_months <= 1:
            continue

        months = (
            pd.date_range(start=start, end=end + pd.offsets.MonthEnd(), freq="MS")
            .to_series()
            .apply(lambda x: x.date())
            .to_list()
        )

        r  = returns.loc[returns.fund == fund.fund, start:end].squeeze()
        pg = returnspg.loc[returnspg.fund == fund.pg, start:end].squeeze()
        rs = gdxp.loc[start:end, "returns_stock"].squeeze()
        rt = gdxp.loc[start:end, "returns_treasury"].squeeze()
        rc = gdxp.loc[start:end, "returns_commodity"].squeeze()

        # Drawdowns
        def _dd_series(series):
            try:
                dd_list, maxdd_list = zip(*[drawdown(series[:i]) for i in range(1, len(series) + 1)])
                return list(dd_list), list(maxdd_list)
            except (TypeError, ValueError):
                return [], []

        dd, maxdd       = _dd_series(r)
        dd_s, maxdd_s   = _dd_series(rs)
        dd_t, maxdd_t   = _dd_series(rt)
        dd_c, maxdd_c   = _dd_series(rc)

        # Macro volatility features
        def _vola_feature(asset_col: str, window: int) -> list:
            result = []
            for m in months:
                try:
                    result.append(annualised_vol(gdxp.loc[subtract_months(m, window - 1) : m, asset_col]))
                except KeyError:
                    result.append(np.nan)
            return result

        df = pd.DataFrame({
            "fund":              [fund.fund] * n_months,
            "date":              months,
            "aum":               aum.loc[aum.fund == fund.fund, start:end].squeeze().values,
            "tr":                list(range(1, n_months + 1)),
            "tr_rev":            list(range(n_months, 0, -1)),
            "strategy":          [fund.strategy] * n_months,
            "returns":           r.values,
            "returns_pg":        pg.values,
            "returns_stock":     rs.values,
            "returns_treas":     rt.values,
            "returns_commo":     rc.values,
            "dd":                dd,
            "maxdd":             maxdd,
            "maxDDPhase_fund":   max_dd_phase(r),
            "dd_stock":          dd_s,
            "maxdd_stock":       maxdd_s,
            "maxDDPhase_stock":  max_dd_phase(rs),
            "dd_treas":          dd_t,
            "maxdd_treas":       maxdd_t,
            "maxDDPhase_treas":  max_dd_phase(rt),
            "dd_commo":          dd_c,
            "maxdd_commo":       maxdd_c,
            "maxDDPhase_commo":  max_dd_phase(rc),
            "vola_stock_s":      _vola_feature("returns_stock", vola_short),
            "vola_stock_l":      _vola_feature("returns_stock", vola_long),
            "vola_treas_s":      _vola_feature("returns_treasury", vola_short),
            "vola_treas_l":      _vola_feature("returns_treasury", vola_long),
            "vola_commo_s":      _vola_feature("returns_commodity", vola_short),
            "vola_commo_l":      _vola_feature("returns_commodity", vola_long),
            "F_returns":         [empirical_cdf(r[:i + 1].values, r.iloc[i]) for i in range(len(r))],
            "F_returns_pg":      [empirical_cdf(pg[:i + 1].values, pg.iloc[i]) for i in range(len(r))],
            "F_returns_stock":   [empirical_cdf(rs[:i + 1].values, rs.iloc[i]) for i in range(len(r))],
            "F_returns_treas":   [empirical_cdf(rt[:i + 1].values, rt.iloc[i]) for i in range(len(r))],
            "F_returns_commo":   [empirical_cdf(rc[:i + 1].values, rc.iloc[i]) for i in range(len(r))],
        })
        panels.append(df)

        if (idx + 1) % 100 == 0:
            logger.info("  … processed %d / %d funds", idx + 1, len(funds))

    result = pd.concat(panels, ignore_index=True)
    logger.info("Raw panel: %d rows × %d cols.", *result.shape)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — Derived features + target variables
# ═══════════════════════════════════════════════════════════════════════════════

def build_modelling_dataset(
    raw_panel: pd.DataFrame,
    funds: pd.DataFrame,
    gdxp: pd.DataFrame,
    b: int = LOOKBACK_MONTHS,
    f: int = FORECAST_MONTHS,
    current_date=CURRENT_DATE,
) -> pd.DataFrame:
    """
    Enrich the raw panel with Sharpe ratios, excess returns, AUM stats,
    volatility ratios, ECDFs of backward returns, and the three target
    variables (died / performance / outperformance).
    """
    rf_series = gdxp["rf"]
    fund_names = raw_panel.fund.unique()
    eligible = funds.loc[funds.fund.isin(fund_names) & (funds.months >= b + f), "fund"]
    panels = []

    for name in eligible:
        fund_df = raw_panel.loc[raw_panel.fund == name].copy()

        # Sharpe ratios & returns
        ret_b, ret_f     = geometric_mean_return(fund_df.returns.values, b, f)
        ret_b_pg, ret_f_pg = geometric_mean_return(fund_df.returns_pg.values, b, f)
        ret_b_stock, _   = geometric_mean_return(fund_df.returns_stock.values, b, f)
        ret_b_treas, _   = geometric_mean_return(fund_df.returns_treas.values, b, f)
        ret_b_commo, _   = geometric_mean_return(fund_df.returns_commo.values, b, f)

        sr_b, sr_f       = sharpe_ratio_ts(fund_df.returns.values, b, f, fund_df.date, rf_series)
        sr_b_pg, sr_f_pg = sharpe_ratio_ts(fund_df.returns_pg.values, b, f, fund_df.date, rf_series)
        er_b, er_f       = excess_return_ts(fund_df.returns.values, b, f, fund_df.date, rf_series)
        er_b_pg, er_f_pg = excess_return_ts(fund_df.returns_pg.values, b, f, fund_df.date, rf_series)
        sp_b, sp_f       = sharpe_product_ts(fund_df.returns.values, b, f, fund_df.date, rf_series)
        sp_b_pg, sp_f_pg = sharpe_product_ts(fund_df.returns_pg.values, b, f, fund_df.date, rf_series)
        rav_b, rav_f     = risk_adjusted_value(fund_df.returns.values, b, f)
        rav_b_pg, rav_f_pg = risk_adjusted_value(fund_df.returns_pg.values, b, f)
        aum_mean, aum_range = aum_features(fund_df.aum, b)

        # Assign features
        fund_df = fund_df.assign(
            aum_mean=aum_mean,
            aum_range=aum_range,
            return_fund=ret_b,
            return_pg=ret_b_pg,
            return_stock=ret_b_stock,
            return_treas=ret_b_treas,
            return_commo=ret_b_commo,
            sr_fund=sr_b,
            sr_pg=sr_b_pg,
            sr_f=sr_f,
            sr_f_pg=sr_f_pg,
            er_b=er_b,
            er_b_pg=er_b_pg,
            er_f=er_f,
            er_f_pg=er_f_pg,
            srProd_f=sp_f,
            srProd_f_pg=sp_f_pg,
            returns_f=np.power(geometric_mean_return(fund_df.returns.values, b, f)[1] + 1, 12) - 1,
            highvola_stock=fund_df.vola_stock_s / fund_df.vola_stock_l,
            highvola_treas=fund_df.vola_treas_s / fund_df.vola_treas_l,
            highvola_commo=fund_df.vola_commo_s / fund_df.vola_commo_l,
            F_fund=np.array([empirical_cdf(fund_df.returns.values[:i + 1], ret_b[i]) for i in range(len(ret_b))]),
            F_pg=np.array([empirical_cdf(fund_df.returns_pg.values[:i + 1], ret_b_pg[i]) for i in range(len(ret_b))]),
            F_stock=np.array([empirical_cdf(fund_df.returns_stock.values[:i + 1], ret_b_stock[i]) for i in range(len(ret_b))]),
            F_treas=np.array([empirical_cdf(fund_df.returns_treas.values[:i + 1], ret_b_treas[i]) for i in range(len(ret_b))]),
            F_commo=np.array([empirical_cdf(fund_df.returns_commo.values[:i + 1], ret_b_commo[i]) for i in range(len(ret_b))]),
        )

        # ── Target: died ─────────────────────────────────────────────────
        fund_df["died"] = "no"
        last_date = fund_df.date.iloc[-1]
        if last_date < current_date:
            fund_df.loc[fund_df.tr_rev <= f, "died"] = "yes"
        else:
            fund_df.loc[fund_df.date >= subtract_months(current_date, f), "died"] = np.nan

        # ── Target: performance ──────────────────────────────────────────
        fund_df["performance"] = "negative"
        fund_df.loc[fund_df.sr_f > 0, "performance"] = "positive"
        fund_df.loc[fund_df.returns_f.isna(), "performance"] = np.nan

        # ── Target: outperformance ───────────────────────────────────────
        fund_df["outperformance"] = np.nan
        for i, row in fund_df.iterrows():
            if np.isnan(row.sr_f) or np.isnan(row.sr_f_pg):
                continue
            if row.sr_f >= 0 or row.sr_f_pg >= 0:
                fund_df.loc[i, "outperformance"] = (
                    "outperform" if row.sr_f >= row.sr_f_pg else "underperform"
                )
            elif row.er_f > -0.7:
                fund_df.loc[i, "outperformance"] = (
                    "outperform" if row.srProd_f >= row.srProd_f_pg else "underperform"
                )
            else:
                fund_df.loc[i, "outperformance"] = (
                    "outperform" if row.er_f >= row.er_f_pg + 0.01 else "underperform"
                )

        panels.append(fund_df)

    result = pd.concat(panels, ignore_index=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = f"vars_b{b}_f{f}_{ts}.csv"
    result.to_csv(out_path, index=False)
    logger.info("Modelling dataset: %d rows, saved → %s", len(result), out_path)
    return result
