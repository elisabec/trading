"""
Configuration and constants for the hedge fund stop-predictability study.

Central registry of all hyperparameters, file paths, and universe definitions
so that every module draws from a single source of truth.
"""

import os
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Sample period ────────────────────────────────────────────────────────────
START_DATE = datetime(2004, 2, 1).date()
CURRENT_DATE = datetime(2019, 9, 1).date()

# ── Fund filtering ───────────────────────────────────────────────────────────
MIN_TRACK_RECORD_MONTHS = 24
CURRENCY = "USD"

STRATEGIES = [
    "Long Short Equities",
    "Multi-Strategy",
    "Bottom-Up",
    "Event Driven",
    "Fixed Income",
    "CTA_Managed Futures",
    "Arbitrage",
    "Macro",
    "Distressed Debt",
    "Top-Down",
    "Dual Approach",
    "Relative Value",
    "Value",
    "Diversified Debt",
]

# ── Feature engineering ──────────────────────────────────────────────────────
LOOKBACK_MONTHS = 12          # backward-looking window  (b)
FORECAST_MONTHS = 12          # forward-looking window   (f)
VOLA_WINDOW_SHORT = 12        # short-term volatility window (months)
VOLA_WINDOW_LONG = 24         # long-term volatility window  (months)
SIMILAR_RETURN_TOL = 1e-6     # threshold for "too-similar" consecutive returns

# ── Target variables ─────────────────────────────────────────────────────────
TARGET_LABELS = {
    "died":            ("no", "yes"),
    "performance":     ("negative", "positive"),
    "outperformance":  ("underperform", "outperform"),
}

# Explanatory variables used in modelling (order matters for interpretation)
FEATURE_COLUMNS = [
    "tr",
    "aum_mean",
    "aum_range",
    "sr_fund",
    "sr_pg",
    "highvola_stock",
    "highvola_treas",
    "highvola_commo",
    "return_fund",
    "return_pg",
    "return_stock",
    "return_treas",
    "return_commo",
    "maxDDPhase_fund",
    "maxDDPhase_stock",
    "maxDDPhase_treas",
    "maxDDPhase_commo",
    "F_fund",
    "F_pg",
    "F_stock",
    "F_treas",
    "F_commo",
]

# ── XGBoost hyperparameter grid ─────────────────────────────────────────────
XGB_PARAM_GRID = {
    "max_depth":      [4, 6, 8, 10],
    "n_estimators":   [100, 200, 500, 1000],
    "learning_rate":  [0.01, 0.1, 0.2, 0.3],
    "gamma":          [2],
}

XGB_CV_FOLDS = 10             # TimeSeriesSplit folds
TEST_SIZE = 0.25
RANDOM_STATE = 0

# ── Mapping from target name → default F-beta weight ────────────────────────
DEFAULT_BETA = {
    "died":           2.0,
    "performance":    0.5,
    "outperformance": 0.5,
}
