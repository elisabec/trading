#!/usr/bin/env python3
"""
Hedge-Fund Stop Predictability — Master Pipeline
=================================================

End-to-end workflow:

    1. Load & filter the fund universe
    2. Construct peer-group benchmarks
    3. Build the raw panel  (fund × month)
    4. Engineer features + target variables
    5. Train XGBoost classifiers for each target
    6. Evaluate & export results

Usage
-----
    python main.py                      # full pipeline
    python main.py --skip-data          # skip steps 1-4 (re-use existing CSV)
    python main.py --data-file <path>   # point to a specific feature CSV

"""

import argparse
import logging
import sys

from config import (
    CURRENT_DATE,
    DEFAULT_BETA,
    FORECAST_MONTHS,
    LOOKBACK_MONTHS,
    START_DATE,
    TARGET_LABELS,
    VOLA_WINDOW_LONG,
    VOLA_WINDOW_SHORT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def build_data() -> str:
    """Run data loading, filtering, and feature engineering. Returns CSV path."""
    from data_loader import (
        compute_peer_group_indices,
        load_filtered_funds,
        load_peer_group_returns,
        load_raw_data,
    )
    from feature_engineering import build_modelling_dataset, build_raw_panel

    logger.info("Step 1 — Loading raw data")
    raw = load_raw_data()
    funds = load_filtered_funds()

    # Override peer-group key to use strategy-level benchmarks
    funds["pg_EH"] = funds["pg"]
    funds["pg"] = funds["strategy"]

    logger.info("Step 2 — Loading peer-group benchmarks")
    returnspg = load_peer_group_returns()

    logger.info("Step 3 — Building raw panel")
    raw_panel = build_raw_panel(
        funds,
        raw["returns"],
        returnspg,
        raw["aum"],
        raw["gdxp"],
        start_date=START_DATE,
        current_date=CURRENT_DATE,
        vola_short=VOLA_WINDOW_SHORT,
        vola_long=VOLA_WINDOW_LONG,
    )

    logger.info("Step 4 — Feature engineering")
    modelling_df = build_modelling_dataset(
        raw_panel,
        funds,
        raw["gdxp"],
        b=LOOKBACK_MONTHS,
        f=FORECAST_MONTHS,
        current_date=CURRENT_DATE,
    )

    # Return the path of the auto-saved CSV
    import glob
    latest = sorted(glob.glob("vars_b*_f*_*.csv"))[-1]
    return latest


def run_training(data_path: str) -> list:
    """Train one model per target variable and return results."""
    from modelling import train_model

    results = []
    for target in TARGET_LABELS:
        logger.info("Step 5 — Training model: %s", target)
        result = train_model(
            data_path=data_path,
            target=target,
            b=LOOKBACK_MONTHS,
            f=FORECAST_MONTHS,
        )
        results.append(result)
    return results


def run_evaluation(results: list) -> None:
    """Generate all evaluation artefacts."""
    from evaluation import evaluate_all

    logger.info("Step 6 — Evaluation & reporting")
    evaluate_all(results)


def main():
    parser = argparse.ArgumentParser(
        description="Hedge-fund stop-predictability pipeline.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data building; requires --data-file.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to a pre-built feature CSV.",
    )
    args = parser.parse_args()

    if args.skip_data:
        if args.data_file is None:
            logger.error("--skip-data requires --data-file. Exiting.")
            sys.exit(1)
        data_path = args.data_file
    else:
        data_path = build_data()

    logger.info("Using feature data: %s", data_path)
    results = run_training(data_path)
    run_evaluation(results)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
