"""
Model training module — XGBoost classification with time-series cross-validation.

Trains an XGBoost classifier for each of the three target variables
(died, performance, outperformance) using GridSearchCV over a
TimeSeriesSplit with class-weight-aware sample weighting.
"""

import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.utils import class_weight
from warnings import simplefilter
from xgboost import XGBClassifier

from config import (
    DEFAULT_BETA,
    FEATURE_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_LABELS,
    TEST_SIZE,
    XGB_CV_FOLDS,
    XGB_PARAM_GRID,
)

simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Captures everything produced by a single training run."""
    target: str
    best_params: dict
    model: XGBClassifier
    scaler: preprocessing.MinMaxScaler
    cm_train: np.ndarray
    cm_test: np.ndarray
    metrics: dict
    report_row: pd.Series
    X_test: pd.DataFrame = field(repr=False)
    Y_test: pd.Series = field(repr=False)


# ── Scoring registry ─────────────────────────────────────────────────────────

def _build_scorers(beta: float) -> dict:
    return {
        "accuracy":    make_scorer(accuracy_score),
        "f1":          make_scorer(f1_score, pos_label=1),
        "f1b":         make_scorer(fbeta_score, beta=beta, pos_label=1),
        "precision":   make_scorer(precision_score, pos_label=1),
        "recall":      make_scorer(recall_score, pos_label=1),
        "specificity": make_scorer(recall_score, pos_label=0),
    }


# ── Main training function ───────────────────────────────────────────────────

def train_model(
    data_path: str,
    target: str,
    b: int = 12,
    f: int = 12,
    beta: float | None = None,
    param_grid: dict | None = None,
) -> TrainResult:
    """
    End-to-end model training for one target variable.

    Parameters
    ----------
    data_path : str
        Path to the CSV produced by feature engineering.
    target : str
        One of ``"died"``, ``"performance"``, ``"outperformance"``.
    b, f : int
        Lookback / forecast horizon (months).
    beta : float, optional
        Weight for the F-beta scorer. Falls back to ``DEFAULT_BETA[target]``.
    param_grid : dict, optional
        Override the default XGBoost hyperparameter grid.

    Returns
    -------
    TrainResult
    """
    if beta is None:
        beta = DEFAULT_BETA[target]
    if param_grid is None:
        param_grid = XGB_PARAM_GRID

    neg_label, pos_label = TARGET_LABELS[target]
    logger.info("Training target=%s  beta=%.2f", target, beta)

    # ── Load & select features ───────────────────────────────────────────
    data = pd.read_csv(data_path, parse_dates=["date"])
    keep = ["fund", "date"] + FEATURE_COLUMNS + [target]
    data = data[keep].dropna().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    logger.info("  observations: %d", len(data))

    # ── Train / test split ───────────────────────────────────────────────
    X = data.drop(columns=[target, "date", "fund"])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    logger.info("  train=%d  test=%d", len(y_train), len(y_test))

    # ── Scaling ──────────────────────────────────────────────────────────
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      index=X_test.index,  columns=X_test.columns)

    # ── Encode labels ────────────────────────────────────────────────────
    y_train_enc = (y_train != neg_label).astype(int)
    y_test_enc  = (y_test  != neg_label).astype(int)

    # ── Sample weights ───────────────────────────────────────────────────
    cw = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y_train_enc,
    )
    sw = np.where(y_train_enc == 0, cw[0], cw[1])

    # ── Grid search with time-series CV ──────────────────────────────────
    scorers = _build_scorers(beta)
    tscv = TimeSeriesSplit(n_splits=XGB_CV_FOLDS)

    gs = GridSearchCV(
        estimator=XGBClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=tscv.split(X_train_s),
        scoring=scorers,
        refit="f1b",
        n_jobs=-1,
    )

    t0 = datetime.now()
    gs.fit(X_train_s, y_train_enc, sample_weight=sw)
    duration = datetime.now() - t0
    logger.info("  GridSearch done in %s — best: %s", duration, gs.best_params_)

    # ── Refit on best params ─────────────────────────────────────────────
    model = XGBClassifier(random_state=RANDOM_STATE, **gs.best_params_)
    model.fit(X_train_s, y_train_enc, sample_weight=sw)

    # ── Predictions & metrics ────────────────────────────────────────────
    y_pred_train = model.predict(X_train_s)
    y_pred_test  = model.predict(X_test_s)

    cm_train = confusion_matrix(y_train_enc, y_pred_train, labels=[0, 1])
    cm_test  = confusion_matrix(y_test_enc,  y_pred_test,  labels=[0, 1])

    metrics = {
        "accuracy":    accuracy_score(y_test_enc, y_pred_test),
        "precision":   precision_score(y_test_enc, y_pred_test, pos_label=1),
        "recall":      recall_score(y_test_enc, y_pred_test, pos_label=1),
        "specificity": recall_score(y_test_enc, y_pred_test, pos_label=0),
        "f1":          f1_score(y_test_enc, y_pred_test, pos_label=1),
        "f1b":         fbeta_score(y_test_enc, y_pred_test, beta=beta, pos_label=1),
    }

    # ── Store metadata on the model object ───────────────────────────────
    model.feature_names = list(X_train.columns)
    model.target = target
    model.b = b
    model.f = f
    model.bestparams = gs.best_params_

    # ── Report row ───────────────────────────────────────────────────────
    cv = gs.cv_results_
    report_row = pd.Series({
        "Model":                "XGBoost",
        "Target":               target,
        "Acc. Train (CV)":      cv["mean_test_accuracy"].mean(),
        "Acc. Test":            metrics["accuracy"],
        "Precision Train (CV)": cv["mean_test_precision"].mean(),
        "Precision Test":       metrics["precision"],
        "Recall Train (CV)":    cv["mean_test_recall"].mean(),
        "Recall Test":          metrics["recall"],
        "Spec. Train (CV)":     cv["mean_test_specificity"].mean(),
        "Spec. Test":           metrics["specificity"],
        "F1β Train (CV)":       cv["mean_test_f1b"].mean(),
        "F1β Test":             metrics["f1b"],
        "Timestamp":            time.strftime("%Y%m%d-%H%M%S"),
    })

    # ── Persist ──────────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_path = MODELS_DIR / f"xgb_{target}_b{b}_f{f}_{ts}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    report_path = RESULTS_DIR / f"report_{target}_b{b}_f{f}_{ts}.csv"
    report_row.to_frame().T.to_csv(report_path, index=False)
    logger.info("  Model saved → %s", model_path.name)
    logger.info("  Report saved → %s", report_path.name)

    return TrainResult(
        target=target,
        best_params=gs.best_params_,
        model=model,
        scaler=scaler,
        cm_train=cm_train,
        cm_test=cm_test,
        metrics=metrics,
        report_row=report_row,
        X_test=X_test_s,
        Y_test=y_test_enc,
    )
