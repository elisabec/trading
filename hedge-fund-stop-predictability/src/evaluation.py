"""
Model evaluation and interpretation.

Produces confusion-matrix heatmaps, classification reports, and three
types of XGBoost feature-importance analysis (weight, gain, cover)
across the three target variables.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import plot_importance

from config import FEATURE_COLUMNS, FIGURES_DIR, RESULTS_DIR, TARGET_LABELS

logger = logging.getLogger(__name__)

# Consistent style
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "figure.figsize":    (10, 5),
    "savefig.bbox":      "tight",
})


# ═══════════════════════════════════════════════════════════════════════════════
#  Confusion matrices
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm: np.ndarray,
    target: str,
    split: str = "test",
    save: bool = True,
) -> None:
    """
    Plot and optionally save a confusion-matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        2×2 confusion matrix (rows = true, cols = predicted).
    target : str
        Target variable name (``"died"`` | ``"performance"`` | ``"outperformance"``).
    split : str
        ``"train"`` or ``"test"`` — used in the figure title and filename.
    """
    neg, pos = TARGET_LABELS[target]
    labels = [neg, pos]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{l} (pred)" for l in labels],
        yticklabels=[f"{l} (true)" for l in labels],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {target} ({split})")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    if save:
        path = FIGURES_DIR / f"cm_{target}_{split}.png"
        fig.savefig(path)
        logger.info("Saved confusion matrix → %s", path.name)
    plt.close(fig)


def print_classification_summary(metrics: dict, target: str) -> None:
    """Pretty-print test-set metrics to the console."""
    header = f"  Classification Summary — {target}  "
    border = "═" * len(header)
    print(f"\n{border}\n{header}\n{border}")
    for k, v in metrics.items():
        print(f"  {k:<14s}: {v:.4f}")
    print(border + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature importance (weight / gain / cover)
# ═══════════════════════════════════════════════════════════════════════════════

def _importance_df(models: dict, importance_type: str) -> pd.DataFrame:
    """
    Build a DataFrame of feature importances for all three target models.

    Parameters
    ----------
    models : dict
        ``{target_name: fitted XGBClassifier}``
    importance_type : str
        One of ``"weight"``, ``"gain"``, ``"cover"``.
    """
    rows = {}
    for target, model in models.items():
        scores = model.get_booster().get_score(fmap="", importance_type=importance_type)
        rows[target] = scores

    df = pd.DataFrame(rows).T
    # Normalise each row to sum to 1
    df_rel = df.div(df.sum(axis=1), axis=0)
    return df, df_rel


def plot_feature_importance_comparison(
    models: dict,
    importance_type: str = "weight",
    save: bool = True,
) -> pd.DataFrame:
    """
    Horizontal bar chart of relative feature importances averaged across
    the three target variables.
    """
    _, df_rel = _importance_df(models, importance_type)
    avg = df_rel.mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    avg.plot.barh(ax=ax, color="black", edgecolor="black")
    ax.set_title(f"Feature Importance — {importance_type} (avg. across targets)")
    ax.set_xlabel("Relative importance")
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / f"feature_importance_{importance_type}_avg.pdf"
        fig.savefig(path)
        logger.info("Saved feature importance → %s", path.name)
    plt.close(fig)
    return df_rel


def plot_feature_importance_per_target(
    models: dict,
    importance_type: str = "weight",
    save: bool = True,
) -> None:
    """Side-by-side bar chart of relative importance per target."""
    _, df_rel = _importance_df(models, importance_type)

    fig, ax = plt.subplots(figsize=(10, 6))
    df_rel.T.plot.barh(ax=ax, rot=0)
    ax.set_title(f"Feature Importance — {importance_type} (per target)")
    ax.set_xlabel("Relative importance")
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / f"feature_importance_{importance_type}_per_target.png"
        fig.savefig(path)
        logger.info("Saved per-target importance → %s", path.name)
    plt.close(fig)


def export_importance_tables(models: dict) -> pd.DataFrame:
    """
    Compute weight/gain/cover importances for the canonical feature order
    and export a single CSV.
    """
    order = FEATURE_COLUMNS
    targets = list(models.keys())

    col_names = []
    for imp_type in ("weight", "gain", "cover"):
        for t in targets:
            col_names.append(f"{imp_type}_{t}")

    result = pd.DataFrame(index=order, columns=col_names)

    for imp_type in ("weight", "gain", "cover"):
        _, df_rel = _importance_df(models, imp_type)
        for t in targets:
            for feat in order:
                result.loc[feat, f"{imp_type}_{t}"] = df_rel.loc[t, feat] if feat in df_rel.columns else np.nan

    result = (result.astype(float) * 100).round(2)
    out = RESULTS_DIR / "feature_importance_summary.csv"
    result.to_csv(out)
    logger.info("Exported importance table → %s", out.name)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Built-in XGBoost importance plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_xgb_importance(models: dict, save: bool = True) -> None:
    """Wrapper around ``xgboost.plot_importance`` for each target model."""
    for target, model in models.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_importance(model, ax=ax)
        ax.set_title(f"XGBoost Feature Importance — {target}")
        plt.tight_layout()

        if save:
            path = FIGURES_DIR / f"xgb_importance_{target}.png"
            fig.savefig(path)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: evaluate a set of TrainResults
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_all(train_results: list) -> None:
    """
    Given a list of ``TrainResult`` objects (one per target), produce
    all standard evaluation artefacts.
    """
    models = {}
    for tr in train_results:
        # Confusion matrices
        plot_confusion_matrix(tr.cm_train, tr.target, split="train")
        plot_confusion_matrix(tr.cm_test,  tr.target, split="test")
        print_classification_summary(tr.metrics, tr.target)
        models[tr.target] = tr.model

    # Feature importance
    for imp_type in ("weight", "gain", "cover"):
        plot_feature_importance_comparison(models, imp_type)
        plot_feature_importance_per_target(models, imp_type)

    plot_xgb_importance(models)
    export_importance_tables(models)

    # Consolidated report
    rows = [tr.report_row for tr in train_results]
    report = pd.DataFrame(rows)
    out = RESULTS_DIR / "consolidated_report.csv"
    report.to_csv(out, index=False)
    logger.info("Consolidated report → %s", out.name)
    print("\n", report.to_string(index=False))
