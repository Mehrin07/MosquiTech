"""Metrics, empirical prediction intervals, explainability and risk-tier evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from .config import FEATURE_COLS, FEATURE_LABELS, RANDOM_STATE
from .risk_classification import classify, tier_index


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def residual_interval(y_true, y_pred, lower_q=0.10, upper_q=0.90) -> dict:
    """Empirical 80% interval offsets from VALIDATION residuals (actual - predicted)."""
    r = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return {"lo": float(np.quantile(r, lower_q)), "hi": float(np.quantile(r, upper_q)),
            "nominal_coverage": float(upper_q - lower_q)}


def apply_interval(pred, interval) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, float)
    return np.clip(pred + interval["lo"], 0, None), np.clip(pred + interval["hi"], 0, None)


def interval_coverage(y_true, lo, hi) -> float:
    y = np.asarray(y_true, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def risk_metrics(y_true_cases, y_pred_cases, thresholds) -> dict:
    """Compare the tier implied by actual cases with the tier implied by predicted cases."""
    t = tier_index(classify(y_true_cases, thresholds))
    p = tier_index(classify(y_pred_cases, thresholds))
    return {
        "tier_accuracy": float(accuracy_score(t, p)),
        "within_one_tier": float(np.mean(np.abs(t - p) <= 1)),
        "macro_f1": float(f1_score(t, p, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
        "high_or_critical_recall": float(np.mean(p[t >= 2] >= 2)) if (t >= 2).any() else float("nan"),
    }


def permutation_importance_table(model, X: pd.DataFrame, y, n_repeats=5, max_rows=6000) -> pd.DataFrame:
    """Permutation importance on held-out VALIDATION data (increase in MSE when a feature is shuffled).

    This measures how much the model relies on a feature, not whether it causes dengue.
    """
    if len(X) > max_rows:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    res = permutation_importance(model, X[FEATURE_COLS], y, n_repeats=n_repeats,
                                 random_state=RANDOM_STATE, scoring="neg_mean_squared_error", n_jobs=-1)
    out = pd.DataFrame({
        "feature": FEATURE_COLS,
        "label": [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "importance": res.importances_mean,
        "std": res.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    total = out["importance"].clip(lower=0).sum()
    out["share_pct"] = 100 * out["importance"].clip(lower=0) / total if total > 0 else 0.0
    return out
