
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import RISK_LABELS, RISK_QUANTILES


def fit_thresholds(train_cases: pd.Series, quantiles=RISK_QUANTILES) -> list[float]:
    s = pd.Series(train_cases).dropna()
    if s.empty:
        raise ValueError("Cannot fit risk thresholds without training case counts.")
    t = [float(np.quantile(s, q)) for q in quantiles]
    # keep strictly increasing even for very skewed data
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1.0
    return t


def classify(cases, thresholds):
    """Map case counts (scalar / array / Series) to tier names. cases >= threshold moves up a tier."""
    arr = np.asarray(cases, dtype=float)

    idx = np.searchsorted(np.asarray(thresholds, dtype=float), arr, side="right")
    labels = np.asarray(RISK_LABELS)[idx]
    return str(labels) if arr.ndim == 0 else labels


def tier_index(labels) -> np.ndarray:
    lookup = {l: i for i, l in enumerate(RISK_LABELS)}
    return np.array([lookup[l] for l in labels])
