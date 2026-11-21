"""Ward-grouped lag / rolling features and future-aligned targets.

Every lag, rolling mean and shift is computed INSIDE each ward (groupby ward_id) on data sorted
by date, so one ward's rainfall can never leak into another ward's features.
Rolling windows include the current week and only look backwards. Targets look forward.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import FEATURE_COLS, HORIZONS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df (sorted by ward_id, date) with all FEATURE_COLS added."""
    d = df.sort_values(["ward_id", "date"]).reset_index(drop=True).copy()
    g = d.groupby("ward_id", sort=False)

    for k in (1, 2, 3, 4):
        d[f"rain_lag{k}"] = g["rainfall_mm"].shift(k)
    d["rain_cum2"] = g["rainfall_mm"].transform(lambda s: s.rolling(2, min_periods=2).sum())
    d["rain_cum4"] = g["rainfall_mm"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    d["temp_roll2"] = g["avg_temp_c"].transform(lambda s: s.rolling(2, min_periods=2).mean())
    d["temp_roll4"] = g["avg_temp_c"].transform(lambda s: s.rolling(4, min_periods=4).mean())
    d["humid_roll2"] = g["relative_humidity_pct"].transform(lambda s: s.rolling(2, min_periods=2).mean())
    d["humid_roll4"] = g["relative_humidity_pct"].transform(lambda s: s.rolling(4, min_periods=4).mean())

    d["cases_now"] = d["dengue_cases"]
    for k in (1, 2, 3):
        d[f"cases_lag{k}"] = g["dengue_cases"].shift(k)
    d["cases_roll4"] = g["dengue_cases"].transform(lambda s: s.rolling(4, min_periods=4).mean())

    week = d["date"].dt.isocalendar().week.astype(float)
    d["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    d["week_cos"] = np.cos(2 * np.pi * week / 52.0)
    return d


def add_targets(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Add target_h{h} (cases h weeks AHEAD) and target_date_h{h} for each horizon.

    df must already be sorted by ward_id, date on a regular weekly grid.
    The row for week t holds only information known at t; target_h is the value at t + h weeks.
    """
    d = df.copy()
    g = d.groupby("ward_id", sort=False)
    for h in horizons:
        d[f"target_h{h}"] = g["dengue_cases"].shift(-h)
        d[f"target_date_h{h}"] = d["date"] + pd.Timedelta(weeks=h)
    return d


def make_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    return add_targets(build_features(clean_df))


def model_rows(feat_df: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """Rows with a complete feature vector (and a known target when horizon is given)."""
    need = list(FEATURE_COLS)
    if horizon is not None:
        need.append(f"target_h{horizon}")
    return feat_df.dropna(subset=need)
