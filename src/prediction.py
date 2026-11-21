"""Fast local inference with serialized (joblib) models. No training happens here."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from . import config as C
from .data_loader import load_raw
from .evaluation import apply_interval
from .feature_engineering import make_dataset
from .preprocessing import clean
from .risk_classification import classify


class ModelsMissingError(Exception):
    """Raised when trained models are not on disk (run `python -m src.train_model`)."""


def models_exist(models_dir=C.MODELS_DIR) -> bool:
    return all((models_dir / f"model_h{h}.joblib").exists() for h in C.HORIZONS) and (models_dir / "metadata.json").exists()


def load_models(models_dir=C.MODELS_DIR) -> dict:
    if not models_exist(models_dir):
        raise ModelsMissingError("Trained models not found. Run:  python -m src.train_model")
    bundles = {h: joblib.load(models_dir / f"model_h{h}.joblib") for h in C.HORIZONS}
    metadata = json.loads((models_dir / "metadata.json").read_text())
    return {"bundles": bundles, "metadata": metadata, "thresholds": metadata["risk"]["thresholds_cases_per_week"]}


def prepare_data(path=None) -> pd.DataFrame:
    """Raw Excel -> cleaned, feature-engineered table (one row per ward-week)."""
    raw = load_raw(path) if path else load_raw()
    return make_dataset(clean(raw))


def predict_all(feat_df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Predict h=2,3,4 weeks ahead for every ward-week that has a complete feature vector.

    Returns one row per (ward, as-of week) with pred_h*, lo_h*, hi_h*, risk_h*, target_date_h*.
    """
    rows = feat_df.dropna(subset=C.FEATURE_COLS)
    out = rows[["ward_id", "city", "date"]].reset_index(drop=True).copy()
    X = rows[C.FEATURE_COLS]
    for h in C.HORIZONS:
        b = models["bundles"][h]
        pred = np.clip(b["model"].predict(X), 0, None)
        lo, hi = apply_interval(pred, b["interval"])
        out[f"pred_h{h}"], out[f"lo_h{h}"], out[f"hi_h{h}"] = pred, lo, hi
        out[f"risk_h{h}"] = classify(pred, models["thresholds"])
        out[f"target_date_h{h}"] = (rows["date"] + pd.Timedelta(weeks=h)).values
    return out


def horizon_table(pred_row: pd.Series) -> pd.DataFrame:
    """Long table (one row per horizon) for a single ward / as-of week."""
    return pd.DataFrame([{
        "horizon_weeks": h, "target_date": pred_row[f"target_date_h{h}"],
        "predicted_cases": pred_row[f"pred_h{h}"], "lower": pred_row[f"lo_h{h}"],
        "upper": pred_row[f"hi_h{h}"], "risk": pred_row[f"risk_h{h}"],
    } for h in C.HORIZONS])


def summarize_risk(preds_at_date: pd.DataFrame, horizon: int) -> dict:
    counts = preds_at_date[f"risk_h{horizon}"].value_counts()
    return {lab: int(counts.get(lab, 0)) for lab in C.RISK_LABELS}


def driver_percentiles(row: pd.Series, reference: pd.DataFrame) -> dict:
    """Where this ward-week sits (0-100 percentile) versus the TRAINING-period history.

    Descriptive only: a high percentile means 'unusually high compared with history', not 'cause'.
    """
    cols = ["rain_cum2", "rain_cum4", "rain_lag3", "temp_roll4", "humid_roll4", "cases_roll4",
            "population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
    out = {}
    for c in cols:
        ref = reference[c].dropna()
        out[c] = float((ref <= row[c]).mean() * 100) if len(ref) and pd.notna(row[c]) else float("nan")
    return out
