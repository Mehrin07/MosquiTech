"""Load the hackathon Excel dataset and report on its quality (real data only)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import DATA_PATH, REQUIRED_COLUMNS


class DataError(Exception):
    """Raised when the dataset is missing or has an unexpected schema."""


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset not found at {path}. Place the Excel file at data/hackathon_dataset.xlsx.")
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:  # corrupted file, wrong format, ...
        raise DataError(f"Could not read {path}: {exc}") from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Dataset is missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summary statistics describing the raw dataset (all computed, nothing hardcoded)."""
    d = df.sort_values(["ward_id", "date"])
    gaps = d.groupby("ward_id")["date"].diff().dt.days.dropna()
    cases = df["dengue_cases"]
    q1, q3 = cases.quantile([0.25, 0.75])
    iqr_upper = q3 + 1.5 * (q3 - q1)
    return {
        "n_records": int(len(df)),
        "n_cities": int(df["city"].nunique()),
        "n_wards": int(df["ward_id"].nunique()),
        "wards_per_city": {k: int(v) for k, v in df.groupby("city")["ward_id"].nunique().items()},
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_weeks": int(df["date"].nunique()),
        "observation_gap_days": {int(k): int(v) for k, v in gaps.value_counts().items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ward_dates": int(df.duplicated(["ward_id", "date"]).sum()),
        "dengue_cases": {
            "mean": round(float(cases.mean()), 3), "max": int(cases.max()),
            "share_zero": round(float((cases == 0).mean()), 3),
            "p90": float(cases.quantile(0.9)), "p99": float(cases.quantile(0.99)),
            "iqr_outlier_threshold": float(iqr_upper),
            "n_above_iqr_threshold": int((cases > iqr_upper).sum()),
        },
        "negative_rainfall_rows": int((df["rainfall_mm"] < 0).sum()),
        "temp_order_violations": int(((df["min_temp_c"] > df["avg_temp_c"]) | (df["avg_temp_c"] > df["max_temp_c"])).sum()),
    }
