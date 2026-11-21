"""Validation of the RISK-TIER methodology against history, using OUT-OF-SAMPLE predictions only.

Inputs are the files written by training:
  * validation_predictions.csv - made by models fitted on the training years only
  * test_predictions.csv       - made by the final models on the untouched 2025 test year
Nothing here changes thresholds; it measures how well the existing tiers match what actually happened.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C
from .risk_classification import classify, tier_index


def calibration_table(preds: pd.DataFrame) -> pd.DataFrame:
    """For each predicted tier: how many ward-weeks, mean predicted and mean ACTUAL cases."""
    g = preds.groupby("predicted_tier").agg(ward_weeks=("actual", "size"), mean_predicted=("predicted", "mean"), mean_actual=("actual", "mean"))
    return g.reindex(list(C.RISK_LABELS)).reset_index().rename(columns={"predicted_tier": "tier"})


def confusion(preds: pd.DataFrame) -> pd.DataFrame:
    order = list(C.RISK_LABELS)
    return pd.crosstab(preds["actual_tier"], preds["predicted_tier"]).reindex(index=order, columns=order, fill_value=0)


def tier_summary(preds: pd.DataFrame) -> dict:
    a, p = tier_index(preds["actual_tier"]), tier_index(preds["predicted_tier"])
    hi_a, hi_p = a >= 2, p >= 2
    return {
        "ward_weeks": int(len(preds)),
        "exact_tier_accuracy": float(np.mean(a == p)),
        "within_one_tier": float(np.mean(np.abs(a - p) <= 1)),
        "actual_high_or_critical": int(hi_a.sum()),
        "high_or_critical_recall": float(hi_p[hi_a].mean()) if hi_a.any() else float("nan"),
        "high_or_critical_precision": float(hi_a[hi_p].mean()) if hi_p.any() else float("nan"),
        "missed_alarms_pred_low_actual_high_or_critical": int(((p == 0) & hi_a).sum()),
        "false_alarms_pred_high_or_critical_actual_low": int(((a == 0) & hi_p).sum()),
    }


def monthly_tier_table(preds: pd.DataFrame) -> pd.DataFrame:
    d = preds.assign(month=preds["target_date"].dt.month)
    out = d.groupby("month").apply(lambda g: pd.Series({
        "ward_weeks": len(g), "actual_LOW_pct": 100 * (g["actual_tier"] == "LOW").mean(), "predicted_LOW_pct": 100 * (g["predicted_tier"] == "LOW").mean(),
        "actual_HIGH_or_CRITICAL_pct": 100 * g["actual_tier"].isin(["HIGH", "CRITICAL"]).mean(),
        "predicted_HIGH_or_CRITICAL_pct": 100 * g["predicted_tier"].isin(["HIGH", "CRITICAL"]).mean(),
        "mean_actual_cases": g["actual"].mean(), "mean_predicted_cases": g["predicted"].mean()}), include_groups=False)
    return out.reset_index()


def seasonal_history(feat_df: pd.DataFrame, target_date, thresholds, window_days: int = 21) -> dict:
    """Dataset-only facts: what actually happened in the same calendar weeks of every year (2016-2025)."""
    t = pd.Timestamp(target_date)
    doy = feat_df["date"].dt.dayofyear
    diff = (doy - t.dayofyear).abs()
    circ = np.minimum(diff, 365 - diff)
    sel = feat_df[(circ <= window_days) & feat_df["dengue_cases"].notna()]
    if sel.empty:
        return {}
    tiers = classify(sel["dengue_cases"].to_numpy(), thresholds)
    return {
        "calendar_window": f"within {window_days} days of {t.strftime('%d %b')} in every year of the dataset",
        "ward_weeks": int(len(sel)), "years": int(sel["date"].dt.year.nunique()),
        "share_actual_LOW_pct": float(100 * np.mean(tiers == "LOW")),
        "share_actual_HIGH_or_CRITICAL_pct": float(100 * np.mean(np.isin(tiers, ["HIGH", "CRITICAL"]))),
        "mean_cases_per_ward_week": float(sel["dengue_cases"].mean()), "max_cases_in_a_ward_week": int(sel["dengue_cases"].max()),
    }


def evidence_status(target_date, last_observed) -> str:
    """Is a forecast for `target_date` an honest out-of-sample prediction, or a replay of training data?"""
    t = pd.Timestamp(target_date)
    if t > pd.Timestamp(last_observed):
        return "future forecast (no actual data yet)"
    if t <= pd.Timestamp(C.VAL_END):
        return "backtest on weeks the final model was trained on (in-sample; NOT a fair accuracy test)"
    return "held-out test period (2025): not used for training or model selection"


def write_report(val: pd.DataFrame, test: pd.DataFrame, path=None) -> str:
    """Write reports/risk_validation.md (+ CSVs) from out-of-sample predictions."""
    path = path or (C.REPORTS_DIR / "risk_validation.md")
    lines = ["# MosquiTech - Risk-tier validation (out-of-sample)",
             "_Thresholds are NOT tuned here. This report measures how well the training-period tiers match what happened._", ""]
    csvs = []
    for name, df in (("validation (2023-2024, train-only models)", val), ("test (2025, untouched)", test)):
        s, cal = tier_summary(df), calibration_table(df).round(2)
        lines += [f"## {name}", f"- Ward-weeks: {s['ward_weeks']:,}; exact tier accuracy {s['exact_tier_accuracy']*100:.1f}%; within one tier {s['within_one_tier']*100:.1f}%.",
                  f"- HIGH/CRITICAL recall {s['high_or_critical_recall']*100:.1f}%, precision {s['high_or_critical_precision']*100:.1f}%.",
                  f"- Missed alarms (forecast LOW, actually HIGH/CRITICAL): {s['missed_alarms_pred_low_actual_high_or_critical']} of {s['actual_high_or_critical']}.",
                  f"- False alarms (forecast HIGH/CRITICAL, actually LOW): {s['false_alarms_pred_high_or_critical_actual_low']}.", "",
                  "Calibration - mean actual cases for each forecast tier:", "", cal.to_markdown(index=False) if hasattr(cal, "to_markdown") and _has_tabulate() else cal.to_string(index=False), ""]
    m = monthly_tier_table(pd.concat([val, test])[lambda d: d["horizon"] == 3]).round(1)
    lines += ["## Seasonality check (3-week forecasts, validation + test)",
              "Dry-season months are ~100% LOW in the ACTUAL data as well, so an all-LOW dashboard in December is the correct answer, not a defect.", "",
              m.to_markdown(index=False) if _has_tabulate() else m.to_string(index=False)]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    val = pd.read_csv(C.REPORTS_DIR / "validation_predictions.csv", parse_dates=["target_date", "as_of_date"])
    test = pd.read_csv(C.REPORTS_DIR / "test_predictions.csv", parse_dates=["target_date", "as_of_date"])
    return write_report(val, test)


if __name__ == "__main__":
    main()
