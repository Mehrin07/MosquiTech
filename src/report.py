"""Generate reports/evaluation_report.md from the CSV/JSON files written by training (no hand-typed numbers)."""
from __future__ import annotations
import json
import pandas as pd
from . import config as C


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> str:
    q = json.loads((C.REPORTS_DIR / "data_quality.json").read_text())
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    mc = pd.read_csv(C.REPORTS_DIR / "model_comparison.csv")
    tm = pd.read_csv(C.REPORTS_DIR / "test_metrics.csv")
    imp = pd.read_csv(C.REPORTS_DIR / "feature_importance.csv")
    th = meta["risk"]["thresholds_cases_per_week"]

    val = mc[mc.split == "validation"].round(3)[["horizon", "model", "MAE", "RMSE", "R2", "selected"]]
    test = tm.round(3)[["horizon", "model", "MAE", "RMSE", "R2", "baseline_MAE", "baseline_RMSE", "baseline_R2"]]
    cls = tm.assign(**{c: (tm[c] * 100).round(1) for c in ["interval_coverage", "tier_accuracy", "within_one_tier", "high_or_critical_recall"]})
    cls = cls[["horizon", "interval_coverage", "tier_accuracy", "within_one_tier", "high_or_critical_recall", "macro_f1"]].round(3)
    cls["horizon"] = cls["horizon"].astype(int)
    cls.columns = ["horizon", "80% interval coverage (%)", "tier accuracy (%)", "within one tier (%)", "HIGH/CRITICAL recall (%)", "macro F1"]

    parts = [
        "# MosquiTech - Evaluation Report",
        f"_Generated from training outputs ({meta['trained_at_utc']} UTC, scikit-learn {meta['scikit_learn']})._",
        "## 1. Data",
        f"- {q['n_records']:,} weekly ward records, {q['n_wards']} wards in {q['n_cities']} cities, {q['date_min']} to {q['date_max']} ({q['n_weeks']} weeks).",
        f"- Missing values: {q['missing_values'] or 'none'}; duplicate ward-weeks: {q['duplicate_ward_dates']}; observation gaps (days): {q['observation_gap_days']}.",
        f"- Weekly cases per ward: mean {q['dengue_cases']['mean']}, max {q['dengue_cases']['max']}, {q['dengue_cases']['share_zero']*100:.0f}% of weeks have zero cases "
        f"(zero-inflated, right-skewed). {q['dengue_cases']['n_above_iqr_threshold']} weeks exceed the IQR outlier fence ({q['dengue_cases']['iqr_outlier_threshold']:.0f}); they were kept because they are the outbreak peaks the system must anticipate.",
        "## 2. Chronological split (by target week)",
        f"- Train up to {meta['split']['train_end']}, validation up to {meta['split']['val_end']}, test afterwards.",
        _md(pd.DataFrame([{"horizon": int(h), "train rows": v["train"], "validation rows": v["val"], "test rows": v["test"],
                           "test target weeks": f"{v['test_target_range'][0]} to {v['test_target_range'][1]}"}
                          for h, v in meta["split"]["sizes"].items()])),
        "## 3. Risk thresholds (fitted on training data only)",
        f"- MEDIUM >= {th[0]:.1f}, HIGH >= {th[1]:.1f}, CRITICAL >= {th[2]:.1f} cases/ward/week (training-period {', '.join(str(int(x*100)) + 'th' for x in C.RISK_QUANTILES)} percentiles).",
        "## 4. Model comparison on VALIDATION data (used for selection)",
        _md(val),
        "## 5. Final TEST-year performance (scored once, after refitting the selected model on train + validation)",
        _md(test),
        "Baseline = last 4-week mean. Lower MAE/RMSE and higher R2 are better.",
        "## 6. Interval and risk-tier quality on the test year",
        _md(cls),
        "## 7. Top permutation-importance features (validation data)",
    ]
    for h in C.HORIZONS:
        top = imp[imp.horizon == h].head(5)[["label", "share_pct"]].round(1)
        parts.append(f"**{h}-week model:** " + "; ".join(f"{r.label} ({r.share_pct}%)" for r in top.itertuples()))
    parts += [
        "\nImportance describes what the model uses; it does not establish causation.",
        "## 8. Limitations",
        "- Test-year scores are lower than validation scores: expect real-world accuracy closer to the test numbers.",
        "- Predictions are conditional averages and under-predict sharp peaks; accuracy declines as the horizon grows.",
        "- Risk tiers are relative to historical experience, not clinical thresholds.",
        "- Intervals are empirical (validation residuals) and not guaranteed to cover.",
        "- A single 2016-2025 dataset with no mobility, serotype, intervention or geographic data.",
    ]
    text = "\n\n".join(parts) + "\n"
    (C.REPORTS_DIR / "evaluation_report.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    main()
