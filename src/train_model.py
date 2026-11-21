"""Train, select and evaluate dengue forecasting models.   Run:  python -m src.train_model

Protocol (see ARCHITECTURE.md):
  1. Build ward-grouped features and future-aligned targets for h = 2, 3, 4 weeks.
  2. Split CHRONOLOGICALLY by target date: train <= 2022, validation 2023-2024, test 2025.
  3. Fit risk thresholds on training targets only.
  4. Fit every candidate on TRAIN, score on VALIDATION, choose the best by validation RMSE.
  5. Empirical prediction-interval offsets + permutation importance come from VALIDATION only.
  6. Refit the chosen model on TRAIN + VALIDATION, score ONCE on the untouched TEST year.
  7. Save models (joblib), metrics and reports.
"""
from __future__ import annotations
import argparse, json, platform, time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from . import config as C
from .data_loader import data_quality_report, load_raw
from .evaluation import (apply_interval, interval_coverage, permutation_importance_table,
                         regression_metrics, residual_interval, risk_metrics)
from .feature_engineering import make_dataset, model_rows
from .preprocessing import clean
from .risk_classification import classify, fit_thresholds


def candidate_models() -> dict:
    rs = C.RANDOM_STATE
    return {
        "RandomForest (leaf=5)": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=rs),
        "RandomForest (leaf=15)": RandomForestRegressor(n_estimators=200, min_samples_leaf=15, max_features=0.5, n_jobs=-1, random_state=rs),
        "HistGradientBoosting (leaves=15)": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=rs),
        "HistGradientBoosting (depth=4)": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=80, l2_regularization=1.0, random_state=rs),
        "HistGradientBoosting (poisson)": HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=rs),
    }


def split_by_target_date(rows: pd.DataFrame, h: int):
    t = rows[f"target_date_h{h}"]
    train = rows[t <= pd.Timestamp(C.TRAIN_END)]
    val = rows[(t > pd.Timestamp(C.TRAIN_END)) & (t <= pd.Timestamp(C.VAL_END))]
    test = rows[t > pd.Timestamp(C.VAL_END)]
    return train, val, test


def _pred(model, X):
    return np.clip(model.predict(X[C.FEATURE_COLS]), 0, None)


def _pred_table(rows, h, y, pred, interval, thresholds, split):
    lo, hi = apply_interval(pred, interval)
    return pd.DataFrame({
        "split": split, "horizon": h, "ward_id": rows["ward_id"].values, "city": rows["city"].values,
        "as_of_date": rows["date"].values, "target_date": rows[f"target_date_h{h}"].values,
        "actual": np.asarray(y), "predicted": pred, "lower": lo, "upper": hi,
        "actual_tier": classify(np.asarray(y), thresholds), "predicted_tier": classify(pred, thresholds),
    })


def main(quiet: bool = False) -> dict:
    log = (lambda *a, **k: None) if quiet else print
    t0 = time.time()
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.REPORTS_DIR.mkdir(exist_ok=True)

    raw = load_raw()
    quality = data_quality_report(raw)
    (C.REPORTS_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2))
    log(f"Loaded {quality['n_records']:,} rows | {quality['n_wards']} wards | {quality['date_min']} -> {quality['date_max']}")

    data = make_dataset(clean(raw))

    # Risk thresholds: TRAINING-period observed weekly cases only (target-date <= TRAIN_END)
    train_cases = data.loc[data["date"] <= pd.Timestamp(C.TRAIN_END), "dengue_cases"]
    thresholds = fit_thresholds(train_cases)
    log(f"Risk thresholds (cases/week) MEDIUM>={thresholds[0]:.2f}  HIGH>={thresholds[1]:.2f}  CRITICAL>={thresholds[2]:.2f}")

    comparison, test_rows_out, val_rows_out, importances, final_metrics = [], [], [], [], []
    models_meta = {}
    split_sizes = {}

    for h in C.HORIZONS:
        rows = model_rows(data, h)
        train, val, test = split_by_target_date(rows, h)
        ytr, yva, yte = train[f"target_h{h}"], val[f"target_h{h}"], test[f"target_h{h}"]
        split_sizes[h] = {"train": len(train), "val": len(val), "test": len(test),
                          "train_target_range": [str(train[f"target_date_h{h}"].min().date()), str(train[f"target_date_h{h}"].max().date())],
                          "val_target_range": [str(val[f"target_date_h{h}"].min().date()), str(val[f"target_date_h{h}"].max().date())],
                          "test_target_range": [str(test[f"target_date_h{h}"].min().date()), str(test[f"target_date_h{h}"].max().date())]}
        assert train[f"target_date_h{h}"].max() < val[f"target_date_h{h}"].min() <= val[f"target_date_h{h}"].max() < test[f"target_date_h{h}"].min()
        log(f"\n=== Horizon {h} weeks | train {len(train):,} | val {len(val):,} | test {len(test):,} ===")

        # Naive reference: "next weeks look like the last 4 weeks"
        base_val = regression_metrics(yva, val["cases_roll4"])
        comparison.append({"horizon": h, "model": "Baseline: last 4-wk mean", "split": "validation", **base_val, "selected": False})

        fitted, scores = {}, {}
        for name, est in candidate_models().items():
            m = clone(est).fit(train[C.FEATURE_COLS], ytr)
            p = _pred(m, val)
            met = regression_metrics(yva, p)
            fitted[name], scores[name] = m, met
            comparison.append({"horizon": h, "model": name, "split": "validation", **met, "selected": False})
            log(f"  {name:36s} val MAE {met['MAE']:.3f}  RMSE {met['RMSE']:.3f}  R2 {met['R2']:.3f}")

        best = min(scores, key=lambda n: scores[n]["RMSE"])        # selection uses VALIDATION only
        for r in comparison:
            if r["horizon"] == h and r["model"] == best and r["split"] == "validation":
                r["selected"] = True
        log(f"  -> selected on validation RMSE: {best}")

        val_pred = _pred(fitted[best], val)
        interval = residual_interval(yva, val_pred)
        imp = permutation_importance_table(fitted[best], val, yva)
        imp.insert(0, "horizon", h)
        importances.append(imp)
        val_rows_out.append(_pred_table(val, h, yva, val_pred, interval, thresholds, "validation"))

        # Refit chosen configuration on train + validation, then touch the test set exactly once
        trval = pd.concat([train, val])
        final = clone(candidate_models()[best]).fit(trval[C.FEATURE_COLS], trval[f"target_h{h}"])
        test_pred = _pred(final, test)
        tm = regression_metrics(yte, test_pred)
        lo, hi = apply_interval(test_pred, interval)
        cov = interval_coverage(yte, lo, hi)
        rm = risk_metrics(yte, test_pred, thresholds)
        base_test = regression_metrics(yte, test["cases_roll4"])
        comparison.append({"horizon": h, "model": "Baseline: last 4-wk mean", "split": "test", **base_test, "selected": False})
        comparison.append({"horizon": h, "model": best, "split": "test", **tm, "selected": True})
        final_metrics.append({"horizon": h, "model": best, **tm, "interval_coverage": cov,
                              "nominal_coverage": interval["nominal_coverage"], **rm,
                              "baseline_MAE": base_test["MAE"], "baseline_RMSE": base_test["RMSE"], "baseline_R2": base_test["R2"]})
        test_rows_out.append(_pred_table(test, h, yte, test_pred, interval, thresholds, "test"))
        log(f"  TEST ({best}): MAE {tm['MAE']:.3f}  RMSE {tm['RMSE']:.3f}  R2 {tm['R2']:.3f} | 80% interval coverage {cov:.2%} | tier acc {rm['tier_accuracy']:.2%}")

        joblib.dump({"model": final, "model_name": best, "horizon": h, "feature_cols": C.FEATURE_COLS,
                     "interval": interval, "importance": imp.to_dict("records")},
                    C.MODELS_DIR / f"model_h{h}.joblib", compress=3)
        models_meta[h] = {"model": best, "interval": interval}

    pd.DataFrame(comparison).to_csv(C.REPORTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(final_metrics).to_csv(C.REPORTS_DIR / "test_metrics.csv", index=False)
    pd.concat(importances).to_csv(C.REPORTS_DIR / "feature_importance.csv", index=False)
    pd.concat(val_rows_out).to_csv(C.REPORTS_DIR / "validation_predictions.csv", index=False)
    pd.concat(test_rows_out).to_csv(C.REPORTS_DIR / "test_predictions.csv", index=False)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "horizons": list(C.HORIZONS), "feature_cols": C.FEATURE_COLS,
        "split": {"train_end": C.TRAIN_END, "val_end": C.VAL_END, "split_basis": "date of the predicted (target) week", "sizes": split_sizes},
        "risk": {"quantiles": list(C.RISK_QUANTILES), "thresholds_cases_per_week": thresholds, "labels": list(C.RISK_LABELS),
                 "fitted_on": f"training-period weekly cases (<= {C.TRAIN_END})"},
        "models": models_meta, "training_seconds": round(time.time() - t0, 1),
    }
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (C.REPORTS_DIR / "risk_thresholds.json").write_text(json.dumps(metadata["risk"], indent=2))
    from . import report
    report.main()
    log(f"\nDone in {metadata['training_seconds']} s. Models -> {C.MODELS_DIR}, reports -> {C.REPORTS_DIR}")
    return metadata


if __name__ == "__main__":
    argparse.ArgumentParser(description="Train MosquiTech models").parse_args()
    main()
