"""MosquiTech test-suite.  Run:  python -m unittest discover -s tests -v   (pytest also works)."""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import config as C
from src.data_loader import DataError, data_quality_report, load_raw
from src.evaluation import apply_interval, interval_coverage, regression_metrics, risk_metrics
from src.feature_engineering import add_targets, build_features, make_dataset, model_rows
from src.preprocessing import clean
from src.prediction import load_models, predict_all, models_exist
from src.recommendations import DISCLAIMER, recommend
from src.risk_classification import classify, fit_thresholds
from src.train_model import split_by_target_date

_CACHE = {}


def raw():
    if "raw" not in _CACHE:
        _CACHE["raw"] = load_raw()
    return _CACHE["raw"]


def cleaned():
    if "clean" not in _CACHE:
        _CACHE["clean"] = clean(raw())
    return _CACHE["clean"]


def dataset():
    if "ds" not in _CACHE:
        _CACHE["ds"] = make_dataset(cleaned())
    return _CACHE["ds"]


class TestDataLoading(unittest.TestCase):
    def test_schema_and_size(self):
        df = raw()
        self.assertEqual(list(df.columns), C.REQUIRED_COLUMNS)
        self.assertEqual(len(df), 31320)
        self.assertEqual(df["ward_id"].nunique(), 60)
        self.assertEqual(df["city"].nunique(), 5)

    def test_quality_report(self):
        q = data_quality_report(raw())
        self.assertEqual(q["duplicate_ward_dates"], 0)
        self.assertEqual(q["missing_values"], {})
        self.assertEqual(q["observation_gap_days"], {7: 31260})

    def test_missing_file_raises_clean_error(self):
        with self.assertRaises(DataError):
            load_raw("does_not_exist.xlsx")

    def test_clean_keeps_regular_weekly_grid(self):
        d = cleaned()
        gaps = d.groupby("ward_id")["date"].diff().dropna()
        self.assertTrue((gaps == pd.Timedelta(days=7)).all())

    def test_clean_fills_gap_without_inventing_cases(self):
        r = raw()
        w = r[r.ward_id == r.ward_id.iloc[0]].sort_values("date")
        gapped = w.drop(w.index[10])                       # remove one week
        out = clean(gapped)
        self.assertEqual(len(out), len(w))
        missing_date = w.iloc[10]["date"]
        row = out[out.date == missing_date].iloc[0]
        self.assertTrue(np.isnan(row["dengue_cases"]))     # cases never imputed


class TestFeatureEngineering(unittest.TestCase):
    def test_rain_lags_are_ward_specific_and_exact(self):
        ds, c = dataset(), cleaned()
        for ward in ["CCC_Ward_01", ds.ward_id.unique()[37]]:
            w_raw = c[c.ward_id == ward].set_index("date")["rainfall_mm"]
            w_f = ds[ds.ward_id == ward].set_index("date")
            for k in (1, 2, 3, 4):
                ok = w_f[f"rain_lag{k}"].dropna()
                expected = w_raw.shift(k).loc[ok.index]
                np.testing.assert_allclose(ok.values, expected.values)

    def test_no_lag_leaks_across_wards(self):
        ds = dataset()
        first4 = ds.groupby("ward_id").head(4)
        self.assertTrue(first4["rain_lag4"].isna().all())  # would be filled if previous ward bled in

    def test_cumulative_rain_and_rolling_means(self):
        ds = dataset()
        w = ds[ds.ward_id == ds.ward_id.iloc[0]].reset_index(drop=True)
        i = 100
        self.assertAlmostEqual(w.loc[i, "rain_cum2"], w.loc[i - 1:i, "rainfall_mm"].sum())
        self.assertAlmostEqual(w.loc[i, "rain_cum4"], w.loc[i - 3:i, "rainfall_mm"].sum())
        self.assertAlmostEqual(w.loc[i, "temp_roll4"], w.loc[i - 3:i, "avg_temp_c"].mean())
        self.assertAlmostEqual(w.loc[i, "humid_roll2"], w.loc[i - 1:i, "relative_humidity_pct"].mean())
        self.assertAlmostEqual(w.loc[i, "cases_roll4"], w.loc[i - 3:i, "dengue_cases"].mean())

    def test_target_alignment(self):
        ds = dataset()
        lookup = ds.set_index(["ward_id", "date"])["dengue_cases"]
        sample = ds.dropna(subset=["target_h2", "target_h3", "target_h4"]).sample(400, random_state=1)
        for h in C.HORIZONS:
            future = [lookup[(w, d + pd.Timedelta(weeks=h))] for w, d in zip(sample.ward_id, sample.date)]
            np.testing.assert_array_equal(sample[f"target_h{h}"].values, np.array(future))
            self.assertTrue((sample[f"target_date_h{h}"] == sample.date + pd.Timedelta(weeks=h)).all())

    def test_last_rows_have_no_target(self):
        ds = dataset()
        last = ds.groupby("ward_id").tail(4)
        self.assertTrue(last["target_h4"].isna().all())


class TestLeakagePrevention(unittest.TestCase):
    def test_features_ignore_the_future(self):
        """Corrupting every observation after a cut-off must not change features at or before it."""
        c = cleaned()
        cut = pd.Timestamp("2022-06-06")
        tampered = c.copy()
        after = tampered["date"] > cut
        for col in C.WEATHER_COLS + ["dengue_cases"]:
            tampered.loc[after, col] = tampered.loc[after, col] * 7 + 99
        a = build_features(c)
        b = build_features(tampered)
        a, b = a[a.date <= cut].reset_index(drop=True), b[b.date <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[C.FEATURE_COLS], b[C.FEATURE_COLS])

    def test_split_is_chronological_with_no_overlap(self):
        ds = dataset()
        for h in C.HORIZONS:
            tr, va, te = split_by_target_date(model_rows(ds, h), h)
            col = f"target_date_h{h}"
            self.assertLess(tr[col].max(), va[col].min())
            self.assertLess(va[col].max(), te[col].min())
            self.assertLessEqual(tr[col].max(), pd.Timestamp(C.TRAIN_END))
            self.assertGreater(te[col].min(), pd.Timestamp(C.VAL_END))
            self.assertEqual(len(tr) + len(va) + len(te), len(model_rows(ds, h)))

    def test_risk_thresholds_use_training_period_only(self):
        ds = dataset()
        train_cases = ds.loc[ds.date <= pd.Timestamp(C.TRAIN_END), "dengue_cases"]
        meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
        self.assertEqual(meta["risk"]["thresholds_cases_per_week"], fit_thresholds(train_cases))
        # changing later (validation/test) data does not move thresholds
        altered = ds.copy()
        altered.loc[altered.date > pd.Timestamp(C.TRAIN_END), "dengue_cases"] = 1000
        t2 = fit_thresholds(altered.loc[altered.date <= pd.Timestamp(C.TRAIN_END), "dengue_cases"])
        self.assertEqual(t2, fit_thresholds(train_cases))


class TestRisk(unittest.TestCase):
    def test_classification_boundaries(self):
        t = [2.0, 7.0, 13.0]
        self.assertEqual(list(classify([0, 1.99, 2, 6.99, 7, 12.99, 13, 40], t)),
                         ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH", "CRITICAL", "CRITICAL"])
        self.assertEqual(classify(8.0, t), "HIGH")

    def test_thresholds_strictly_increasing(self):
        t = fit_thresholds(pd.Series([0] * 100))           # degenerate data still gives valid tiers
        self.assertTrue(t[0] < t[1] < t[2])
        with self.assertRaises(ValueError):
            fit_thresholds(pd.Series([], dtype=float))


class TestEvaluation(unittest.TestCase):
    def test_metrics_known_values(self):
        m = regression_metrics([1, 2, 3], [1, 2, 5])
        self.assertAlmostEqual(m["MAE"], 2 / 3)
        self.assertAlmostEqual(m["RMSE"], np.sqrt(4 / 3))

    def test_interval_helpers(self):
        lo, hi = apply_interval([0.5, 5], {"lo": -1.0, "hi": 2.0})
        np.testing.assert_allclose(lo, [0, 4])
        np.testing.assert_allclose(hi, [2.5, 7])
        self.assertEqual(interval_coverage([1, 10], lo, hi), 0.5)

    def test_risk_metrics_perfect(self):
        r = risk_metrics([0, 3, 8, 20], [0, 3, 8, 20], [2, 7, 13])
        self.assertEqual(r["tier_accuracy"], 1.0)


class TestModels(unittest.TestCase):
    def test_serialization_roundtrip(self):
        ds = dataset()
        rows = model_rows(ds, 2).head(3000)
        m = RandomForestRegressor(n_estimators=10, random_state=0).fit(rows[C.FEATURE_COLS], rows["target_h2"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.joblib"
            joblib.dump(m, p)
            m2 = joblib.load(p)
        np.testing.assert_array_equal(m.predict(rows[C.FEATURE_COLS]), m2.predict(rows[C.FEATURE_COLS]))

    def test_trained_models_predict_all_horizons(self):
        self.assertTrue(models_exist(), "run `python -m src.train_model` first")
        models = load_models()
        preds = predict_all(dataset(), models)
        for h in C.HORIZONS:
            self.assertTrue((preds[f"pred_h{h}"] >= 0).all())
            self.assertTrue((preds[f"lo_h{h}"] <= preds[f"hi_h{h}"]).all())
            self.assertTrue(set(preds[f"risk_h{h}"]).issubset(set(C.RISK_LABELS)))
            self.assertTrue(preds[f"pred_h{h}"].notna().all())
        self.assertEqual(preds["ward_id"].nunique(), 60)

    def test_predictions_are_dynamic_not_constant(self):
        preds = predict_all(dataset(), load_models())
        self.assertGreater(preds["pred_h3"].nunique(), 1000)
        self.assertGreater(preds["pred_h3"].std(), 1.0)

    def test_trained_model_beats_naive_baseline_on_test(self):
        m = pd.read_csv(C.REPORTS_DIR / "test_metrics.csv")
        self.assertTrue((m["MAE"] < m["baseline_MAE"]).all())

    def test_test_predictions_only_cover_test_year(self):
        p = pd.read_csv(C.REPORTS_DIR / "test_predictions.csv", parse_dates=["target_date"])
        self.assertGreater(p["target_date"].min(), pd.Timestamp(C.VAL_END))


class TestRecommendations(unittest.TestCase):
    def test_high_risk_rules(self):
        recs = recommend("CRITICAL", 14.0, {"water_logging_risk_index": 0.9, "sanitation_score": 0.3},
                         {"rain_cum4": 90, "population_density_sqkm": 80}, 6.0)
        text = " ".join(r["action"] for r in recs)
        self.assertIn("medical resources", text)
        self.assertIn("stagnant-water", text)
        self.assertIn("sanitation", text)
        self.assertEqual(recs[0]["priority"], "High")

    def test_low_risk_default(self):
        recs = recommend("LOW", 0.2, {"water_logging_risk_index": 0.3, "sanitation_score": 0.8}, {}, 0.0)
        self.assertEqual(len(recs), 1)
        self.assertIn("routine", recs[0]["action"])

    def test_disclaimer_present(self):
        self.assertIn("not medical advice", DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
