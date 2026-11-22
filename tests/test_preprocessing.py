"""Focused validation tests for src.preprocessing.clean.

Run from the project root:
    python -m unittest tests.test_preprocessing -v
"""
import unittest

import numpy as np
import pandas as pd

from src.config import STATIC_COLS, WEATHER_COLS
from src.preprocessing import clean


def make_rows(ward_id="WARD_01", city="Test City", periods=5):
    dates = pd.date_range("2025-01-06", periods=periods, freq="7D")
    values = np.arange(periods, dtype=float)
    data = {
        "date": dates,
        "city": [city] * periods,
        "ward_id": [ward_id] * periods,
        "population_density_sqkm": [100.0] * periods,
        "water_logging_risk_index": [0.4] * periods,
        "sanitation_score": [0.7] * periods,
        "avg_temp_c": 25.0 + values,
        "max_temp_c": 30.0 + values,
        "min_temp_c": 20.0 + values,
        "rainfall_mm": 1.0 + values,
        "relative_humidity_pct": 60.0 + values,
        "dengue_cases": 2.0 + values,
    }
    return pd.DataFrame(data)



class TestPreprocessingValidation(unittest.TestCase):
    def test_drops_rows_missing_date_or_ward_id(self):
        df = make_rows()
        df.loc[0, "date"] = pd.NaT
        df.loc[1, "ward_id"] = None

        result = clean(df)

        self.assertFalse(result["date"].isna().any())
        self.assertFalse(result["ward_id"].isna().any())
        self.assertEqual(result["ward_id"].nunique(), 1)

    def test_duplicate_ward_date_keeps_last_record(self):
        df = make_rows()
        duplicate = df.iloc[[2]].copy()
        duplicate["dengue_cases"] = 99
        df = pd.concat([df, duplicate], ignore_index=True)

        result = clean(df)
        selected = result.loc[
            (result["ward_id"] == "WARD_01")
            & (result["date"] == pd.Timestamp("2025-01-20")),
            "dengue_cases",
        ]

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0], 99)

    def test_output_is_sorted_by_ward_and_date(self):
        df = pd.concat(
            [make_rows("WARD_B"), make_rows("WARD_A")],
            ignore_index=True,
        ).sample(frac=1, random_state=7)

        result = clean(df)
        expected = result.sort_values(["ward_id", "date"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(result, expected)

    def test_missing_week_is_inserted_on_seven_day_grid(self):
        df = make_rows().drop(index=2)

        result = clean(df)

        self.assertEqual(len(result), 5)
        gaps = result.groupby("ward_id")["date"].diff().dropna()
        self.assertTrue((gaps == pd.Timedelta(days=7)).all())
        self.assertIn(pd.Timestamp("2025-01-20"), set(result["date"]))

    def test_missing_dengue_case_is_not_imputed(self):
        df = make_rows().drop(index=2)

        result = clean(df)
        inserted = result.loc[result["date"] == pd.Timestamp("2025-01-20")].iloc[0]

        self.assertTrue(pd.isna(inserted["dengue_cases"]))

    def test_short_weather_gap_is_interpolated(self):
        df = make_rows()
        df.loc[2, "rainfall_mm"] = np.nan

        result = clean(df)
        middle = result.loc[result["date"] == pd.Timestamp("2025-01-20"), "rainfall_mm"].iloc[0]

        self.assertAlmostEqual(middle, 3.0)

    def test_static_columns_are_filled_within_ward(self):
        df = make_rows()
        df.loc[2, STATIC_COLS] = np.nan

        result = clean(df)
        middle = result.loc[result["date"] == pd.Timestamp("2025-01-20")].iloc[0]

        self.assertTrue(all(pd.notna(middle[col]) for col in STATIC_COLS))
        self.assertEqual(middle["population_density_sqkm"], 100.0)

    def test_negative_rainfall_is_clipped_to_zero(self):
        df = make_rows()
        df.loc[2, "rainfall_mm"] = -5.0

        result = clean(df)

        self.assertGreaterEqual(result["rainfall_mm"].min(), 0.0)
        self.assertEqual(
            result.loc[result["date"] == pd.Timestamp("2025-01-20"), "rainfall_mm"].iloc[0],
            0.0,
        )

    def test_wards_are_processed_independently(self):
        ward_a = make_rows("WARD_A")
        ward_b = make_rows("WARD_B")
        ward_b["rainfall_mm"] = [101.0, 102.0, 103.0, 104.0, 105.0]
        combined = pd.concat([ward_a, ward_b], ignore_index=True)

        result = clean(combined)
        first_b = result[result["ward_id"] == "WARD_B"].iloc[0]

        self.assertEqual(first_b["rainfall_mm"], 101.0)
        self.assertEqual(result.groupby("ward_id").size().to_dict(), {"WARD_A": 5, "WARD_B": 5})


if __name__ == "__main__":
    unittest.main()
