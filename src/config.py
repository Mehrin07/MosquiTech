"""Central configuration for MosquiTech (paths, horizons, split dates, feature names)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hackathon_dataset.xlsx"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

HORIZONS = (2, 3, 4)          # forecast horizons in weeks
RANDOM_STATE = 42

# Chronological split, defined on the DATE OF THE TARGET (the week being predicted).
# train: 2016-01 .. 2022-12 | validation: 2023-01 .. 2024-12 | test: 2025-01 .. 2025-12
TRAIN_END = "2022-12-31"
VAL_END = "2024-12-31"

# Risk tiers are fitted on TRAINING-period weekly cases only (see risk_classification.py)
RISK_QUANTILES = (0.70, 0.85, 0.95)      # -> MEDIUM, HIGH, CRITICAL cut-points
RISK_LABELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
RISK_COLORS = {"LOW": "#2ecc71", "MEDIUM": "#f4d03f", "HIGH": "#e67e22", "CRITICAL": "#e74c3c"}

REQUIRED_COLUMNS = [
    "date", "city", "ward_id", "population_density_sqkm", "water_logging_risk_index",
    "sanitation_score", "avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm",
    "relative_humidity_pct", "dengue_cases",
]
STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
WEATHER_COLS = ["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]

FEATURE_COLS = [
    # rainfall (ward-grouped lags + cumulative sums)
    "rainfall_mm", "rain_lag1", "rain_lag2", "rain_lag3", "rain_lag4", "rain_cum2", "rain_cum4",
    # temperature / humidity
    "avg_temp_c", "temp_roll2", "temp_roll4", "relative_humidity_pct", "humid_roll2", "humid_roll4",
    # historical dengue cases
    "cases_now", "cases_lag1", "cases_lag2", "cases_lag3", "cases_roll4",
    # ward vulnerability
    "population_density_sqkm", "water_logging_risk_index", "sanitation_score",
    # calendar seasonality (known in advance, so not leakage)
    "week_sin", "week_cos",
]

FEATURE_LABELS = {
    "rainfall_mm": "Rainfall this week", "rain_lag1": "Rainfall 1 wk ago", "rain_lag2": "Rainfall 2 wk ago",
    "rain_lag3": "Rainfall 3 wk ago", "rain_lag4": "Rainfall 4 wk ago",
    "rain_cum2": "Cumulative rainfall (2 wk)", "rain_cum4": "Cumulative rainfall (4 wk)",
    "avg_temp_c": "Avg temperature (now)", "temp_roll2": "Avg temperature (2-wk mean)",
    "temp_roll4": "Avg temperature (4-wk mean)", "relative_humidity_pct": "Humidity (now)",
    "humid_roll2": "Humidity (2-wk mean)", "humid_roll4": "Humidity (4-wk mean)",
    "cases_now": "Dengue cases this week", "cases_lag1": "Dengue cases 1 wk ago",
    "cases_lag2": "Dengue cases 2 wk ago", "cases_lag3": "Dengue cases 3 wk ago",
    "cases_roll4": "Dengue cases (4-wk mean)", "population_density_sqkm": "Population density",
    "water_logging_risk_index": "Waterlogging risk", "sanitation_score": "Sanitation score",
    "week_sin": "Season (sin)", "week_cos": "Season (cos)",
}

# ---- Explainability: features grouped into human-readable drivers --------------------------------
FEATURE_GROUPS = {
    "Rainfall": ["rainfall_mm", "rain_lag1", "rain_lag2", "rain_lag3", "rain_lag4", "rain_cum2", "rain_cum4"],
    "Temperature": ["avg_temp_c", "temp_roll2", "temp_roll4"],
    "Humidity": ["relative_humidity_pct", "humid_roll2", "humid_roll4"],
    "Recent dengue cases": ["cases_now", "cases_lag1", "cases_lag2", "cases_lag3", "cases_roll4"],
    "Ward vulnerability": ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"],
    "Season (calendar)": ["week_sin", "week_cos"],
}
WEATHER_GROUPS = ("Rainfall", "Temperature", "Humidity")
