# MosquiTech - Evaluation Report

_Generated from training outputs (2026-11-21T10:21:11+00:00 UTC, scikit-learn 1.9.1)._

## 1. Data

- 31,320 weekly ward records, 60 wards in 5 cities, 2016-01-04 to 2025-12-29 (522 weeks).

- Missing values: none; duplicate ward-weeks: 0; observation gaps (days): {'7': 31260}.

- Weekly cases per ward: mean 2.636, max 41, 58% of weeks have zero cases (zero-inflated, right-skewed). 2495 weeks exceed the IQR outlier fence (10); they were kept because they are the outbreak peaks the system must anticipate.

## 2. Chronological split (by target week)

- Train up to 2022-12-31, validation up to 2024-12-31, test afterwards.

| horizon | train rows | validation rows | test rows | test target weeks |
|---|---|---|---|---|
| 2 | 21540 | 6300 | 3120 | 2025-01-06 to 2025-12-29 |
| 3 | 21480 | 6300 | 3120 | 2025-01-06 to 2025-12-29 |
| 4 | 21420 | 6300 | 3120 | 2025-01-06 to 2025-12-29 |

## 3. Risk thresholds (fitted on training data only)

- MEDIUM >= 2.0, HIGH >= 7.0, CRITICAL >= 13.0 cases/ward/week (training-period 70th, 85th, 95th percentiles).

## 4. Model comparison on VALIDATION data (used for selection)

| horizon | model | MAE | RMSE | R2 | selected |
|---|---|---|---|---|---|
| 2 | Baseline: last 4-wk mean | 2.01 | 3.76 | 0.382 | False |
| 2 | RandomForest (leaf=5) | 1.091 | 2.189 | 0.79 | False |
| 2 | RandomForest (leaf=15) | 1.096 | 2.223 | 0.784 | False |
| 2 | HistGradientBoosting (leaves=15) | 1.069 | 2.143 | 0.799 | False |
| 2 | HistGradientBoosting (depth=4) | 1.066 | 2.116 | 0.804 | True |
| 2 | HistGradientBoosting (poisson) | 1.059 | 2.154 | 0.797 | False |
| 3 | Baseline: last 4-wk mean | 2.212 | 4.067 | 0.276 | False |
| 3 | RandomForest (leaf=5) | 1.063 | 2.104 | 0.806 | False |
| 3 | RandomForest (leaf=15) | 1.061 | 2.106 | 0.806 | False |
| 3 | HistGradientBoosting (leaves=15) | 1.039 | 2.06 | 0.814 | True |
| 3 | HistGradientBoosting (depth=4) | 1.058 | 2.088 | 0.809 | False |
| 3 | HistGradientBoosting (poisson) | 1.021 | 2.062 | 0.814 | False |
| 4 | Baseline: last 4-wk mean | 2.476 | 4.451 | 0.133 | False |
| 4 | RandomForest (leaf=5) | 1.369 | 2.679 | 0.686 | False |
| 4 | RandomForest (leaf=15) | 1.349 | 2.663 | 0.69 | True |
| 4 | HistGradientBoosting (leaves=15) | 1.377 | 2.685 | 0.685 | False |
| 4 | HistGradientBoosting (depth=4) | 1.379 | 2.716 | 0.677 | False |
| 4 | HistGradientBoosting (poisson) | 1.373 | 2.748 | 0.67 | False |

## 5. Final TEST-year performance (scored once, after refitting the selected model on train + validation)

| horizon | model | MAE | RMSE | R2 | baseline_MAE | baseline_RMSE | baseline_R2 |
|---|---|---|---|---|---|---|---|
| 2 | HistGradientBoosting (depth=4) | 1.141 | 2.268 | 0.669 | 1.793 | 3.342 | 0.282 |
| 3 | HistGradientBoosting (leaves=15) | 1.182 | 2.27 | 0.669 | 2.012 | 3.648 | 0.145 |
| 4 | RandomForest (leaf=15) | 1.204 | 2.303 | 0.659 | 2.195 | 3.907 | 0.019 |

Baseline = last 4-week mean. Lower MAE/RMSE and higher R2 are better.

## 6. Interval and risk-tier quality on the test year

| horizon | 80% interval coverage (%) | tier accuracy (%) | within one tier (%) | HIGH/CRITICAL recall (%) | macro F1 |
|---|---|---|---|---|---|
| 2.0 | 78.1 | 81.8 | 98.9 | 76.2 | 0.579 |
| 3.0 | 76.3 | 79.7 | 98.8 | 76.4 | 0.561 |
| 4.0 | 81.9 | 81.5 | 98.4 | 75.6 | 0.565 |

## 7. Top permutation-importance features (validation data)

**2-week model:** Humidity (now) (20.9%); Cumulative rainfall (4 wk) (16.3%); Rainfall 1 wk ago (13.3%); Waterlogging risk (11.1%); Cumulative rainfall (2 wk) (10.3%)

**3-week model:** Cumulative rainfall (2 wk) (26.5%); Season (cos) (22.3%); Rainfall this week (19.8%); Waterlogging risk (10.7%); Population density (3.8%)

**4-week model:** Season (cos) (42.8%); Waterlogging risk (14.7%); Cumulative rainfall (4 wk) (12.0%); Rainfall this week (9.6%); Season (sin) (9.4%)


Importance describes what the model uses; it does not establish causation.

## 8. Limitations

- Test-year scores are lower than validation scores: expect real-world accuracy closer to the test numbers.

- Predictions are conditional averages and under-predict sharp peaks; accuracy declines as the horizon grows.

- Risk tiers are relative to historical experience, not clinical thresholds.

- Intervals are empirical (validation residuals) and not guaranteed to cover.

- A single 2016-2025 dataset with no mobility, serotype, intervention or geographic data.
