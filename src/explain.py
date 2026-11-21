"""Prediction-level explanations using Shapley values (the idea behind SHAP), implemented with NumPy.

Why not the `shap` package?  It does not natively support scikit-learn's HistGradientBoostingRegressor
(one of our two model families), and a ~60-line model-agnostic version is easy for students to explain.

What is computed for ONE ward-week x:
  * Background = a fixed random sample of TRAINING-period ward-weeks ("typical" conditions).
  * baseline  = average model prediction over the background.
  * value of a coalition S of features = average prediction when the features in S take x's values and all
    other features take values from background rows (interventional Shapley value).
  * Shapley value of a feature/group = its average marginal effect over all orderings.
  * Efficiency property (checked in tests):  baseline + sum(contributions) == prediction for x.

Two levels:
  * GROUP level (Rainfall, Temperature, Humidity, Recent cases, Vulnerability, Season): EXACT, 2^6 = 64 coalitions.
  * FEATURE level (23 features): Monte-Carlo over random feature orderings (seeded, so repeatable).

Interpretation limits (also shown in the UI): contributions explain what the MODEL does, not what causes dengue;
the background mixes features independently, so unrealistic combinations are sampled; correlated features share credit.
"""
from __future__ import annotations
from itertools import combinations
from math import factorial
import numpy as np
import pandas as pd
from . import config as C


def make_background(feat_df: pd.DataFrame, n: int = 100, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Random sample of TRAINING-period ward-weeks (complete features) used as the 'typical conditions' reference."""
    ref = feat_df[feat_df["date"] <= pd.Timestamp(C.TRAIN_END)].dropna(subset=C.FEATURE_COLS)
    if ref.empty:
        raise ValueError("No training-period rows available for the explanation background.")
    return ref.sample(n=min(n, len(ref)), random_state=seed)[C.FEATURE_COLS].reset_index(drop=True)


def _f(model, arr: np.ndarray) -> np.ndarray:
    """Model output exactly as the dashboard shows it (never negative)."""
    return np.clip(model.predict(pd.DataFrame(arr, columns=C.FEATURE_COLS)), 0, None)


def explain_groups(model, x: np.ndarray, background: np.ndarray, groups=None) -> dict:
    """Exact Shapley values for feature groups. Returns {group_name: contribution (cases/week)}."""
    groups = groups or C.FEATURE_GROUPS
    names = list(groups)
    idx = [[C.FEATURE_COLS.index(f) for f in groups[g]] for g in names]
    G, nb = len(names), len(background)
    masks = list(range(2 ** G))
    stacked = np.empty((len(masks) * nb, len(C.FEATURE_COLS)))
    for m in masks:
        h = background.copy()
        for gi in range(G):
            if m >> gi & 1:
                h[:, idx[gi]] = x[idx[gi]]
        stacked[m * nb:(m + 1) * nb] = h
    v = _f(model, stacked).reshape(len(masks), nb).mean(axis=1)   # value of every coalition
    phi = np.zeros(G)
    for gi in range(G):
        others = [g for g in range(G) if g != gi]
        for k in range(G):
            w = factorial(k) * factorial(G - k - 1) / factorial(G)
            for S in combinations(others, k):
                mask = sum(1 << g for g in S)
                phi[gi] += w * (v[mask | (1 << gi)] - v[mask])
    return {n: float(p) for n, p in zip(names, phi)}


def explain_features(model, x: np.ndarray, background: np.ndarray, n_perm: int = 30, seed: int = C.RANDOM_STATE) -> np.ndarray:
    """Monte-Carlo Shapley values per feature (random orderings x all background rows)."""
    rng = np.random.RandomState(seed)
    nf, nb = len(x), len(background)
    phi = np.zeros(nf)
    for _ in range(n_perm):
        perm = rng.permutation(nf)
        stacked = np.empty(((nf + 1) * nb, nf))
        for k in range(nf + 1):
            h = background.copy()
            if k:
                h[:, perm[:k]] = x[perm[:k]]
            stacked[k * nb:(k + 1) * nb] = h
        vals = _f(model, stacked).reshape(nf + 1, nb).mean(axis=1)
        phi[perm] += np.diff(vals)            # marginal effect of the k-th feature added
    return phi / n_perm


def explain_instance(model, feature_row: pd.Series, background: pd.DataFrame, n_perm: int = 30) -> dict:
    """Full explanation for one ward-week. feature_row must contain all FEATURE_COLS."""
    x = feature_row[C.FEATURE_COLS].to_numpy(dtype=float)
    if np.isnan(x).any():
        raise ValueError("Cannot explain a ward-week with missing feature values.")
    B = background[C.FEATURE_COLS].to_numpy(dtype=float)
    base = float(_f(model, B).mean())
    pred = float(_f(model, x.reshape(1, -1))[0])
    groups = explain_groups(model, x, B)
    phi = explain_features(model, x, B, n_perm=n_perm)
    feats = pd.DataFrame({
        "feature": C.FEATURE_COLS, "label": [C.FEATURE_LABELS[f] for f in C.FEATURE_COLS],
        "value": x, "contribution": phi,
        "group": [next(g for g, fs in C.FEATURE_GROUPS.items() if f in fs) for f in C.FEATURE_COLS],
    })
    feats["abs"] = feats["contribution"].abs()
    feats = feats.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    return {
        "baseline": base, "prediction": pred, "groups": groups, "features": feats,
        "group_additivity_error": abs(base + sum(groups.values()) - pred),
        "feature_additivity_error": abs(base + float(phi.sum()) - pred),
    }


def global_group_importance(model, sample: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Mean |group Shapley value| over a sample of ward-weeks (cases/week). Descriptive of the model only."""
    B = background[C.FEATURE_COLS].to_numpy(dtype=float)
    rows = [explain_groups(model, r, B) for r in sample[C.FEATURE_COLS].to_numpy(dtype=float)]
    df = pd.DataFrame(rows)
    out = df.abs().mean().rename("mean_abs_contribution").reset_index().rename(columns={"index": "group"})
    out["share_pct"] = 100 * out["mean_abs_contribution"] / out["mean_abs_contribution"].sum()
    return out.sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)


def describe(expl: dict, horizon: int, top_n: int = 3) -> list[str]:
    """Short plain-language statements generated by rules from the numbers (no LLM involved)."""
    base, pred = expl["baseline"], expl["prediction"]
    direction = "above" if pred > base else "below"
    lines = [f"The {horizon}-week model predicts {pred:.1f} cases/week, {abs(pred - base):.1f} {direction} the "
             f"{base:.1f} cases/week it predicts for a typical ward-week in the training years."]
    g = expl["groups"]
    weather = sum(g[k] for k in C.WEATHER_GROUPS if k in g)
    lines.append(f"Weather (rainfall + temperature + humidity) moves the prediction by {weather:+.1f}; rainfall alone by {g['Rainfall']:+.1f}.")
    lines.append(f"Ward vulnerability (density, waterlogging, sanitation) contributes {g['Ward vulnerability']:+.1f}; "
                 f"recent dengue cases {g['Recent dengue cases']:+.1f}; time of year {g['Season (calendar)']:+.1f}.")
    top = expl["features"].head(top_n)
    lines.append("Largest individual factors: " + "; ".join(f"{r.label} ({r.contribution:+.1f})" for r in top.itertuples()) + ".")
    lines.append("These describe how the model arrives at its number, not what causes dengue.")
    return lines
