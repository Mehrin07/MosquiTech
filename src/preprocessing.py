"""Cleaning: sort, de-duplicate, and guarantee a regular weekly grid per ward.

Regular spacing matters: a lag of "k rows" only equals "k weeks" when no weeks are missing.
For the supplied dataset no weeks are missing, so this step changes nothing, but it keeps
the pipeline correct if the officers' data later has gaps.
"""
from __future__ import annotations
import pandas as pd
from .config import STATIC_COLS, WEATHER_COLS


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date", "ward_id"]).copy()
    df = df.drop_duplicates(subset=["ward_id", "date"], keep="last")
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)

    parts = []
    for ward, g in df.groupby("ward_id", sort=True):
        grid = pd.date_range(g["date"].min(), g["date"].max(), freq="7D")
        g = g.set_index("date").reindex(grid)
        g.index.name = "date"
        g["ward_id"] = ward
        g["city"] = g["city"].ffill().bfill()
        g[STATIC_COLS] = g[STATIC_COLS].ffill().bfill()
        # short weather gaps are interpolated; dengue_cases is NEVER imputed (stays NaN)
        g[WEATHER_COLS] = g[WEATHER_COLS].interpolate(limit=2, limit_direction="both")
        parts.append(g.reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["rainfall_mm"] = out["rainfall_mm"].clip(lower=0)
    return out
