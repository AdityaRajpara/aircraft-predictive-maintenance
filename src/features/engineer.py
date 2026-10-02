"""Feature engineering: normalization, rolling stats, baseline deltas.

All per-engine operations use groupby so nothing leaks across engines, and all
are causal (a row only sees the current and earlier cycles of its own engine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import SENSOR_COLS

BASELINE_CYCLES = 20
WINDOW = 10


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in SENSOR_COLS if c in df.columns]


def fit_scaler(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-sensor min/max from the training set (rows: min, max)."""
    return train[cols].agg(["min", "max"])


def normalize(df: pd.DataFrame, scaler: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale sensors to [0, 1] using training statistics."""
    out = df.copy()
    for c in scaler.columns:
        lo, hi = scaler.loc["min", c], scaler.loc["max", c]
        out[c] = (out[c] - lo) / (hi - lo) if hi > lo else 0.0
    return out


def _slope(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=float)
    t -= t.mean()
    return float((t * (x - x.mean())).sum() / (t * t).sum())


def add_rolling(df: pd.DataFrame, cols: list[str], window: int = WINDOW) -> pd.DataFrame:
    """Rolling mean / std / slope per engine over the last `window` cycles."""
    g = df.groupby("engine_id", sort=False)
    new = {}
    for c in cols:
        roll = g[c].rolling(window, min_periods=1)
        new[f"{c}_mean{window}"] = roll.mean().reset_index(level=0, drop=True)
        new[f"{c}_std{window}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        new[f"{c}_slope{window}"] = (
            roll.apply(_slope, raw=True).reset_index(level=0, drop=True)
        )
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_baseline_delta(
    df: pd.DataFrame, cols: list[str], n: int = BASELINE_CYCLES
) -> pd.DataFrame:
    """Deviation of each sensor from the engine's own early-life mean."""
    first = df.sort_values(["engine_id", "cycle"]).groupby("engine_id").head(n)
    base = first.groupby("engine_id")[cols].mean()
    out = df.copy()
    for c in cols:
        out[f"{c}_delta"] = out[c] - out["engine_id"].map(base[c])
    return out


def build_features(
    df: pd.DataFrame, scaler: pd.DataFrame, window: int = WINDOW, n_base: int = BASELINE_CYCLES
) -> pd.DataFrame:
    cols = sensor_columns(df)
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    df = normalize(df, scaler)
    df = add_baseline_delta(df, cols, n_base)
    return add_rolling(df, cols, window)
