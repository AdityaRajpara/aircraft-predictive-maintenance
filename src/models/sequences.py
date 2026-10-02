"""Sliding-window sequences for sequence models (never cross engine boundaries)."""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 30


def make_windows(
    df: pd.DataFrame, cols: list[str], window: int = WINDOW, target: str | None = "rul"
):
    """One window per row: the last `window` cycles of the same engine.

    Early cycles are left-padded by repeating the engine's first row, so every
    row (cycle) gets a prediction. Returns X (n, window, n_features), y (n,) or
    None, and the matching (engine_id, cycle) index frame.
    """
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    xs, ys, keys = [], [], []
    for _, g in df.groupby("engine_id", sort=False):
        vals = g[cols].to_numpy(dtype=np.float32)
        pad = np.repeat(vals[:1], window - 1, axis=0)
        padded = np.concatenate([pad, vals], axis=0)
        idx = np.arange(len(vals))[:, None] + np.arange(window)[None, :]
        xs.append(padded[idx])
        if target is not None:
            ys.append(g[target].to_numpy(dtype=np.float32))
        keys.append(g[["engine_id", "cycle"]])
    X = np.concatenate(xs)
    y = np.concatenate(ys) if target is not None else None
    return X, y, pd.concat(keys, ignore_index=True)
