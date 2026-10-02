"""Map predicted RUL to health status bands and rank engines for maintenance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path("config.yaml")
STATUS_ORDER = ["Critical", "Warning", "Watch", "Healthy"]  # most to least urgent


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())["health"]
    t = cfg["thresholds"]
    if not 0 < t["critical"] < t["warning"] < t["watch"]:
        raise ValueError("thresholds must satisfy 0 < critical < warning < watch")
    unknown = set(cfg["maintenance_statuses"]) - set(STATUS_ORDER)
    if unknown:
        raise ValueError(f"unknown maintenance statuses: {sorted(unknown)}")
    return cfg


def classify(rul, cfg: dict | None = None) -> np.ndarray:
    """Vectorised RUL -> status label."""
    t = (cfg or load_config())["thresholds"]
    rul = np.asarray(rul, dtype=float)
    return np.select(
        [rul < t["critical"], rul < t["warning"], rul < t["watch"]],
        ["Critical", "Warning", "Watch"],
        default="Healthy",
    )


def add_status(df: pd.DataFrame, rul_col: str = "predicted_rul", cfg: dict | None = None):
    out = df.copy()
    out["health_status"] = classify(out[rul_col], cfg)
    return out


def engines_needing_maintenance(
    df: pd.DataFrame, rul_col: str = "predicted_rul", cfg: dict | None = None
) -> pd.DataFrame:
    """Latest reading per engine, filtered to maintenance statuses, lowest RUL first."""
    cfg = cfg or load_config()
    latest = df.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
    latest = add_status(latest, rul_col, cfg)
    due = latest[latest["health_status"].isin(cfg["maintenance_statuses"])]
    return due.sort_values([rul_col, "engine_id"]).reset_index(drop=True)
