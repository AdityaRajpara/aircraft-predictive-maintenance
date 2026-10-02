"""Data helpers for the dashboard (no Streamlit imports, so they are testable)."""
from __future__ import annotations

import pandas as pd

from src.health.status import STATUS_ORDER, engines_needing_maintenance

STATUS_COLORS = {
    "Healthy": "#2e9e5b",
    "Watch": "#e0b400",
    "Warning": "#ed7d2b",
    "Critical": "#d64545",
}


def latest_per_engine(preds: pd.DataFrame) -> pd.DataFrame:
    return preds.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)


def fleet_counts(preds: pd.DataFrame) -> pd.DataFrame:
    """Engines per health band (latest cycle), in urgency order, zeros included."""
    counts = latest_per_engine(preds)["health_status"].value_counts()
    return (
        counts.reindex(STATUS_ORDER, fill_value=0)
        .rename_axis("health_status")
        .reset_index(name="engines")
    )


def maintenance_table(preds: pd.DataFrame) -> pd.DataFrame:
    cols = ["engine_id", "cycle", "predicted_rul", "health_status"]
    out = engines_needing_maintenance(preds)[cols].copy()
    out["predicted_rul"] = out["predicted_rul"].round(1)
    return out.rename(columns={"cycle": "current_cycle"})
