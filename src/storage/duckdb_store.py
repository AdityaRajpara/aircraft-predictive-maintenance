"""DuckDB storage for the scored analytics table."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path("data/processed/predictions.duckdb")
TABLE = "predictions"
COLUMNS = [
    "engine_id", "cycle", "predicted_rul", "health_status",
    "model_version", "scored_at", "true_rul",
]


def write_predictions(df: pd.DataFrame, path: Path | str = DB_PATH) -> int:
    """Replace the predictions table (idempotent re-scoring). Returns row count."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = df[COLUMNS]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.register("incoming", data)
        con.execute(f"CREATE OR REPLACE TABLE {TABLE} AS SELECT * FROM incoming")
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_engine ON {TABLE}(engine_id, cycle)")
    finally:
        con.close()
    return len(data)


def read_predictions(path: Path | str = DB_PATH, query: str | None = None) -> pd.DataFrame:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(query or f"SELECT * FROM {TABLE} ORDER BY engine_id, cycle").df()
    finally:
        con.close()
