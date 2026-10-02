"""Batch scoring: score every engine/cycle with the selected model into DuckDB."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.data.load import PROCESSED_DIR
from src.features.labels import RUL_CAP
from src.health.status import add_status, load_config
from src.storage.duckdb_store import DB_PATH, write_predictions

BEST_MODEL_FILE = Path("models/best_model.json")


def model_version(name: str, subset: str, artifact: Path) -> str:
    digest = hashlib.sha1(artifact.read_bytes()).hexdigest()[:8]
    return f"{name}-{subset}-{digest}"


def score(df: pd.DataFrame, model, cols: list[str], version: str, cfg: dict | None = None):
    """Return the analytics table for a feature frame."""
    out = df[["engine_id", "cycle"]].copy()
    out["predicted_rul"] = model.predict(df[cols]).clip(0, RUL_CAP)
    out = add_status(out, cfg=cfg or load_config())
    out["model_version"] = version
    out["scored_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    out["true_rul"] = df["rul"].to_numpy() if "rul" in df.columns else float("nan")
    return out


def main(subset: str = "FD001", db_path: Path = DB_PATH) -> pd.DataFrame:
    best = json.loads(BEST_MODEL_FILE.read_text())
    if best["model"] != "lightgbm":
        raise NotImplementedError("batch scoring currently supports the LightGBM model")
    artifact = Path(best["artifact"])
    bundle = joblib.load(artifact)
    test = pd.read_parquet(PROCESSED_DIR / f"features_test_{subset}.parquet")

    version = model_version(best["model"], subset, artifact)
    scored = score(test, bundle["model"], bundle["features"], version)
    n = write_predictions(scored, db_path)
    print(f"scored {n} rows ({scored['engine_id'].nunique()} engines) with {version} -> {db_path}")
    return scored


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FD001")
