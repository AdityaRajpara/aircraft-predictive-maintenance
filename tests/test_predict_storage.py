import numpy as np
import pandas as pd
import pytest

from src.predict import score
from src.storage.duckdb_store import COLUMNS, read_predictions, write_predictions

CFG = {
    "thresholds": {"critical": 20, "warning": 50, "watch": 80},
    "maintenance_statuses": ["Critical", "Warning"],
}


class Dummy:
    def predict(self, X):
        return X["f"].to_numpy() * 100  # allows out-of-range values to test clipping


def frame():
    return pd.DataFrame(
        {"engine_id": [1, 1, 2], "cycle": [1, 2, 1], "f": [0.1, 2.0, 0.6], "rul": [9, 8, 7]}
    )


def test_score_schema_status_and_clipping():
    out = score(frame(), Dummy(), ["f"], "v1", CFG)
    assert set(COLUMNS) <= set(out.columns)
    assert out["predicted_rul"].tolist() == [10.0, 125.0, 60.0]  # clipped at 125
    assert out["health_status"].tolist() == ["Critical", "Healthy", "Watch"]
    assert (out["model_version"] == "v1").all()


def test_roundtrip_and_idempotent(tmp_path):
    db = tmp_path / "p.duckdb"
    out = score(frame(), Dummy(), ["f"], "v1", CFG)
    assert write_predictions(out, db) == 3
    write_predictions(out, db)  # re-run replaces, not appends
    back = read_predictions(db)
    assert len(back) == 3
    assert back["predicted_rul"].tolist() == [10.0, 125.0, 60.0]
    assert np.issubdtype(back["scored_at"].dtype, np.datetime64)


def test_missing_columns_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_predictions(pd.DataFrame({"engine_id": [1]}), tmp_path / "x.duckdb")
