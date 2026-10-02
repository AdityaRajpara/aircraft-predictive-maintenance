import pandas as pd
import pytest
import yaml

from src.health.status import (
    add_status,
    classify,
    engines_needing_maintenance,
    load_config,
)

CFG = {
    "thresholds": {"critical": 20, "warning": 50, "watch": 80},
    "maintenance_statuses": ["Critical", "Warning"],
}


def test_band_boundaries():
    got = classify([0, 19.9, 20, 49.9, 50, 79.9, 80, 125], CFG)
    assert got.tolist() == [
        "Critical", "Critical", "Warning", "Warning", "Watch", "Watch", "Healthy", "Healthy",
    ]


def test_ranking_uses_latest_cycle_and_sorts_by_rul():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2, 3, 4],
            "cycle": [1, 2, 1, 2, 1, 1],
            "predicted_rul": [5, 90, 60, 30, 10, 100],  # engine 1 recovered to 90
        }
    )
    out = engines_needing_maintenance(df, cfg=CFG)
    assert out["engine_id"].tolist() == [3, 2]  # 10 (Critical), 30 (Warning)
    assert out["health_status"].tolist() == ["Critical", "Warning"]


def test_add_status_column():
    df = pd.DataFrame({"predicted_rul": [10, 100]})
    assert add_status(df, cfg=CFG)["health_status"].tolist() == ["Critical", "Healthy"]


def test_load_default_config():
    cfg = load_config()
    assert cfg["thresholds"]["critical"] < cfg["thresholds"]["warning"]


def test_invalid_thresholds_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump({"health": {
        "thresholds": {"critical": 50, "warning": 20, "watch": 80},
        "maintenance_statuses": ["Critical"]}}))
    with pytest.raises(ValueError):
        load_config(p)


def test_unknown_status_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump({"health": {
        "thresholds": {"critical": 20, "warning": 50, "watch": 80},
        "maintenance_statuses": ["Bad"]}}))
    with pytest.raises(ValueError):
        load_config(p)


def test_thresholds_are_configurable(tmp_path):
    cfg = {"thresholds": {"critical": 5, "warning": 10, "watch": 15},
           "maintenance_statuses": ["Critical"]}
    assert classify([4, 12, 30], cfg).tolist() == ["Critical", "Watch", "Healthy"]
