import pandas as pd

from dashboard.data import fleet_counts, latest_per_engine, maintenance_table


def preds():
    return pd.DataFrame({
        "engine_id": [1, 1, 2, 3],
        "cycle": [1, 2, 1, 1],
        "predicted_rul": [10.0, 100.0, 30.0, 5.0],
        "health_status": ["Critical", "Healthy", "Warning", "Critical"],
    })


def test_latest_per_engine():
    assert latest_per_engine(preds())["cycle"].tolist() == [2, 1, 1]


def test_fleet_counts_orders_and_zero_fills():
    c = fleet_counts(preds())
    assert c["health_status"].tolist() == ["Critical", "Warning", "Watch", "Healthy"]
    assert c["engines"].tolist() == [1, 1, 0, 1]


def test_maintenance_table_ranked():
    t = maintenance_table(preds())
    assert t["engine_id"].tolist() == [3, 2]
