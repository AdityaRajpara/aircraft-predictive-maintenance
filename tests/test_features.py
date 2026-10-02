import numpy as np
import pandas as pd

from src.features.engineer import (
    add_baseline_delta,
    add_rolling,
    build_features,
    fit_scaler,
    normalize,
)
from src.features.labels import add_rul_test, add_rul_train


def make(values_by_engine):
    rows = []
    for e, vals in values_by_engine.items():
        for c, v in enumerate(vals, start=1):
            rows.append((e, c, v))
    return pd.DataFrame(rows, columns=["engine_id", "cycle", "s_2"])


def test_rul_train_capped():
    df = make({1: range(200)})
    out = add_rul_train(df, cap=125)
    assert out["rul"].iloc[-1] == 0
    assert out["rul"].max() == 125
    assert out["rul"].iloc[0] == 125  # 199 capped


def test_rul_test_uses_true_rul():
    df = make({1: range(5), 2: range(3)})
    rul = pd.DataFrame({"rul": [10, 20]}, index=pd.Index([1, 2], name="engine_id"))
    out = add_rul_test(df, rul)
    assert out[out.engine_id == 1]["rul"].tolist() == [14, 13, 12, 11, 10]
    assert out[out.engine_id == 2]["rul"].iloc[-1] == 20


def test_rolling_does_not_cross_engine_boundaries():
    df = make({1: [1.0, 2.0, 3.0, 4.0], 2: [100.0, 101.0, 102.0, 103.0]})
    out = add_rolling(df, ["s_2"], window=3)
    first2 = out[(out.engine_id == 2) & (out.cycle == 1)].iloc[0]
    # first row of engine 2 sees only itself
    assert first2["s_2_mean3"] == 100.0
    assert first2["s_2_std3"] == 0.0
    assert first2["s_2_slope3"] == 0.0
    # full window inside engine 2 only
    row = out[(out.engine_id == 2) & (out.cycle == 3)].iloc[0]
    assert row["s_2_mean3"] == 101.0
    assert np.isclose(row["s_2_slope3"], 1.0)


def test_engine_features_independent_of_other_engines():
    a = make({1: [1.0, 5.0, 2.0, 9.0], 2: [3.0, 4.0, 6.0, 8.0]})
    b = make({1: [900.0, -50.0, 7.0, 1.0], 2: [3.0, 4.0, 6.0, 8.0]})
    fa = add_baseline_delta(add_rolling(a, ["s_2"], 3), ["s_2"], 2)
    fb = add_baseline_delta(add_rolling(b, ["s_2"], 3), ["s_2"], 2)
    e2a = fa[fa.engine_id == 2].reset_index(drop=True)
    e2b = fb[fb.engine_id == 2].reset_index(drop=True)
    pd.testing.assert_frame_equal(e2a, e2b)


def test_rolling_is_causal():
    base = make({1: [1.0, 2.0, 3.0, 4.0, 5.0]})
    changed = base.copy()
    changed.loc[4, "s_2"] = 999.0  # change the last cycle only
    fa = add_rolling(base, ["s_2"], 3)
    fb = add_rolling(changed, ["s_2"], 3)
    pd.testing.assert_frame_equal(fa.iloc[:4], fb.iloc[:4])


def test_baseline_delta_zero_on_baseline_mean():
    df = make({1: [2.0, 4.0, 10.0]})
    out = add_baseline_delta(df, ["s_2"], n=2)  # baseline mean = 3
    assert out["s_2_delta"].tolist() == [-1.0, 1.0, 7.0]


def test_normalize_uses_train_stats():
    train = make({1: [0.0, 10.0]})
    scaler = fit_scaler(train, ["s_2"])
    test = make({1: [5.0, 20.0]})
    assert normalize(test, scaler)["s_2"].tolist() == [0.5, 2.0]


def test_build_features_columns():
    df = add_rul_train(make({1: list(range(30)), 2: list(range(25))}))
    scaler = fit_scaler(df, ["s_2"])
    out = build_features(df, scaler, window=5, n_base=5)
    for suffix in ["_mean5", "_std5", "_slope5", "_delta"]:
        assert f"s_2{suffix}" in out.columns
    assert not out.isna().any().any()
