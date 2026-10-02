import numpy as np
import pandas as pd

from src.models.sequences import make_windows


def frame():
    rows = []
    for e, n in [(1, 4), (2, 3)]:
        for c in range(1, n + 1):
            rows.append((e, c, e * 100 + c, float(n - c)))
    return pd.DataFrame(rows, columns=["engine_id", "cycle", "s_2", "rul"])


def test_shapes_and_alignment():
    X, y, keys = make_windows(frame(), ["s_2"], window=3)
    assert X.shape == (7, 3, 1) and y.shape == (7,) and len(keys) == 7


def test_windows_do_not_cross_engines():
    X, _, keys = make_windows(frame(), ["s_2"], window=3)
    for i, e in enumerate(keys["engine_id"]):
        assert set(np.floor(X[i, :, 0] / 100)) == {e}


def test_left_padding_repeats_first_row_and_last_is_current():
    X, _, _ = make_windows(frame(), ["s_2"], window=3)
    assert X[0, :, 0].tolist() == [101, 101, 101]
    assert X[2, :, 0].tolist() == [101, 102, 103]
    assert X[3, :, 0].tolist() == [102, 103, 104]
    assert X[3, -1, 0] == 104  # last element is the row's own cycle
