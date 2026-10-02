import pandas as pd
import pytest

from src.data.load import COLUMNS, clean, constant_columns, validate


def _frame(rows):
    return pd.DataFrame(rows, columns=["engine_id", "cycle"])


def test_validate_ok():
    validate(_frame([(1, 1), (1, 2), (2, 1)]))


def test_validate_duplicates():
    with pytest.raises(ValueError):
        validate(_frame([(1, 1), (1, 1)]))


def test_validate_unsorted():
    with pytest.raises(ValueError):
        validate(_frame([(1, 2), (1, 1)]))


def test_validate_nulls():
    df = pd.DataFrame({"engine_id": [1], "cycle": [None]})
    with pytest.raises(ValueError):
        validate(df)


def test_constant_columns():
    df = pd.DataFrame({"a": [1, 1], "b": [1, 2]})
    assert constant_columns(df, ["a", "b"]) == ["a"]


def test_clean_fd001(tmp_path):
    def write(name, n_engines, cycles):
        lines = []
        for e in range(1, n_engines + 1):
            for c in range(1, cycles + 1):
                vals = [e, c] + [0.1] * 3 + [float(c) + i for i in range(21)]
                lines.append(" ".join(map(str, vals)))
        (tmp_path / name).write_text("\n".join(lines))

    write("train_FD001.txt", 2, 5)
    write("test_FD001.txt", 2, 3)
    (tmp_path / "RUL_FD001.txt").write_text("10\n20\n")
    train, test, rul, dropped = clean("FD001", tmp_path)
    assert set(dropped) >= {"setting_1", "setting_2", "setting_3"}
    assert len(COLUMNS) == 26
    assert list(rul["rul"]) == [10, 20]
    assert list(train.columns) == list(test.columns)
