"""Load and clean NASA C-MAPSS data (FD001 by default)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

INDEX_COLS = ["engine_id", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_raw(subset: str = "FD001", raw_dir: Path = RAW_DIR):
    """Return (train, test, rul) for a C-MAPSS subset.

    `rul` is indexed by engine_id (1-based) with column `rul`: the true
    remaining cycles after the last recorded cycle of each test engine.
    """
    train = _read(raw_dir / f"train_{subset}.txt")
    test = _read(raw_dir / f"test_{subset}.txt")
    rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None, names=["rul"])
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="engine_id")
    return train, test, rul


def validate(df: pd.DataFrame, name: str = "df") -> None:
    """Raise if the frame has nulls, duplicate keys, or bad ordering."""
    if df.isna().any().any():
        raise ValueError(f"{name}: contains null values")
    if df.duplicated(INDEX_COLS).any():
        raise ValueError(f"{name}: duplicate (engine_id, cycle) rows")
    ordered = df.sort_values(INDEX_COLS).reset_index(drop=True)
    if not ordered[INDEX_COLS].equals(df[INDEX_COLS].reset_index(drop=True)):
        raise ValueError(f"{name}: rows not sorted by engine_id, cycle")
    # cycles must be contiguous starting at 1 for every engine
    g = df.groupby("engine_id")["cycle"]
    if not ((g.min() == 1) & (g.max() == g.count())).all():
        raise ValueError(f"{name}: cycles are not contiguous from 1")


def constant_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns with a single unique value (no information)."""
    return [c for c in cols if df[c].nunique(dropna=False) <= 1]


def clean(subset: str = "FD001", raw_dir: Path = RAW_DIR):
    """Load, validate and drop constant columns (determined on train)."""
    train, test, rul = load_raw(subset, raw_dir)
    train = train.sort_values(INDEX_COLS).reset_index(drop=True)
    test = test.sort_values(INDEX_COLS).reset_index(drop=True)
    validate(train, "train")
    validate(test, "test")
    if len(rul) != test["engine_id"].nunique():
        raise ValueError("RUL file length does not match number of test engines")

    drop = constant_columns(train, SETTING_COLS + SENSOR_COLS)
    return train.drop(columns=drop), test.drop(columns=drop), rul, drop


def main(subset: str = "FD001") -> None:
    train, test, rul, dropped = clean(subset)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / f"train_{subset}.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / f"test_{subset}.parquet", index=False)
    rul.to_parquet(PROCESSED_DIR / f"rul_{subset}.parquet")
    print(f"{subset}: train {train.shape}, test {test.shape}, rul {rul.shape}")
    print(f"dropped constant columns ({len(dropped)}): {dropped}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "FD001")
