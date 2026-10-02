"""RUL labels (piece-wise linear, capped)."""
from __future__ import annotations

import pandas as pd

RUL_CAP = 125


def add_rul_train(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """RUL = (last cycle of the engine - current cycle), capped at `cap`."""
    out = train.copy()
    last = out.groupby("engine_id")["cycle"].transform("max")
    out["rul"] = (last - out["cycle"]).clip(upper=cap)
    return out


def add_rul_test(test: pd.DataFrame, rul_true: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Per-row test RUL from the true RUL after each engine's last cycle.

    `rul_true` is indexed by engine_id with column `rul`.
    """
    out = test.copy()
    last = out.groupby("engine_id")["cycle"].transform("max")
    final = out["engine_id"].map(rul_true["rul"])
    out["rul"] = (final + (last - out["cycle"])).clip(upper=cap)
    return out
