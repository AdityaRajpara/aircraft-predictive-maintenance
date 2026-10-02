"""LightGBM baseline: train/validate split by engine, evaluate on test last cycles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.load import PROCESSED_DIR, clean
from src.features.labels import RUL_CAP
from src.models.metrics import nasa_score, rmse

MODELS_DIR = Path("models")
RESULTS_DIR = Path("docs/results")
NON_FEATURES = ["engine_id", "cycle", "rul"]
SEED = 42

PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def split_by_engine(df: pd.DataFrame, val_frac: float = 0.2, seed: int = SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(gss.split(df, groups=df["engine_id"]))
    return df.iloc[tr], df.iloc[va]


def last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)


def predict(model, df: pd.DataFrame, cols: list[str]):
    return model.predict(df[cols]).clip(0, RUL_CAP)


def main(subset: str = "FD001") -> dict:
    train = pd.read_parquet(PROCESSED_DIR / f"features_train_{subset}.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"features_test_{subset}.parquet")
    cols = feature_columns(train)

    tr, va = split_by_engine(train)
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        tr[cols], tr["rul"],
        eval_set=[(va[cols], va["rul"])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # headline numbers: last cycle of each test engine vs. the official RUL file
    _, _, rul_true, _ = clean(subset)
    test_last = last_cycle(test)
    y_true = rul_true["rul"].reindex(test_last["engine_id"]).to_numpy()
    y_pred = predict(model, test_last, cols)

    va_last = last_cycle(va)
    results = {
        "model": "lightgbm",
        "subset": subset,
        "best_iteration": int(model.best_iteration_),
        "n_features": len(cols),
        "val_rmse_all_cycles": rmse(va["rul"], predict(model, va, cols)),
        "val_rmse_last_cycle": rmse(va_last["rul"], predict(model, va_last, cols)),
        "test_rmse": rmse(y_true, y_pred),
        "test_nasa_score": nasa_score(y_true, y_pred),
    }
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": cols}, MODELS_DIR / f"baseline_lgbm_{subset}.pkl")
    (RESULTS_DIR / f"baseline_{subset}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FD001")
