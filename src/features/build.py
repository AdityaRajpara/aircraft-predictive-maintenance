"""One-command feature pipeline: cleaned FD001 -> labeled feature tables."""
from __future__ import annotations

import sys

from src.data.load import PROCESSED_DIR, clean
from src.features.engineer import build_features, fit_scaler, sensor_columns
from src.features.labels import add_rul_test, add_rul_train


def main(subset: str = "FD001") -> None:
    train, test, rul, _ = clean(subset)
    scaler = fit_scaler(train, sensor_columns(train))
    train_f = build_features(add_rul_train(train), scaler)
    test_f = build_features(add_rul_test(test, rul), scaler)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_f.to_parquet(PROCESSED_DIR / f"features_train_{subset}.parquet", index=False)
    test_f.to_parquet(PROCESSED_DIR / f"features_test_{subset}.parquet", index=False)
    print(f"{subset}: train features {train_f.shape}, test features {test_f.shape}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FD001")
