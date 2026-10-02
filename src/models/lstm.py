"""LSTM RUL model on 30-cycle sensor windows; compares against the baseline."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.load import PROCESSED_DIR, clean
from src.features.engineer import sensor_columns
from src.features.labels import RUL_CAP
from src.models.baseline import MODELS_DIR, RESULTS_DIR, SEED, split_by_engine
from src.models.metrics import nasa_score, rmse
from src.models.sequences import WINDOW, make_windows

DEVICE = torch.device("cpu")  # small data; CPU is fast and deterministic


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)  # RUL / RUL_CAP


def predict(model: nn.Module, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(torch.from_numpy(X[i : i + batch]).to(DEVICE)).cpu().numpy())
    return (np.concatenate(outs) * RUL_CAP).clip(0, RUL_CAP)


def train_model(Xtr, ytr, Xva, yva, epochs=60, patience=8, lr=1e-3, batch=256):
    torch.manual_seed(SEED)
    model = LSTMRegressor(Xtr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr / RUL_CAP)
    best, best_state, bad = float("inf"), None, 0
    gen = torch.Generator().manual_seed(SEED)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b].to(DEVICE)), yt[b].to(DEVICE))
            loss.backward()
            opt.step()
        val = rmse(yva, predict(model, Xva))
        print(f"epoch {epoch + 1:02d}  val_rmse={val:.3f}")
        if val < best - 1e-3:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def main(subset: str = "FD001") -> dict:
    train = pd.read_parquet(PROCESSED_DIR / f"features_train_{subset}.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"features_test_{subset}.parquet")
    cols = sensor_columns(train)  # already min-max normalized by the feature pipeline

    tr, va = split_by_engine(train)  # same split as the baseline
    Xtr, ytr, _ = make_windows(tr, cols, WINDOW)
    Xva, yva, _ = make_windows(va, cols, WINDOW)
    model, best_val = train_model(Xtr, ytr, Xva, yva)

    Xte, _, keys = make_windows(test, cols, WINDOW, target=None)
    last = keys.groupby("engine_id")["cycle"].transform("max") == keys["cycle"]
    _, _, rul_true, _ = clean(subset)
    y_true = rul_true["rul"].reindex(keys[last]["engine_id"]).to_numpy()
    y_pred = predict(model, Xte[last.to_numpy()])

    results = {
        "model": "lstm",
        "subset": subset,
        "window": WINDOW,
        "n_features": len(cols),
        "val_rmse_all_cycles": best_val,
        "test_rmse": rmse(y_true, y_pred),
        "test_nasa_score": nasa_score(y_true, y_pred),
    }
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "features": cols, "window": WINDOW,
         "rul_cap": RUL_CAP, "n_features": len(cols)},
        MODELS_DIR / f"lstm_{subset}.pt",
    )
    (RESULTS_DIR / f"lstm_{subset}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FD001")
