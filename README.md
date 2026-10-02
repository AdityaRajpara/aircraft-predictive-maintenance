# Aircraft Predictive Maintenance

Predict the Remaining Useful Life (RUL) of turbofan engines from sensor data
(NASA C-MAPSS), classify engine health, and surface a prioritized maintenance
list in a dashboard.

## Problem statement

Unplanned engine failures are costly and unsafe; fixed-interval maintenance
wastes useful life. Given multivariate sensor readings per flight cycle, we
estimate each engine's RUL and rank the fleet by maintenance urgency.

## Pipeline

```
Raw C-MAPSS data -> Cleaning -> Feature engineering -> RUL model (GBM / LSTM)
   -> Health status bands -> Batch scoring (DuckDB/Parquet) -> Streamlit dashboard
```

## Repository layout

```
data/{raw,processed}   datasets (not committed)
notebooks/             EDA and experiments
src/{data,features,models,health,storage}
dashboard/             Streamlit app
tests/                 unit tests
models/                trained models (not committed)
```

## Data

Download the NASA C-MAPSS dataset (FD001) and place `train_FD001.txt`,
`test_FD001.txt`, and `RUL_FD001.txt` in `data/raw/`. Data files are git-ignored.

## Setup

```bash
make setup
source .venv/bin/activate
make test
```

## Results

Evaluated on the last observed cycle of each of the 100 FD001 test engines
(true RUL from `RUL_FD001.txt`). Validation is a 20% hold-out split by engine.

| Model | Test RMSE | Test NASA score | Notes |
|-------|-----------|-----------------|-------|
| LightGBM baseline (77 features, RUL cap 125) | 15.01 | 321.9 | `make baseline` |
| LSTM (2x64, 30-cycle windows, 15 sensors) | 15.18 | 431.4 | `make lstm` |

Validation RMSE over all cycles is 11.03. Validation RMSE on last cycles is
not comparable: training engines run to failure, so their last cycle (RUL 0)
is far easier than the truncated test engines.

## Model comparison

Both models use the same engine-level train/validation split and are scored on
the last cycle of the 100 test engines.

- **RMSE is effectively tied** (15.01 vs 15.18); with only 100 test engines the
  gap is within noise.
- **NASA score favours LightGBM** (322 vs 431). The score penalises late
  predictions (over-estimating remaining life) exponentially, so the LSTM makes
  more of those costly errors on some engines.
- The LSTM sees only the raw normalised sensors; LightGBM gets engineered
  rolling-mean/std/slope and baseline-delta features. That hand-built history
  appears to be worth about as much as the sequence model's learned memory on
  a dataset this small (FD001, 100 training engines).
- **Selected model: LightGBM** (see `models/best_model.json`). Both trained
  models are kept in `models/`. The LSTM is not tuned; there is likely headroom
  (longer windows, feature inputs, ensembling).

## Health status and maintenance

`src/health/status.py` maps predicted RUL to bands using thresholds in
`config.yaml` (Critical < 20, Warning < 50, Watch < 80, otherwise Healthy).
`engines_needing_maintenance` returns each engine's latest reading, filtered to
Critical/Warning and ranked by lowest RUL. On the FD001 test set the baseline
puts 14 engines in Critical, 18 in Warning, 17 in Watch and 51 in Healthy.

## Batch scoring

```bash
make predict   # features -> baseline model -> scored table in DuckDB
```

`src/predict.py` scores every cycle of every test engine with the model named in
`models/best_model.json` and writes `data/processed/predictions.duckdb`
(table `predictions`: `engine_id, cycle, predicted_rul, health_status,
model_version, scored_at`, plus `true_rul` for the dashboard). Re-running
replaces the table. `model_version` includes a hash of the model file.

## Status

Phases 0-7 complete: data cleaning, EDA, feature pipeline, gradient boosting baseline, LSTM comparison, health bands, DuckDB batch scoring.

## License

MIT
