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

Validation RMSE over all cycles is 11.03. Validation RMSE on last cycles is
not comparable: training engines run to failure, so their last cycle (RUL 0)
is far easier than the truncated test engines.

## Status

Phases 0-4 complete: data cleaning, EDA, feature pipeline, gradient boosting baseline.

## License

MIT
