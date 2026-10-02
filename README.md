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

## Status

Phase 0 complete (repo skeleton). See the roadmap for upcoming phases.

## License

MIT
