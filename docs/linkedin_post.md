I built an end-to-end predictive maintenance pipeline on NASA's turbofan engine dataset.

The goal: estimate how many flight cycles each engine has left (Remaining Useful Life) and turn that into a ranked maintenance list.

What's in it:
- Leak-safe feature engineering (rolling stats and per-engine baselines, unit-tested so nothing crosses engine boundaries)
- LightGBM baseline vs. an LSTM on 30-cycle windows
- Health bands (Healthy / Watch / Warning / Critical) driven by a config file
- Batch scoring into DuckDB and a Streamlit dashboard for fleet overview and engine drill-down
- CI, pre-commit and Docker

The result I found most interesting: the LSTM did not beat gradient boosting. RMSE was effectively tied (15.0 vs 15.2), and LightGBM was better on the NASA score, which punishes over-estimating remaining life. With 100 training engines, hand-built history features were worth about as much as a sequence model's learned memory.

Caveats I wrote down in the README: single operating condition, model comparison done on the test set, no prediction intervals yet.

Code: <repo link>

#MachineLearning #PredictiveMaintenance #DataScience #Aviation
