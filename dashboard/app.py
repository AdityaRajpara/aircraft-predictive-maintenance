"""Streamlit dashboard: fleet overview, maintenance queue, engine drill-down.

Run:  streamlit run dashboard/app.py   (after `make predict`)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.data import (  # noqa: E402
    STATUS_COLORS,
    fleet_counts,
    latest_per_engine,
    maintenance_table,
)
from src.data.load import PROCESSED_DIR  # noqa: E402
from src.health.status import load_config  # noqa: E402
from src.storage.duckdb_store import DB_PATH, read_predictions  # noqa: E402

st.set_page_config(page_title="Fleet Health", page_icon="✈️", layout="wide")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return read_predictions(DB_PATH)


@st.cache_data
def load_sensors(subset: str = "FD001") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"test_{subset}.parquet")


if not Path(DB_PATH).exists():
    st.error(f"No predictions found at `{DB_PATH}`. Run `make predict` first.")
    st.stop()

preds = load_predictions()
cfg = load_config()
latest = latest_per_engine(preds)

st.title("✈️ Engine Fleet Health")
st.caption(
    f"{preds['engine_id'].nunique()} engines · model `{preds['model_version'].iloc[0]}` · "
    f"scored {preds['scored_at'].max():%Y-%m-%d %H:%M} UTC"
)

tab_fleet, tab_maint, tab_engine = st.tabs(
    ["Fleet overview", "Maintenance queue", "Engine drill-down"]
)

with tab_fleet:
    counts = fleet_counts(preds)
    cols = st.columns(4)
    for col, row in zip(cols, counts.itertuples()):
        col.metric(row.health_status, int(row.engines))
    fig = go.Figure(
        go.Bar(
            x=counts["health_status"],
            y=counts["engines"],
            marker_color=[STATUS_COLORS[s] for s in counts["health_status"]],
            text=counts["engines"],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="Engines", xaxis_title=None, height=360, margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    t = cfg["thresholds"]
    st.caption(
        f"Bands on predicted RUL: Critical < {t['critical']}, Warning < {t['warning']}, "
        f"Watch < {t['watch']}, otherwise Healthy (edit `config.yaml`)."
    )

with tab_maint:
    table = maintenance_table(preds)
    st.subheader(f"{len(table)} engines need maintenance")
    st.caption("Latest prediction per engine, lowest remaining useful life first.")
    st.dataframe(
        table.style.format({"predicted_rul": "{:.1f}"}).apply(
            lambda r: [f"color: {STATUS_COLORS[r['health_status']]}; font-weight: 600"
                       if c == "health_status" else "" for c in r.index],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_engine:
    order = latest.sort_values("predicted_rul")["engine_id"].tolist()
    engine = st.selectbox("Engine (most urgent first)", order)
    ep = preds[preds["engine_id"] == engine].sort_values("cycle")
    now = ep.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Health status", now["health_status"])
    c2.metric("Predicted RUL", f"{now['predicted_rul']:.1f} cycles")
    c3.metric("Current cycle", int(now["cycle"]))

    rul = go.Figure()
    rul.add_trace(go.Scatter(x=ep["cycle"], y=ep["predicted_rul"], name="Predicted RUL",
                             line=dict(color="#2a6fdb", width=3)))
    rul.add_trace(go.Scatter(x=ep["cycle"], y=ep["true_rul"], name="True RUL (capped at 125)",
                             line=dict(color="#888", dash="dash")))
    for name, y in cfg["thresholds"].items():
        rul.add_hline(y=y, line_dash="dot", line_color="#bbb", annotation_text=name,
                      annotation_position="top left")
    rul.update_layout(title="Remaining useful life", xaxis_title="Cycle",
                      yaxis_title="RUL (cycles)", height=380, margin=dict(t=50, b=20),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(rul, use_container_width=True)

    sensors = load_sensors()
    sensors = sensors[sensors["engine_id"] == engine]
    sensor_cols = [c for c in sensors.columns if c.startswith("s_")]
    default = [c for c in ["s_4", "s_11", "s_12"] if c in sensor_cols]
    chosen = st.multiselect("Sensors", sensor_cols, default=default)
    if chosen:
        fig = go.Figure()
        for c in chosen:
            s = sensors[c]
            fig.add_trace(go.Scatter(x=sensors["cycle"], y=(s - s.mean()) / (s.std() or 1),
                                     name=c))
        fig.update_layout(title="Sensor trends (z-scored within this engine)",
                          xaxis_title="Cycle", yaxis_title="Std. deviations", height=380,
                          margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
