from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from db import fetch_history

from ui_styles import inject_global_styles, render_hero, render_kpi, open_card, close_card

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

inject_global_styles()

render_hero(
    "Dashboard",
    "A quick overview of inspection activity and simple KPI metrics."
)


def format_score(score):
    if score is None:
        return 0.0
    return float(score)


rows = fetch_history()

if not rows:
    st.info("No data yet. Run at least one analysis first.")
    st.stop()

df = pd.DataFrame(rows)
df["anomaly_score"] = df["anomaly_score"].fillna(0.0)

total_inspections = len(df)
anomalous_count = int((df["verdict"] == "ANOMALOUS").sum())
normal_count = int((df["verdict"] == "NORMAL").sum())
anomaly_rate = (anomalous_count / total_inspections) * 100 if total_inspections else 0
avg_score = df["anomaly_score"].mean() if total_inspections else 0

k1, k2, k3, k4 = st.columns(4, gap="medium")

with k1:
    render_kpi("Total inspections", str(total_inspections), "All analyzed images")

with k2:
    render_kpi("Anomalous", str(anomalous_count), "Detected as anomalous")

with k3:
    render_kpi("Normal", str(normal_count), "Detected as normal")

with k4:
    render_kpi("Average score", f"{avg_score:.2f}%", "Across all inspections")

st.markdown("<br>", unsafe_allow_html=True)

left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    open_card("Verdict distribution", "Breakdown of anomalous vs normal results.")
    chart_data = df["verdict"].value_counts()
    st.bar_chart(chart_data)
    close_card()

with right_col:
    open_card("Anomaly rate", "Percentage of records marked as anomalous.")
    st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    close_card()

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Latest 5 inspections")

latest_df = df[["id", "image_name", "verdict", "anomaly_score", "created_at"]].copy()
latest_df["anomaly_score"] = latest_df["anomaly_score"].map(lambda x: f"{x:.2f}%")
latest_df = latest_df.head(5)

st.dataframe(latest_df, use_container_width=True)
