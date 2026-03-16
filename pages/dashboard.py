import json

import pandas as pd
import streamlit as st

from db import fetch_history
from ui_styles import inject_global_styles, render_hero, render_kpi, open_card, close_card

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
inject_global_styles()

render_hero(
    "Dashboard",
    "A quick overview of inspection activity and simple KPI metrics."
)

rows = fetch_history()

if not rows:
    st.info("No data yet. Run at least one analysis first.")
    st.stop()

df = pd.DataFrame(rows)
df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")

total_inspections = len(df)
anomalous_count = int((df["verdict"] == "ANOMALOUS").sum())
normal_count = int((df["verdict"] == "NORMAL").sum())
manual_count = int((df["verdict"] == "MANUAL_INSPECTION").sum())
unsupported_count = int((df["verdict"] == "UNSUPPORTED_FORMAT").sum())

valid_anomaly_scores = df["anomaly_score"].dropna()
avg_score = valid_anomaly_scores.mean() if not valid_anomaly_scores.empty else 0.0

known_classes = df["class_label"].dropna()
if not known_classes.empty:
    most_common_class = known_classes.mode().iloc[0]
else:
    most_common_class = "N/A"

k1, k2, k3, k4, k5, k6 = st.columns(6, gap="medium")

with k1:
    render_kpi("Total inspections", str(total_inspections), "All records")

with k2:
    render_kpi("Anomalous", str(anomalous_count), "Detected anomalies")

with k3:
    render_kpi("Normal", str(normal_count), "Accepted images")

with k4:
    render_kpi("Manual inspection", str(manual_count), "Low-confidence cases")

with k5:
    render_kpi("Unsupported", str(unsupported_count), "Unknown format/type")

with k6:
    render_kpi("Avg anomaly score", f"{avg_score:.2f}%", "Valid anomaly runs only")

st.markdown("<br>", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    open_card("Verdict distribution", "Breakdown of final inspection statuses.")
    verdict_counts = df["verdict"].value_counts()
    st.bar_chart(verdict_counts)
    close_card()

with right_col:
    open_card("Most common class", "Most frequently predicted object class.")
    st.metric("Class", most_common_class)
    close_card()

st.markdown("<br>", unsafe_allow_html=True)

lower_left, lower_right = st.columns([1, 1], gap="large")

with lower_left:
    open_card("Class distribution", "How often each object class appears.")
    class_counts = df["class_label"].fillna("N/A").value_counts()
    st.bar_chart(class_counts)
    close_card()

with lower_right:
    open_card("Latest 5 inspections", "Most recent stored records.")
    latest_df = df[["id", "image_name", "class_label", "verdict", "anomaly_score", "created_at"]].copy()
    latest_df["anomaly_score"] = latest_df["anomaly_score"].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    )
    latest_df = latest_df.head(5)
    st.dataframe(latest_df, width="stretch")
    close_card()