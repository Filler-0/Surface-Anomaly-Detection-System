import altair as alt
import pandas as pd
import streamlit as st

from db import fetch_history
from ui_styles import inject_global_styles, render_hero, render_kpi, open_card, close_card

st.set_page_config(page_title="Dashboard", page_icon="??")
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

SOFT_COLORS = [
    "#8FB7FF",
    "#A8D5BA",
    "#F7C59F",
    "#D7BDE2",
    "#F5B7B1",
    "#AED6F1",
    "#F9E79F",
    "#A3E4D7"
]


def show_warning_if_empty(data):
    if data is None:
        st.warning("Not enough history generated for this visual.")
        return True

    if isinstance(data, pd.Series) and data.empty:
        st.warning("Not enough history generated for this visual.")
        return True

    if isinstance(data, pd.DataFrame) and data.empty:
        st.warning("Not enough history generated for this visual.")
        return True

    return False


def render_bar_with_pie_tabs(series_data, label_name):
    """
    Keeps bar charts in the same native Streamlit style and adds a soft pie chart tab.
    """
    if show_warning_if_empty(series_data):
        return

    tab1, tab2 = st.tabs(["Bar chart", "Pie chart"])

    with tab1:
        bar_df = series_data.to_frame(name="Count")
        st.bar_chart(bar_df, height=320)

    with tab2:
        pie_df = series_data.reset_index()
        pie_df.columns = [label_name, "Count"]
        pie_df[label_name] = pie_df[label_name].astype(str)

        color_scale = alt.Scale(
            domain=pie_df[label_name].tolist(),
            range=SOFT_COLORS[:len(pie_df)]
        )

        pie_chart = (
            alt.Chart(pie_df)
            .mark_arc(outerRadius=120)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    f"{label_name}:N",
                    scale=color_scale,
                    legend=alt.Legend(title=None)
                ),
                tooltip=[
                    alt.Tooltip(f"{label_name}:N", title=label_name),
                    alt.Tooltip("Count:Q", title="Count")
                ]
            )
            .properties(height=320)
        )

        st.altair_chart(pie_chart, width='stretch')


def render_confusion_matrix_style_heatmap(matrix_df):
    """
    Interactive class-verdict matrix in the same dashboard card layout.
    """
    if show_warning_if_empty(matrix_df):
        return

    heatmap_df = matrix_df.reset_index().melt(
        id_vars=matrix_df.index.name or "class_label",
        var_name="Verdict",
        value_name="Count"
    )

    first_col = matrix_df.index.name or "class_label"
    heatmap_df = heatmap_df.rename(columns={first_col: "Class label"})

    base = alt.Chart(heatmap_df)

    rect = base.mark_rect().encode(
        x=alt.X("Verdict:N", axis=alt.Axis(title=None)),
        y=alt.Y("Class label:N", axis=alt.Axis(title=None)),
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=[
            alt.Tooltip("Class label:N", title="Class label"),
            alt.Tooltip("Verdict:N", title="Verdict"),
            alt.Tooltip("Count:Q", title="Count")
        ]
    )

    text = base.mark_text(fontSize=11).encode(
        x=alt.X("Verdict:N"),
        y=alt.Y("Class label:N"),
        text=alt.Text("Count:Q"),
        color=alt.value("black")
    )

    st.altair_chart((rect + text).properties(height=320), width='stretch')


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

k1, k2, k3, k4, k5, k6 = st.columns(6, gap="medium", vertical_alignment="top")

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
    verdict_counts = df["verdict"].fillna("N/A").value_counts()
    render_bar_with_pie_tabs(verdict_counts, "Verdict")
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
    render_bar_with_pie_tabs(class_counts, "Class label")
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

st.markdown("<br>", unsafe_allow_html=True)

mid_left, mid_right = st.columns([1, 1], gap="large")

with mid_left:
    open_card("Anomaly distribution per class", "Classes most frequently associated with anomalous results.")
    anomaly_class_counts = (
        df[df["verdict"] == "ANOMALOUS"]["class_label"]
        .fillna("N/A")
        .value_counts()
    )
    render_bar_with_pie_tabs(anomaly_class_counts, "Class label")
    close_card()

with mid_right:
    open_card("Manual inspection per class", "Classes most frequently sent for manual inspection.")
    manual_class_counts = (
        df[df["verdict"] == "MANUAL_INSPECTION"]["class_label"]
        .fillna("N/A")
        .value_counts()
    )
    render_bar_with_pie_tabs(manual_class_counts, "Class label")
    close_card()

st.markdown("<br>", unsafe_allow_html=True)

bottom_left, bottom_right = st.columns([1, 1], gap="large")

with bottom_left:
    open_card("Class-verdict matrix", "Heatmap-style matrix showing how predicted classes map to final verdicts.")
    matrix_df = pd.crosstab(
        df["class_label"].fillna("N/A"),
        df["verdict"].fillna("N/A")
    )
    render_confusion_matrix_style_heatmap(matrix_df)
    close_card()
