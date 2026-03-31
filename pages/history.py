import json
from pathlib import Path

import streamlit as st
from PIL import Image

from db import fetch_history
from ui_styles import inject_global_styles, render_hero, verdict_badge

st.set_page_config(page_title="History", page_icon="??")
inject_global_styles()

render_hero(
    "Inspection History",
    "Browse previously analyzed images and stored results."
)


def safe_open_image(image_path_str: str | None):
    if not image_path_str:
        return None

    image_path = Path(image_path_str)
    if not image_path.exists():
        return None

    try:
        return Image.open(image_path)
    except Exception:
        return None


def format_score(score):
    if score is None:
        return "N/A"
    return f"{score:.2f}%"


def format_ratio(score):
    if score is None:
        return "N/A"
    return f"{score * 100:.1f}%"


def render_top3(top3_json: str | None):
    if not top3_json:
        st.write("**Top-3 predictions:** N/A")
        return

    try:
        top3 = json.loads(top3_json)
    except Exception:
        st.write("**Top-3 predictions:** N/A")
        return

    st.write("**Top-3 predictions:**")
    for cls_name, cls_score in top3:
        st.write(f"- {cls_name}: {cls_score * 100:.1f}%")


rows = fetch_history()

if not rows:
    st.info("No inspection history found yet.")
    st.stop()

for row in rows:
    with st.expander(f"#{row['id']} | {row['image_name']} | {row['created_at']}"):
        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.write(f"**Image Name:** {row['image_name']}")
            st.write(f"**Class label:** {row['class_label'] or 'N/A'}")
            st.write(f"**Class confidence:** {format_ratio(row['class_confidence'])}")
            st.write(f"**Anomaly score:** {format_score(row['anomaly_score'])}")
            st.write("**Verdict:**")
            verdict_badge(row["verdict"])
            st.write(f"**Created At:** {row['created_at']}")
            render_top3(row["top3_predictions"])

        with right_col:
            original_image = safe_open_image(row["image_path"])
            if original_image is not None:
                st.image(original_image, caption="Original image", width="stretch")

            result_image = safe_open_image(row["result_image_path"])
            if result_image is not None:
                st.image(result_image, caption="Result visualization", width="stretch")

        with st.expander("Technical output"):
            st.code(row["raw_output"] or "No technical output saved.")
