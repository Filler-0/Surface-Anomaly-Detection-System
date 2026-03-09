from pathlib import Path

import streamlit as st
from PIL import Image

from db import fetch_history

from ui_styles import inject_global_styles, render_hero, open_card, close_card, verdict_badge

st.set_page_config(page_title="History", page_icon="🕘", layout="wide")

inject_global_styles()

render_hero(
    "Inspection History",
    "Browse previously analyzed images and stored results from PostgreSQL."
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


rows = fetch_history()

if not rows:
    st.info("No inspection history found yet.")
    st.stop()

for row in rows:
    icon = "✅" if row["verdict"] == "NORMAL" else "⚠️"

    with st.expander(f"{icon} #{row['id']} | {row['image_name']} | {row['verdict']} | {row['created_at']}"):
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.write(f"**Image Name:** {row['image_name']}")
            st.write(f"**Verdict:** {row['verdict']}")
            st.write(f"**Anomaly Score:** {format_score(row['anomaly_score'])}")
            st.write(f"**Created At:** {row['created_at']}")
            st.write(f"**Image Path:** {row['image_path']}")
            st.write(f"**Result Image Path:** {row['result_image_path'] or 'N/A'}")

        with right_col:
            original_image = safe_open_image(row["image_path"])
            if original_image is not None:
                st.image(original_image, caption="Original image", use_container_width=True)

            if row["result_image_path"]:
                result_image = safe_open_image(row["result_image_path"])
                if result_image is not None:
                    st.image(result_image, caption="Result visualization", use_container_width=True)

        with st.expander("Raw detector output"):
            st.code(row["raw_output"] or "No raw output saved.")
