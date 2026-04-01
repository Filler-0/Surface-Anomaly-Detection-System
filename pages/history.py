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


def parse_top3(top3_json: str | None):
    if not top3_json:
        return []

    try:
        parsed = json.loads(top3_json)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []

    return []


def render_top3(top3_json: str | None):
    top3 = parse_top3(top3_json)
    if not top3:
        st.write("**Top-3 predictions:** N/A")
        return

    st.write("**Top-3 predictions:**")
    for cls_name, cls_score in top3:
        st.write(f"- {cls_name}: {cls_score * 100:.1f}%")


def is_unsupported(row: dict) -> bool:
    return (row.get("verdict") or "").upper() == "UNSUPPORTED_FORMAT"


def class_label_for_display(row: dict) -> str:
    class_label = (row.get("class_label") or "N/A").upper()
    if is_unsupported(row) and class_label in {"REJECTED", "UNKNOWN", "UNCERTAIN"}:
        return "UNSUPPORTED FORMAT"
    return class_label


def class_badge_token(label: str) -> str:
    color_map = {
        "CARPET": "\U0001F7EB",
        "GRID": "\U0001F7E6",
        "TILE": "\u2B1C",
        "WOOD": "\U0001F7E7",
        "LEATHER": "\U0001F7E4",
        "UNSUPPORTED FORMAT": "\U0001F7EA",
    }
    token = color_map.get(label, "\U0001F539")
    return f"{token} [{label}]"


def verdict_badge_token(verdict: str) -> str:
    verdict_upper = (verdict or "").upper()
    if verdict_upper == "NORMAL":
        return "\U0001F7E2 [NORMAL]"
    if verdict_upper == "ANOMALOUS":
        return "\U0001F534 [ANOMALOUS]"
    if verdict_upper == "MANUAL_INSPECTION":
        return "\U0001F7E0 [MANUAL_INSPECTION]"
    return "\U0001F7E3 [UNSUPPORTED_FORMAT]"


def build_technical_details(row: dict, display_class: str) -> dict:
    return {
        "inspection_id": row["id"],
        "uploaded_file": row["image_name"],
        "image_path": row["image_path"],
        "class_label": display_class,
        "class_confidence_percent": (
            round(float(row["class_confidence"]) * 100.0, 2)
            if row["class_confidence"] is not None
            else None
        ),
        "verdict": row["verdict"],
        "anomaly_score_percent": (
            round(float(row["anomaly_score"]), 2)
            if row["anomaly_score"] is not None
            else None
        ),
        "result_image_path": row["result_image_path"],
        "top3_predictions": parse_top3(row["top3_predictions"]),
        "created_at": str(row["created_at"]),
        "raw_output": row["raw_output"],
    }


rows = fetch_history()

if not rows:
    st.info("No inspection history found yet.")
    st.stop()

# --- Filters ---
# Derive available options from actual data
all_classes = sorted({class_label_for_display(r) for r in rows})
all_verdicts = sorted({(r.get("verdict") or "N/A").upper() for r in rows})

st.markdown("### Filters")
filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

with filter_col1:
    selected_classes = st.multiselect(
        "Filter by class type",
        options=all_classes,
        default=[],
        placeholder="All classes",
    )

with filter_col2:
    selected_verdicts = st.multiselect(
        "Filter by verdict",
        options=all_verdicts,
        default=[],
        placeholder="All verdicts",
    )

with filter_col3:
    sort_order = st.selectbox(
        "Sort by",
        options=["Newest first", "Oldest first"],
        index=0,
    )

st.divider()

# --- Apply filters ---
filtered_rows = rows

if selected_classes:
    filtered_rows = [r for r in filtered_rows if class_label_for_display(r) in selected_classes]

if selected_verdicts:
    filtered_rows = [r for r in filtered_rows if (r.get("verdict") or "N/A").upper() in selected_verdicts]

if sort_order == "Oldest first":
    filtered_rows = list(reversed(filtered_rows))

# --- Results summary ---
st.caption(f"Showing {len(filtered_rows)} of {len(rows)} inspections")

if not filtered_rows:
    st.warning("No inspections match the selected filters.")
    st.stop()

# --- Render rows ---
for row in filtered_rows:
    display_class = class_label_for_display(row)
    class_token = class_badge_token(display_class)
    verdict_token = verdict_badge_token(row["verdict"])
    expander_title = (
        f"#{row['id']} | {row['image_name']} | {row['created_at']} | "
        f"{class_token} {verdict_token}"
    )

    with st.expander(expander_title):
        left_col, right_col = st.columns([1, 1], gap="large")
        unsupported = is_unsupported(row)

        with left_col:
            st.write(f"**Image Name:** {row['image_name']}")
            st.write(f"**Class label:** {display_class}")
            if not unsupported:
                st.write(f"**Class confidence:** {format_ratio(row['class_confidence'])}")
            st.write(f"**Anomaly score:** {format_score(row['anomaly_score'])}")
            st.write("**Verdict:**")
            verdict_badge(row["verdict"])
            st.write(f"**Created At:** {row['created_at']}")
            if not unsupported:
                render_top3(row["top3_predictions"])

        with right_col:
            original_image = safe_open_image(row["image_path"])
            if original_image is not None:
                st.image(original_image, caption="Original image", width="stretch")

            result_image = safe_open_image(row["result_image_path"])
            if result_image is not None:
                st.image(result_image, caption="Result visualization", width="stretch")

        with st.expander("Technical output"):
            technical_details = build_technical_details(row, display_class)
            st.json(technical_details)
            st.code(row["raw_output"] or "No technical output saved.", language="text")