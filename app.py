import json
from pathlib import Path

import streamlit as st
from PIL import Image

from db import insert_inspection
from pipeline import run_full_pipeline, cleanup_temp_run
from ui_styles import inject_global_styles, render_hero, open_card, close_card

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Surface Anomaly Detection System",
    page_icon="??",
    layout="wide",
)

inject_global_styles()

render_hero(
    "Surface Anomaly Detection System",
    "Upload one or more images, run classification and anomaly detection, and review the generated result visualizations.",
)


def save_uploaded_file(uploaded_file) -> Path:
    unique_name = f"{uploaded_file.name}"
    save_path = UPLOADS_DIR / unique_name

    counter = 1
    while save_path.exists():
        save_path = UPLOADS_DIR / f"{save_path.stem}_{counter}{save_path.suffix}"
        counter += 1

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path


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


def format_percent_from_ratio(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def parse_top3_predictions(top3_predictions_json: str | None):
    if not top3_predictions_json:
        return []

    try:
        parsed = json.loads(top3_predictions_json)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []

    return []


def render_top3(top3_predictions_json: str | None):
    top3 = parse_top3_predictions(top3_predictions_json)
    if not top3:
        return

    st.write("Top 3 predictions")
    for cls_name, cls_score in top3:
        st.write(f"- **{cls_name}** - {cls_score * 100:.1f}%")
        st.progress(max(0.0, min(1.0, float(cls_score))))


def get_verdict_style(verdict: str) -> tuple[str, str, str]:
    styles = {
        "UNSUPPORTED_FORMAT": ("Unsupported format", "#ef4444", "#fef2f2"),
        "MANUAL_INSPECTION": ("Manual inspection required", "#f59e0b", "#fff7ed"),
        "ANOMALOUS": ("Anomalous", "#dc2626", "#fef2f2"),
        "NORMAL": ("Normal", "#16a34a", "#f0fdf4"),
    }
    return styles.get(verdict, (verdict.replace("_", " ").title(), "#64748b", "#f8fafc"))


def render_verdict_metric_card(verdict: str):
    verdict_text, verdict_color, verdict_bg = get_verdict_style(verdict)
    st.markdown(
        f"""
        <div style="
            border: 1px solid {verdict_color};
            border-radius: 16px;
            padding: 14px;
            background: {verdict_bg};
            min-height: 96px;
            box-sizing: border-box;
        ">
            <div style="font-size: 0.92rem; color: #64748b; margin-bottom: 8px;">Verdict</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: {verdict_color}; line-height: 1.2;">{verdict_text.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_technical_details(uploaded_name: str, pipeline_result: dict, inspection_id: int) -> dict:
    return {
        "inspection_id": inspection_id,
        "uploaded_file": uploaded_name,
        "image_path": pipeline_result["image_path"],
        "class_label": pipeline_result["class_label"],
        "class_confidence_percent": (
            round(float(pipeline_result["class_confidence"]) * 100.0, 2)
            if pipeline_result["class_confidence"] is not None
            else None
        ),
        "verdict": pipeline_result["verdict"],
        "anomaly_score_percent": (
            round(float(pipeline_result["anomaly_score"]), 2)
            if pipeline_result["anomaly_score"] is not None
            else None
        ),
        "result_image_path": pipeline_result["result_image_path"],
        "top3_predictions": parse_top3_predictions(pipeline_result["top3_predictions"]),
        "raw_output": pipeline_result["raw_output"],
    }


left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    if "main_uploader_version" not in st.session_state:
        st.session_state.main_uploader_version = 0

    open_card("Upload images", "Choose one or more JPG or PNG images to start analysis.")
    uploaded_files = st.file_uploader(
        "Choose JPG or PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"main_uploader_{st.session_state.main_uploader_version}",
    )

    if st.session_state.get("uploads_cleared_message"):
        st.info(st.session_state.uploads_cleared_message)
        del st.session_state["uploads_cleared_message"]

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} image(s) selected")
        preview_cols = st.columns(3)

        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                preview_image = Image.open(uploaded_file).convert("RGB")
            except Exception:
                st.error(f"Invalid image file: {uploaded_file.name}")
                st.stop()

            with preview_cols[idx % 3]:
                st.image(preview_image, caption=uploaded_file.name, width="stretch")

    action_col_1, action_col_2 = st.columns(2, gap="small")
    with action_col_1:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    with action_col_2:
        clear_clicked = st.button("Clear selection", type="secondary", use_container_width=True)

    if clear_clicked:
        st.session_state.main_uploader_version += 1
        st.session_state.uploads_cleared_message = "Selection cleared. You can choose new files now."
        st.rerun()

    close_card()

with right_col:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("How it works")
    st.write("1. Upload one or more images (supported: leather, carpet, grid, tile and wood).")
    st.write("2. The system classifies each object.")
    st.write("3. If an object is supported, anomaly detection is run.")
    st.write("4. Each result is saved to database.")
    st.write("5. You can review saved results later in Dashboard and History.")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files and analyze_clicked:
    for file_index, uploaded_file in enumerate(uploaded_files, start=1):
        temp_run_dir = None

        st.markdown("---")
        st.subheader(f"Image {file_index}: {uploaded_file.name}")

        try:
            saved_image_path = save_uploaded_file(uploaded_file)

            with st.spinner(f"Running full pipeline for {uploaded_file.name}..."):
                pipeline_result = run_full_pipeline(str(saved_image_path))

            temp_run_dir = pipeline_result["temp_run_dir"]

            inspection_id = insert_inspection(
                image_name=uploaded_file.name,
                image_path=pipeline_result["image_path"],
                class_label=pipeline_result["class_label"],
                class_confidence=pipeline_result["class_confidence"],
                top3_predictions=pipeline_result["top3_predictions"],
                heatmap_path=pipeline_result["heatmap_path"],
                result_image_path=pipeline_result["result_image_path"],
                anomaly_score=pipeline_result["anomaly_score"],
                verdict=pipeline_result["verdict"],
                raw_output=pipeline_result["raw_output"],
            )

            st.subheader("Step 1 - Object classification")

            class_col1, class_col2 = st.columns([1, 1], gap="large")
            verdict = pipeline_result["verdict"]
            class_label = pipeline_result["class_label"]
            top3_hidden_for_unsupported = verdict == "UNSUPPORTED_FORMAT"
            display_object_type = (
                "UNSUPPORTED FORMAT"
                if top3_hidden_for_unsupported and class_label.lower() in {"rejected", "unknown", "uncertain"}
                else class_label.upper()
            )

            with class_col1:
                original_image = safe_open_image(pipeline_result["image_path"])
                if original_image is not None:
                    st.image(original_image, caption="Input image", width="stretch")

            with class_col2:
                st.metric("Object type", display_object_type)
                if top3_hidden_for_unsupported:
                    st.caption("Confidence hidden for unsupported formats.")
                    st.caption("Top-3 predictions hidden for unsupported formats.")
                else:
                    st.metric("Confidence", format_percent_from_ratio(pipeline_result["class_confidence"]))
                    render_top3(pipeline_result["top3_predictions"])

            st.markdown("---")
            st.subheader("Step 2 - Final decision")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Object class", display_object_type)
            with m2:
                class_conf_display = (
                    "N/A"
                    if top3_hidden_for_unsupported
                    else format_percent_from_ratio(pipeline_result["class_confidence"])
                )
                st.metric("Class confidence", class_conf_display)
            with m3:
                render_verdict_metric_card(verdict)
            with m4:
                st.metric("Anomaly score", format_percent(pipeline_result["anomaly_score"]))

            if verdict == "UNSUPPORTED_FORMAT":
                st.error("Unsupported format: anomaly detection was not run.")
                st.info(f"Record #{inspection_id} saved with status: UNSUPPORTED_FORMAT")

            elif verdict == "MANUAL_INSPECTION":
                st.warning("Manual inspection required: automated confidence is insufficient.")
                st.info(f"Record #{inspection_id} saved with status: MANUAL_INSPECTION")

            else:
                st.success(f"Analysis complete. Record #{inspection_id} saved.")

                open_card("Detection result", "Generated model visualization.")
                if pipeline_result["result_image_path"]:
                    result_image = safe_open_image(pipeline_result["result_image_path"])
                    if result_image is not None:
                        st.image(
                            result_image,
                            caption="Generated result visualization",
                            width="stretch",
                        )
                    else:
                        st.warning("Result image was found, but it could not be opened.")
                else:
                    st.warning("No result visualization was found in the results folder.")
                close_card()

            with st.expander(f"Technical output ({uploaded_file.name})"):
                technical_details = build_technical_details(uploaded_file.name, pipeline_result, inspection_id)
                st.json(technical_details)
                st.code(pipeline_result["raw_output"] or "No technical output saved.", language="text")

        except Exception as e:
            st.error(f"{uploaded_file.name}: {e}")
        finally:
            cleanup_temp_run(temp_run_dir)
