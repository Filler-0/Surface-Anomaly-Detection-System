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
    page_icon="🪵",
    layout="wide",
)

inject_global_styles()

render_hero(
    "Surface Anomaly Detection System",
    "Upload an image, run classification and anomaly detection, and review the generated result visualization.",
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


def render_top3(top3_predictions_json: str | None):
    if not top3_predictions_json:
        return

    try:
        top3 = json.loads(top3_predictions_json)
    except Exception:
        return

    st.write("Top 3 predictions")
    for cls_name, cls_score in top3:
        st.write(f"- **{cls_name}** — {cls_score * 100:.1f}%")
        st.progress(float(cls_score))


left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    open_card("Upload image", "Choose a JPG or PNG image to start analysis.")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    preview_image = None
    if uploaded_file is not None:
        try:
            preview_image = Image.open(uploaded_file).convert("RGB")
            st.image(preview_image, caption="Uploaded image", width="stretch")
        except Exception:
            st.error("Invalid image file.")
            st.stop()

    analyze_clicked = st.button("Analyze")
    close_card()

with right_col:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("How it works")
    st.write("1. Upload an image (supported: leather, carpet, grid, tile and wood).")
    st.write("2. The system classifies the object.")
    st.write("3. If the object is supported, anomaly detection is run.")
    st.write("4. The result is saved to database.")
    st.write("5. You can review it later in Dashboard and History.")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and analyze_clicked:
    temp_run_dir = None

    try:
        saved_image_path = save_uploaded_file(uploaded_file)

        with st.spinner("Running full pipeline..."):
            pipeline_result = run_full_pipeline(str(saved_image_path))

        temp_run_dir = pipeline_result["temp_run_dir"]

        inspection_id = insert_inspection(
            image_name=uploaded_file.name,
            image_path=pipeline_result["image_path"],
            crop_path=pipeline_result["crop_path"],
            class_label=pipeline_result["class_label"],
            class_confidence=pipeline_result["class_confidence"],
            top3_predictions=pipeline_result["top3_predictions"],
            heatmap_path=pipeline_result["heatmap_path"],
            result_image_path=pipeline_result["result_image_path"],
            anomaly_score=pipeline_result["anomaly_score"],
            verdict=pipeline_result["verdict"],
            raw_output=pipeline_result["raw_output"],
        )

        st.markdown("---")
        st.subheader("Step 1 — Object classification")

        class_col1, class_col2 = st.columns([1, 1], gap="large")

        with class_col1:
            crop_image = safe_open_image(pipeline_result["crop_path"])
            if crop_image is not None:
                st.image(crop_image, caption="Cropped ROI", width="stretch")

        with class_col2:
            st.metric("Object type", pipeline_result["class_label"].upper())
            st.metric("Confidence", format_percent_from_ratio(pipeline_result["class_confidence"]))
            render_top3(pipeline_result["top3_predictions"])

        st.markdown("---")
        st.subheader("Step 2 — Final decision")

        verdict = pipeline_result["verdict"]

        if verdict == "UNSUPPORTED_FORMAT":
            st.warning("The uploaded product format/type is not supported by the system.")
            st.info(f"Record #{inspection_id} saved with status: UNSUPPORTED_FORMAT")

        elif verdict == "MANUAL_INSPECTION":
            st.warning("The image should be sent for manual inspection.")
            st.info(f"Record #{inspection_id} saved with status: MANUAL_INSPECTION")

            if pipeline_result["anomaly_score"] is not None:
                st.metric("Anomaly confidence", format_percent(pipeline_result["anomaly_score"]))

        else:
            st.success(f"Analysis complete. Record #{inspection_id} saved.")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Object class", pipeline_result["class_label"].upper())
            with m2:
                st.metric("Class confidence", format_percent_from_ratio(pipeline_result["class_confidence"]))
            with m3:
                st.metric("Verdict", pipeline_result["verdict"])
            with m4:
                st.metric("Anomaly score", format_percent(pipeline_result["anomaly_score"]))

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

        with st.expander("Technical output"):
            st.code(pipeline_result["raw_output"] or "No technical output saved.")

    except Exception as e:
        st.error(str(e))
    finally:
        cleanup_temp_run(temp_run_dir)