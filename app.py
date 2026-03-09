import re
import shutil
import subprocess
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

from db import insert_inspection

from ui_styles import inject_global_styles, render_hero, open_card, close_card

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_RUNS_DIR = BASE_DIR / "temp_runs"
RESULTS_DIR = BASE_DIR / "results"

UPLOADS_DIR.mkdir(exist_ok=True)
TEMP_RUNS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Surface Anomaly Detection System",
    page_icon="🪵",
    layout="wide",
)

inject_global_styles()

render_hero(
    "Surface Anomaly Detection System",
    "Upload an image, run anomaly detection, and review the generated result visualization."
)

def save_uploaded_file(uploaded_file) -> Path:
    unique_name = f"{uuid.uuid4()}_{uploaded_file.name}"
    save_path = UPLOADS_DIR / unique_name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def create_single_image_temp_folder(saved_image_path: Path) -> Path:
    run_dir = TEMP_RUNS_DIR / str(uuid.uuid4())
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_image_path = run_dir / saved_image_path.name
    shutil.copy(saved_image_path, temp_image_path)
    return run_dir


def run_detector(image_dir: Path) -> str:
    result = subprocess.run(
        ["python", "wood_detector.py", str(image_dir)],
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Detector failed.")

    return result.stdout.strip()


def parse_detector_output(output: str):
    anomalous_match = re.search(r"(.+) is anomalous with ([0-9.]+)% certainty", output)
    normal_match = re.search(r"(.+) is normal", output)

    if anomalous_match:
        image_path = anomalous_match.group(1).strip()
        score = float(anomalous_match.group(2))
        return image_path, "ANOMALOUS", score

    if normal_match:
        image_path = normal_match.group(1).strip()
        return image_path, "NORMAL", 0.0

    raise ValueError(f"Unexpected detector output: {output}")


def find_latest_result_image() -> str | None:
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    candidates = [
        path for path in RESULTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    if not candidates:
        return None

    latest_file = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


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

left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload image")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            preview_image = Image.open(uploaded_file)
            st.image(preview_image, caption="Uploaded image", use_container_width=True)
        except Exception:
            st.error("Invalid image file.")
            st.stop()

    analyze_clicked = st.button("Analyze")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("How it works")
    st.write("1. Upload an image.")
    st.write("2. The detector analyzes the surface.")
    st.write("3. The system generates a result visualization.")
    st.write("4. The result is saved to PostgreSQL.")
    st.write("5. You can review it later in Dashboard and History.")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and analyze_clicked:
    temp_dir = None

    try:
        saved_image_path = save_uploaded_file(uploaded_file)
        temp_dir = create_single_image_temp_folder(saved_image_path)

        with st.spinner("Running anomaly detector..."):
            output = run_detector(temp_dir)

        _, verdict, score = parse_detector_output(output)
        result_image_path = find_latest_result_image()

        inspection_id = insert_inspection(
            image_name=uploaded_file.name,
            image_path=str(saved_image_path),
            crop_path=None,
            heatmap_path=None,
            result_image_path=result_image_path,
            anomaly_score=score,
            verdict=verdict,
            raw_output=output,
        )

        st.success(f"Analysis complete. Record #{inspection_id} saved.")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Verdict", verdict)
        with m2:
            st.metric("Anomaly score", format_score(score))

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Detection result")

        if result_image_path:
            result_image = safe_open_image(result_image_path)
            if result_image is not None:
                st.image(
                    result_image,
                    caption="Generated result visualization",
                    use_container_width=True,
                )
            else:
                st.warning("Result image was found, but it could not be opened.")
        else:
            st.warning("No result visualization was found in the results folder.")

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Technical output"):
            st.code(output)

    except Exception as e:
        st.error(str(e))
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
