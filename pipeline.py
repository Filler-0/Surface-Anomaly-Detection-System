import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from PIL import Image

from object_localisation_classification.prepare_image import prepare_image

BASE_DIR = Path(__file__).parent
TEMP_RUNS_DIR = BASE_DIR / "temp_runs"
RESULTS_DIR = BASE_DIR / "results"
CROPS_DIR = BASE_DIR / "uploads" / "crops"

TEMP_RUNS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_CONFIDENCE_THRESHOLD = 0.75
ANOMALY_CONFIDENCE_THRESHOLD = 60.0


def save_crop_image(cropped_image: Image.Image, original_name: str) -> Path:
    crop_path = CROPS_DIR / f"{uuid.uuid4()}_{original_name}"
    cropped_image.save(crop_path)
    return crop_path


def create_single_image_temp_folder(image_path: Path) -> Path:
    run_dir = TEMP_RUNS_DIR / str(uuid.uuid4())
    run_dir.mkdir(parents=True, exist_ok=True)

    temp_image_path = run_dir / image_path.name
    shutil.copy(image_path, temp_image_path)
    return run_dir


def run_anomaly_detector(category: str, image_dir: Path) -> str:
    result = subprocess.run(
        ["python", "multi_detector.py", category, str(image_dir)],
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Anomaly detector failed.")

    return result.stdout.strip()


def parse_anomaly_output(output: str) -> tuple[str, float]:
    anomalous_match = re.search(r"(.+) is anomalous with ([0-9.]+)% certainty", output)
    normal_match = re.search(r"(.+) is normal", output)

    if anomalous_match:
        score = float(anomalous_match.group(2))
        return "ANOMALOUS", score

    if normal_match:
        return "NORMAL", 0.0

    raise ValueError(f"Unexpected anomaly detector output: {output}")


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


def run_full_pipeline(saved_image_path: str) -> dict:
    saved_image_path = Path(saved_image_path)

    pil_image = Image.open(saved_image_path).convert("RGB")
    prep_result = prepare_image(pil_image)

    class_label = prep_result["label"]
    class_confidence = float(prep_result["confidence"])
    top3_predictions = prep_result["top3"]
    cropped_image = prep_result["cropped"]

    crop_path = save_crop_image(cropped_image, saved_image_path.name)

    result = {
        "image_path": str(saved_image_path),
        "crop_path": str(crop_path),
        "class_label": class_label,
        "class_confidence": class_confidence,
        "top3_predictions": json.dumps(top3_predictions),
        "top3_predictions_list": top3_predictions,
        "heatmap_path": None,
        "result_image_path": None,
        "anomaly_score": None,
        "verdict": None,
        "raw_output": None,
        "temp_run_dir": None,
    }

    if class_label.lower() == "unknown":
        suggestions_text = ", ".join(
            f"{item['label']} ({item['confidence'] * 100:.1f}%)"
            for item in top3_predictions
        )

        result["verdict"] = "UNSUPPORTED_FORMAT"
        result["raw_output"] = (
            "This object is not supported. Please upload only: "
            "bottle, carpet, grid, tile, wood. "
            f"Closest supported predictions were: {suggestions_text}. "
            "Anomaly detection was not run."
        )
        return result

    if class_confidence < CLASS_CONFIDENCE_THRESHOLD:
        result["verdict"] = "MANUAL_INSPECTION"
        result["raw_output"] = (
            f"Classification confidence is too low ({class_confidence * 100:.1f}%). "
            "The image should be reviewed manually."
        )
        return result

    temp_run_dir = create_single_image_temp_folder(crop_path)
    result["temp_run_dir"] = str(temp_run_dir)

    anomaly_output = run_anomaly_detector(class_label, temp_run_dir)
    anomaly_verdict, anomaly_score = parse_anomaly_output(anomaly_output)

    result["anomaly_score"] = anomaly_score
    result["result_image_path"] = find_latest_result_image()

    if anomaly_score < ANOMALY_CONFIDENCE_THRESHOLD and anomaly_verdict != 'NORMAL':
        result["verdict"] = "MANUAL_INSPECTION"
        result["raw_output"] = (
            f"Anomaly detection confidence is too low ({anomaly_score:.2f}%). "
            "The result requires manual inspection."
        )
        return result

    result["verdict"] = anomaly_verdict
    result["raw_output"] = anomaly_output
    return result


def cleanup_temp_run(temp_run_dir: str | None):
    if temp_run_dir:
        temp_path = Path(temp_run_dir)
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)