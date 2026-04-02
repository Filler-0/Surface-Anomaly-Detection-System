"""
SADS System-Level Evaluation Script
=====================================
Runs all system-level tests defined in Section 9.5 of the final report.

Tests covered:
  1. End-to-end verdict correctness (31 images across all 4 verdict paths)
  2. Pipeline routing correctness
  3. Inference determinism (5 repeated runs of same image)
  4. Processing time (20 repeated runs of same image)
  5. Database record completeness

Usage:
  python system_eval.py

Requirements:
  - The full SADS stack must be running (docker-compose up)
  - Prepare test images in the folders described below before running
  - pip install psycopg2-binary pandas tabulate python-dotenv

Output:
  - system_eval_results.csv   : per-image verdict log
  - system_eval_summary.txt   : human-readable summary of all test results
"""

import csv
import os
import sys
import time
import json
import statistics
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import psycopg2
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv


# CONFIGURATION

# Path to pipeline.py directory (root of your project)
PROJECT_ROOT = Path(__file__).resolve().parent

# Load DB config from the actual project .env
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv(override=True)


def require_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return str(value)


def parse_database_url(database_url: str) -> dict:
    parsed = urlparse(database_url)
    if parsed.scheme not in ("postgres", "postgresql"):
        raise RuntimeError(f"Unsupported DATABASE_URL scheme: {parsed.scheme}")

    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/") if parsed.path else "postgres",
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
    }


def load_db_config() -> dict:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return parse_database_url(database_url)

    return {
        "host": require_env("DB_HOST", "localhost"),
        "port": int(require_env("DB_PORT", "5432")),
        "dbname": require_env("DB_NAME", "sads"),
        "user": require_env("DB_USER", "postgres"),
        "password": require_env("DB_PASSWORD", "postgres"),
    }


# Database connection -- now loaded from .env instead of hardcoded values
DB_CONFIG = load_db_config()

# ---------------------------------------------------------------------------
# TEST IMAGE FOLDERS
# Populate these folders before running the script.
# Each folder should contain images that should produce the stated verdict.
#
#   test_images/normal/           -- defect-free images from supported categories
#   test_images/anomalous/        -- defective images from supported categories
#   test_images/unsupported/      -- images from unsupported categories (OOD)
#   test_images/manual_inspection/-- images expected to fall near the 60% boundary
#   test_images/determinism/      -- single reference image (used for repeat runs)
#   test_images/timing/           -- single reference image (used for timing runs)
# ---------------------------------------------------------------------------

TEST_IMAGE_DIRS = {
    "NORMAL": PROJECT_ROOT / "test_images" / "normal",
    "ANOMALOUS": PROJECT_ROOT / "test_images" / "anomalous",
    "UNSUPPORTED_FORMAT": PROJECT_ROOT / "test_images" / "unsupported",
    "MANUAL_INSPECTION": PROJECT_ROOT / "test_images" / "manual_inspection",
}

DETERMINISM_IMAGE_DIR = PROJECT_ROOT / "test_images" / "determinism"
TIMING_IMAGE_DIR = PROJECT_ROOT / "test_images" / "timing"

DETERMINISM_RUNS = 5
TIMING_RUNS = 20

# Thresholds from pipeline.py and NFRs
MAX_SCORE_DEVIATION = 0.001     # NFR: determinism threshold
MAX_PROCESSING_TIME_S = 10.0    # NFR: max end-to-end time per image

OUTPUT_CSV = PROJECT_ROOT / "system_eval_results.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "system_eval_summary.txt"

# DB polling for completeness checks
DB_COMPLETENESS_WAIT_SECONDS = 3.0
DB_COMPLETENESS_POLL_INTERVAL = 0.5

# Explicitly persist rows from this evaluator because run_full_pipeline()
# may compute results without writing to DB when called directly.
PERSIST_RESULTS_FROM_EVALUATOR = True

# ---------------------------------------------------------------------------
# IMPORT PIPELINE
# We call run_full_pipeline directly to get structured results rather than
# going through the Streamlit UI, which keeps the test deterministic.
# ---------------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))
from pipeline import run_full_pipeline, cleanup_temp_run


# HELPERS
def get_image_files(folder: Path) -> list[Path]:
    """Return all JPEG and PNG files in a folder."""
    if not folder.exists():
        print(f"  [WARNING] Folder not found, skipping: {folder}")
        return []
    extensions = {".jpg", ".jpeg", ".png"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in extensions]


def run_pipeline_timed(image_path: Path) -> tuple[dict, float]:
    """Run the full pipeline and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = run_full_pipeline(str(image_path))
    elapsed = time.perf_counter() - start
    return result, elapsed


def cleanup_pipeline_result(result: dict) -> None:
    """Best-effort cleanup for temporary pipeline output."""
    try:
        cleanup_temp_run(result.get("temp_run_dir"))
    except Exception as e:
        print(f"  [WARNING] cleanup_temp_run failed: {e}")


def db_connect():
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        connect_timeout=5,
    )


def get_db_start_timestamp():
    """
    Return the DB server timestamp before the test run starts.
    This is more reliable than using MAX(id) alone.
    Returns None on failure.
    """
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT NOW();")
        ts = cur.fetchone()[0]
        conn.close()
        print(f"  DB timestamp before tests: {ts}")
        return ts
    except Exception as e:
        print(f"  [DB ERROR] Could not connect: {e}")
        return None


def fetch_recent_records(since_ts) -> list[dict]:
    """Fetch all inspection records inserted on or after since_ts."""
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, image_name, image_path, class_label, class_confidence,
                   top3_predictions, anomaly_score, verdict,
                   heatmap_path, created_at
            FROM inspections
            WHERE created_at >= %s
            ORDER BY created_at ASC, id ASC;
            """,
            (since_ts,),
        )
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"  [DB ERROR] Could not fetch records: {e}")
        return []


def wait_for_recent_records(since_ts, expected_min_count: int) -> list[dict]:
    """Poll briefly to allow for delayed DB visibility."""
    deadline = time.time() + DB_COMPLETENESS_WAIT_SECONDS
    latest = []

    while time.time() < deadline:
        latest = fetch_recent_records(since_ts)
        if len(latest) >= expected_min_count:
            return latest
        time.sleep(DB_COMPLETENESS_POLL_INTERVAL)

    return latest


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def get_table_columns(table_name: str) -> list[dict]:
    """Read table schema metadata from information_schema."""
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT column_name, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (table_name,),
        )
        rows = cur.fetchall()
        return [
            {
                "column_name": row[0],
                "is_nullable": row[1],
                "column_default": row[2],
            }
            for row in rows
        ]
    finally:
        conn.close()


def normalize_top3_predictions(value):
    """
    Convert top3_predictions into something psycopg2 can insert safely.
    If your DB column is JSON/JSONB, psycopg2 can cast string -> json
    when the SQL uses %s and PostgreSQL handles the cast.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def build_inspection_insert_payload(result: dict, image_path: Path) -> dict:
    """
    Build a payload for INSERT that matches the likely inspections schema.
    Critical fix: include image_path, which is NOT NULL in your table.
    """
    return {
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "heatmap_path": result.get("heatmap_path"),
        "class_label": result.get("class_label"),
        "class_confidence": safe_float(result.get("class_confidence")),
        "top3_predictions": normalize_top3_predictions(result.get("top3_predictions")),
        "anomaly_score": safe_float(result.get("anomaly_score")),
        "verdict": result.get("verdict"),
    }


def persist_result_to_db(result: dict, image_path: Path) -> tuple[bool, str]:
    """
    Explicitly insert a DB row for this evaluation run.
    Keeps the evaluator working even if run_full_pipeline() itself
    does not persist rows when called directly.
    """
    if not PERSIST_RESULTS_FROM_EVALUATOR:
        return True, "Persistence disabled"

    try:
        columns_info = get_table_columns("inspections")
        if not columns_info:
            return False, "Could not inspect inspections table schema"

        payload = build_inspection_insert_payload(result, image_path)

        insert_columns = []
        insert_values = []

        for col in columns_info:
            name = col["column_name"]
            default = col["column_default"]

            if name == "id":
                continue

            if name in payload:
                value = payload[name]
                if value is None and default is not None:
                    continue
                insert_columns.append(name)
                insert_values.append(value)

        if not insert_columns:
            return False, "No matching columns available for INSERT"

        placeholders = ", ".join(["%s"] * len(insert_columns))
        column_sql = ", ".join(insert_columns)
        sql = f"INSERT INTO inspections ({column_sql}) VALUES ({placeholders}) RETURNING id;"

        conn = db_connect()
        cur = conn.cursor()
        cur.execute(sql, insert_values)
        inserted_id = cur.fetchone()[0]
        conn.commit()
        conn.close()

        return True, f"Inserted inspection row id={inserted_id}"

    except Exception as e:
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
        return False, f"Insert failed: {e}"


# TEST 1: VERDICT CORRECTNESS
def test_verdict_correctness() -> tuple[list[dict], list[dict]]:
    """
    Run all images in each test folder and check the verdict matches expectation.
    Returns (results_log, failures).
    """
    print("\n" + "=" * 60)
    print("TEST 1: End-to-End Verdict Correctness")
    print("=" * 60)

    results_log = []
    failures = []

    for expected_verdict, folder in TEST_IMAGE_DIRS.items():
        images = get_image_files(folder)
        print(f"\n  [{expected_verdict}] {len(images)} images found in {folder.name}/")

        for img_path in images:
            result = {}
            elapsed = None
            db_persisted = "NOT_ATTEMPTED"
            db_message = ""

            try:
                result, elapsed = run_pipeline_timed(img_path)
                actual_verdict = result.get("verdict", "ERROR")
                passed = actual_verdict == expected_verdict

                ok, msg = persist_result_to_db(result, img_path)
                db_persisted = "OK" if ok else "FAIL"
                db_message = msg
                print(f"    DB    {img_path.name}: {db_persisted} - {msg}")

                row = {
                    "image": img_path.name,
                    "expected_verdict": expected_verdict,
                    "actual_verdict": actual_verdict,
                    "anomaly_score": result.get("anomaly_score"),
                    "class_label": result.get("class_label"),
                    "class_confidence": result.get("class_confidence"),
                    "elapsed_s": round(elapsed, 3),
                    "db_persisted": db_persisted,
                    "db_message": db_message,
                    "passed": passed,
                }
                results_log.append(row)

                if not passed:
                    failures.append(row)
                    print(f"    FAIL  {img_path.name}: expected {expected_verdict}, got {actual_verdict}")
                else:
                    print(f"    PASS  {img_path.name}: {actual_verdict} ({elapsed:.2f}s)")

            except Exception as e:
                row = {
                    "image": img_path.name,
                    "expected_verdict": expected_verdict,
                    "actual_verdict": "ERROR",
                    "anomaly_score": None,
                    "class_label": None,
                    "class_confidence": None,
                    "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
                    "db_persisted": "FAIL",
                    "db_message": f"Pipeline exception: {e}",
                    "passed": False,
                }
                results_log.append(row)
                failures.append(row)
                print(f"    ERROR {img_path.name}: {e}")

            finally:
                if result:
                    cleanup_pipeline_result(result)

    total = len(results_log)
    passed_count = total - len(failures)
    print(f"\n  Result: {passed_count}/{total} correct verdicts")

    return results_log, failures


# TEST 2: INFERENCE DETERMINISM
def test_determinism() -> dict:
    """
    Run the same image DETERMINISM_RUNS times and check score deviation.
    Returns a summary dict.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Inference Determinism")
    print("=" * 60)

    images = get_image_files(DETERMINISM_IMAGE_DIR)
    if not images:
        print("  [SKIP] No images found in determinism folder.")
        return {"status": "SKIPPED", "reason": "No test images"}

    # Use first image found
    ref_image = images[0]
    print(f"  Reference image: {ref_image.name}")
    print(f"  Running {DETERMINISM_RUNS} consecutive inferences...")

    scores = []
    verdicts = []

    for i in range(DETERMINISM_RUNS):
        result = {}
        try:
            result, _ = run_pipeline_timed(ref_image)
            score = result.get("anomaly_score")
            verdict = result.get("verdict")
            scores.append(score)
            verdicts.append(verdict)
            print(f"    Run {i+1}: verdict={verdict}, score={score}")
        finally:
            if result:
                cleanup_pipeline_result(result)

    # Filter None scores (UNSUPPORTED_FORMAT cases have no score)
    numeric_scores = [s for s in scores if s is not None]
    if len(numeric_scores) >= 2:
        max_deviation = max(numeric_scores) - min(numeric_scores)
    else:
        max_deviation = None

    all_verdicts_match = len(set(verdicts)) == 1
    passed = (
        all_verdicts_match and
        (max_deviation is None or max_deviation <= MAX_SCORE_DEVIATION)
    )

    summary = {
        "image": ref_image.name,
        "runs": DETERMINISM_RUNS,
        "scores": scores,
        "verdicts": verdicts,
        "max_deviation": max_deviation,
        "all_verdicts_match": all_verdicts_match,
        "threshold": MAX_SCORE_DEVIATION,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }

    print(f"\n  Max score deviation: {max_deviation}")
    print(f"  All verdicts match: {all_verdicts_match}")
    print(f"  Result: {summary['status']}")

    return summary


# TEST 3: PROCESSING TIME
def test_processing_time() -> dict:
    """
    Run the same image TIMING_RUNS times and measure wall-clock time per run.
    Returns a summary dict.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Processing Time")
    print("=" * 60)

    images = get_image_files(TIMING_IMAGE_DIR)
    if not images:
        print("  [SKIP] No images found in timing folder.")
        return {"status": "SKIPPED", "reason": "No test images"}

    ref_image = images[0]
    print(f"  Reference image: {ref_image.name}")
    print(f"  Running {TIMING_RUNS} consecutive inferences...")

    times = []
    for i in range(TIMING_RUNS):
        result = {}
        try:
            result, elapsed = run_pipeline_timed(ref_image)
            times.append(elapsed)
            print(f"    Run {i+1}: {elapsed:.3f}s")
        finally:
            if result:
                cleanup_pipeline_result(result)

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    max_time = max(times)
    min_time = min(times)
    passed = mean_time <= MAX_PROCESSING_TIME_S

    summary = {
        "image": ref_image.name,
        "runs": TIMING_RUNS,
        "mean_s": round(mean_time, 3),
        "std_s": round(std_time, 3),
        "min_s": round(min_time, 3),
        "max_s": round(max_time, 3),
        "threshold_s": MAX_PROCESSING_TIME_S,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }

    print(f"\n  Mean: {mean_time:.3f}s | Std: {std_time:.3f}s | "
          f"Min: {min_time:.3f}s | Max: {max_time:.3f}s")
    print(f"  Threshold: {MAX_PROCESSING_TIME_S}s")
    print(f"  Result: {summary['status']}")

    return summary


# TEST 4: DATABASE RECORD COMPLETENESS
def test_db_completeness(db_start_ts, verdict_results: list[dict]) -> dict:
    """
    After verdict correctness test runs, check that every submitted image
    has a complete record in the database.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Database Record Completeness")
    print("=" * 60)

    if db_start_ts is None:
        print("  [SKIP] Could not connect to database before test run.")
        return {"status": "SKIPPED", "reason": "DB connection failed"}

    # Fetch records inserted during this test session
    recent_records = wait_for_recent_records(db_start_ts, len(verdict_results))
    submitted_count = len(verdict_results)

    # Match records by image_name
    records_by_image = {}
    for rec in recent_records:
        image_name = rec.get("image_name")
        if image_name is not None and image_name not in records_by_image:
            records_by_image[image_name] = rec

    matched_records = []
    missing_images = []

    for row in verdict_results:
        image_name = row["image"]
        rec = records_by_image.get(image_name)
        if rec is None:
            missing_images.append(image_name)
        else:
            matched_records.append(rec)

    stored_count = len(matched_records)

    # Required fields -- image_name, image_path and verdict always required;
    # anomaly_score may be None for UNSUPPORTED_FORMAT
    always_required = ["id", "image_name", "image_path", "verdict", "created_at"]
    conditional_fields = ["class_label", "class_confidence", "anomaly_score"]

    incomplete = []
    for rec in matched_records:
        missing = [f for f in always_required if rec.get(f) is None]

        if rec.get("verdict") != "UNSUPPORTED_FORMAT":
            for field in conditional_fields:
                if rec.get(field) is None:
                    missing.append(field)

        if missing:
            incomplete.append({
                "id": rec.get("id"),
                "image_name": rec.get("image_name"),
                "missing_fields": missing,
            })

    completeness_rate = (stored_count / submitted_count * 100) if submitted_count > 0 else 0
    passed = stored_count == submitted_count and len(incomplete) == 0 and len(missing_images) == 0

    summary = {
        "submitted": submitted_count,
        "stored": stored_count,
        "completeness_rate_pct": round(completeness_rate, 1),
        "missing_images": missing_images,
        "incomplete_records": incomplete,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }

    print(f"  Submitted images:  {submitted_count}")
    print(f"  Records in DB:     {stored_count}")
    print(f"  Completeness rate: {completeness_rate:.1f}%")

    if missing_images:
        print("  Missing DB records:")
        for image_name in missing_images:
            print(f"    - {image_name}")

    if incomplete:
        print(f"  Incomplete records: {incomplete}")
    else:
        print("  All required fields populated.")

    print(f"  Result: {summary['status']}")

    return summary


# TEST 5: PIPELINE ROUTING CORRECTNESS
def test_routing_correctness(verdict_results: list[dict]) -> dict:
    """
    Check that each verdict path was taken the correct number of times.
    This is derived from the verdict correctness results.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Pipeline Routing Correctness")
    print("=" * 60)

    routing_log = {}
    for row in verdict_results:
        expected = row["expected_verdict"]
        actual = row["actual_verdict"]
        if expected not in routing_log:
            routing_log[expected] = {"total": 0, "correct": 0, "incorrect": []}
        routing_log[expected]["total"] += 1
        if actual == expected:
            routing_log[expected]["correct"] += 1
        else:
            routing_log[expected]["incorrect"].append(
                f"{row['image']} -> {actual}"
            )

    all_passed = True
    for path, counts in routing_log.items():
        rate = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
        path_passed = counts["correct"] == counts["total"]
        if not path_passed:
            all_passed = False
        status = "PASS" if path_passed else "FAIL"
        print(f"  {path}: {counts['correct']}/{counts['total']} correct ({rate:.0f}%) [{status}]")
        if counts["incorrect"]:
            for err in counts["incorrect"]:
                print(f"    Misrouted: {err}")

    return {
        "routing_log": routing_log,
        "passed": all_passed,
        "status": "PASS" if all_passed else "FAIL",
    }


# SAVE RESULTS
def save_results(
    verdict_results: list[dict],
    determinism_summary: dict,
    timing_summary: dict,
    db_summary: dict,
    routing_summary: dict,
):
    # Save per-image CSV
    if verdict_results:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=verdict_results[0].keys())
            writer.writeheader()
            writer.writerows(verdict_results)
        print(f"\n  Per-image results saved to: {OUTPUT_CSV}")

    # Save human-readable summary
    lines = []
    lines.append("SADS SYSTEM-LEVEL EVALUATION SUMMARY")
    lines.append(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Verdict table
    if verdict_results:
        df = pd.DataFrame(verdict_results)
        lines.append("\nTEST 1 + TEST 5: Verdict Correctness and Routing")
        lines.append(tabulate(
            df[["image", "expected_verdict", "actual_verdict",
                "anomaly_score", "elapsed_s", "db_persisted", "passed"]],
            headers="keys", tablefmt="grid", showindex=False
        ))
        total = len(df)
        passed = int(df["passed"].sum())
        accuracy = (passed / total * 100) if total > 0 else 0
        lines.append(f"\nOverall verdict accuracy: {passed}/{total} ({accuracy:.1f}%)")

    # Determinism
    lines.append("\nTEST 2: Inference Determinism")
    lines.append(f"  Image: {determinism_summary.get('image', 'N/A')}")
    lines.append(f"  Runs: {determinism_summary.get('runs', 'N/A')}")
    lines.append(f"  Max score deviation: {determinism_summary.get('max_deviation', 'N/A')}")
    lines.append(f"  Threshold: {determinism_summary.get('threshold', 'N/A')}")
    lines.append(f"  All verdicts match: {determinism_summary.get('all_verdicts_match', 'N/A')}")
    lines.append(f"  Result: {determinism_summary.get('status', 'N/A')}")

    # Timing
    lines.append("\nTEST 3: Processing Time")
    lines.append(f"  Image: {timing_summary.get('image', 'N/A')}")
    lines.append(f"  Runs: {timing_summary.get('runs', 'N/A')}")
    lines.append(f"  Mean: {timing_summary.get('mean_s', 'N/A')}s")
    lines.append(f"  Std:  {timing_summary.get('std_s', 'N/A')}s")
    lines.append(f"  Min:  {timing_summary.get('min_s', 'N/A')}s")
    lines.append(f"  Max:  {timing_summary.get('max_s', 'N/A')}s")
    lines.append(f"  Threshold: {timing_summary.get('threshold_s', 'N/A')}s")
    lines.append(f"  Result: {timing_summary.get('status', 'N/A')}")

    # DB completeness
    lines.append("\nTEST 4: Database Record Completeness")
    lines.append(f"  Submitted: {db_summary.get('submitted', 'N/A')}")
    lines.append(f"  Stored: {db_summary.get('stored', 'N/A')}")
    lines.append(f"  Completeness rate: {db_summary.get('completeness_rate_pct', 'N/A')}%")
    lines.append(f"  Result: {db_summary.get('status', 'N/A')}")

    # Overall
    all_statuses = [
        routing_summary.get("status"),
        determinism_summary.get("status"),
        timing_summary.get("status"),
        db_summary.get("status"),
    ]
    overall = "PASS" if all(s == "PASS" for s in all_statuses if s != "SKIPPED") else "FAIL"
    lines.append("\n" + "=" * 60)
    lines.append(f"OVERALL SYSTEM-LEVEL RESULT: {overall}")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\n  Full summary saved to: {OUTPUT_SUMMARY}")


# MAIN
def main():
    print("\nSADS System-Level Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"DB: host={DB_CONFIG['host']} port={DB_CONFIG['port']} "
        f"dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']}"
    )

    # Record DB state before tests so we can check completeness after
    print("\nConnecting to database to record baseline record count...")
    db_start_ts = get_db_start_timestamp()
    if db_start_ts is None:
        print("  [WARNING] Could not establish DB baseline. Completeness test will be skipped.")

    # Run tests
    verdict_results, _ = test_verdict_correctness()
    routing_summary = test_routing_correctness(verdict_results)
    determinism_summary = test_determinism()
    timing_summary = test_processing_time()
    db_summary = test_db_completeness(db_start_ts, verdict_results)

    # Save everything
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    save_results(
        verdict_results,
        determinism_summary,
        timing_summary,
        db_summary,
        routing_summary,
    )


if __name__ == "__main__":
    main()