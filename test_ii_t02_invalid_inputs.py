# test_ii_t02_invalid_inputs.py

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_pipeline():
    try:
        from pipeline import run_full_pipeline  # type: ignore
        return run_full_pipeline
    except Exception as e:
        raise RuntimeError(f"Could not import run_full_pipeline from pipeline.py: {e}") from e


def make_test_files(tmp_dir: Path) -> list[Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []

    txt_file = tmp_dir / "invalid_text.txt"
    txt_file.write_text("this is not an image", encoding="utf-8")
    files.append(txt_file)

    pdf_file = tmp_dir / "fake_document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nThis is not a real PDF file.\n")
    files.append(pdf_file)

    bmp_file = tmp_dir / "fake_bitmap.bmp"
    bmp_file.write_bytes(b"BM" + b"\x00" * 64)
    files.append(bmp_file)

    tiff_file = tmp_dir / "fake_image.tiff"
    tiff_file.write_bytes(b"II*\x00" + b"\x00" * 64)
    files.append(tiff_file)

    corrupt_jpg = tmp_dir / "corrupt.jpg"
    corrupt_jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"not-a-real-jpeg")
    files.append(corrupt_jpg)

    return files


def classify_outcome(result: Any, error: Exception | None) -> tuple[bool, str]:
    """
    PASS if the pipeline rejects the invalid input gracefully.
    Acceptable outcomes:
    - raises an exception
    - returns a dict with an explicit error-like message or unsupported verdict
    """
    if error is not None:
        return True, f"Rejected via exception: {type(error).__name__}: {error}"

    if isinstance(result, dict):
        verdict = str(result.get("verdict", "")).strip().upper()
        message = str(result.get("message", "")).strip()
        raw = json.dumps(result, default=str)

        if verdict in {"UNSUPPORTED_FORMAT", "ERROR", "INVALID_INPUT"}:
            return True, f"Rejected via verdict={verdict}; message={message or 'N/A'}"

        errorish_words = [
            "error",
            "invalid",
            "unsupported",
            "cannot identify image file",
            "failed",
            "corrupt",
        ]
        lowered = raw.lower()
        if any(word in lowered for word in errorish_words):
            return True, f"Rejected via error-style response: {raw}"

        return False, f"Unexpected successful-style response: {raw}"

    return False, f"Unexpected non-dict response without exception: {result!r}"


def main() -> int:
    run_full_pipeline = load_pipeline()
    tmp_dir = PROJECT_ROOT / "tmp_invalid_inputs"
    files = make_test_files(tmp_dir)

    print("=" * 70)
    print("II-T02: Invalid file format rejection")
    print("=" * 70)

    passed = 0
    total = len(files)

    for path in files:
        result = None
        error: Exception | None = None

        try:
            result = run_full_pipeline(str(path))
        except Exception as e:
            error = e

        ok, detail = classify_outcome(result, error)
        status = "PASS" if ok else "FAIL"

        print(f"[{status}] {path.name}")
        print(f"       {detail}")

        if not ok and error is not None:
            print("       Traceback:")
            print("".join(traceback.format_exception(error)).rstrip())

        if ok:
            passed += 1

    print("-" * 70)
    print(f"Result: {passed}/{total} invalid inputs rejected correctly")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())