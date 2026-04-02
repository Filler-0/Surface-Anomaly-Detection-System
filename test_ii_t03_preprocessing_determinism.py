# test_ii_t03_preprocessing_determinism.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the classifier loader directly
from object_localisation_classification.classifier import load_classifier  # type: ignore

DEVICE = "cpu"


def image_to_tensor(image_path: Path):
    """
    Apply the exact classifier preprocessing transform to one image
    and return the resulting tensor as a NumPy array.
    """
    model, meta, transform, device = load_classifier(DEVICE)

    pil_image = Image.open(image_path).convert("RGB")
    tensor = transform(pil_image)

    # Convert to numpy safely
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return tensor.numpy().astype(np.float64, copy=False)

    return np.asarray(tensor, dtype=np.float64)


def main():
    # Change this path if needed
    image_path = PROJECT_ROOT / "test_images" / "normal" / "001.png"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Reference image not found: {image_path}\n"
            f"Edit image_path in the script to point to one valid test image."
        )

    runs = 5
    tensors = []

    print("=" * 70)
    print("II-T03: Pre-processing determinism")
    print("=" * 70)
    print(f"Reference image: {image_path}")

    for i in range(runs):
        arr = image_to_tensor(image_path)
        tensors.append(arr)
        print(f"Run {i + 1}: shape={arr.shape}, dtype={arr.dtype}")

    base = tensors[0]
    max_deviation = 0.0

    for idx, arr in enumerate(tensors[1:], start=2):
        if arr.shape != base.shape:
            print(f"[FAIL] Run 1 shape {base.shape} != Run {idx} shape {arr.shape}")
            raise SystemExit(1)

        deviation = float(np.max(np.abs(base - arr)))
        print(f"Run 1 vs Run {idx}: max absolute deviation = {deviation}")
        max_deviation = max(max_deviation, deviation)

    print("-" * 70)
    print(f"Max tensor deviation: {max_deviation}")
    print(f"Result: {'PASS' if max_deviation == 0.0 else 'FAIL'}")

    raise SystemExit(0 if max_deviation == 0.0 else 1)


if __name__ == "__main__":
    main()
