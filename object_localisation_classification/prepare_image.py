"""
Image Preparation Orchestrator
Subsystem 1 + 2 combined entry point.

Usage (from app.py or wood_detector.py):
    from object_localisation_classification.prepare_image import prepare_image

    result = prepare_image(pil_image)
    # result["label"]      - "screw"
    # result["confidence"] - 0.97
    # result["cropped"]    - PIL.Image (the cropped ROI)
"""

import torch
from PIL import Image
from .localiser  import load_localiser, localise
from .classifier import load_classifier, classify

# Auto-detect device - GPU if available, CPU otherwise
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level singletons - loaded once, reused across calls
_localiser_model      = None
_classifier_model     = None
_classifier_meta      = None
_classifier_transform = None
_device               = None


def _init():
    global _localiser_model, _classifier_model
    global _classifier_meta, _classifier_transform, _device

    if _localiser_model is None:
        _localiser_model = load_localiser()

    if _classifier_model is None:
        _classifier_model, _classifier_meta, _classifier_transform, _device = \
            load_classifier(DEVICE)


def prepare_image(pil_image: Image.Image) -> dict:
    """
    Full pipeline: localise -> classify.

    Args:
        pil_image: PIL.Image - raw uploaded image (cropped or uncropped)

    Returns:
        {
            "label":      str,       - "screw", "wood", "unknown", ...
            "confidence": float,     - 0.0 to 1.0
            "top3":       list,      - top 3 (label, score) pairs
            "cropped":    PIL.Image, - the ROI crop passed to classifier
        }
    """
    _init()

    # Step 1 - Localise (crop ROI or passthrough)
    cropped = localise(pil_image, model=_localiser_model)

    # Step 2 - Classify
    result = classify(
        cropped,
        model=_classifier_model,
        meta=_classifier_meta,
        transform=_classifier_transform,
        device=_device,
    )

    return {
        "label":      result["label"],
        "confidence": result["confidence"],
        "top3":       result["top3"],
        "cropped":    cropped,
    }