"""
Image Preparation Orchestrator
Subsystem 2 entry point.

Usage from app.py or any downstream module:
    from object_localisation_classification.prepare_image import prepare_image

    result = prepare_image(pil_image)
    result["label"]      - class name, "unknown", "uncertain", or "rejected"
    result["confidence"] - top-1 softmax score between 0.0 and 1.0
    result["top3"]       - list of (label, score) pairs for the top 3 predictions
"""

import torch
from PIL import Image
from .classifier import load_classifier, classify

# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level singletons loaded once and reused across calls
_classifier_model     = None
_classifier_meta      = None
_classifier_transform = None
_device               = None


def _init():
    global _classifier_model, _classifier_meta, _classifier_embed, _classifier_transform, _device

    if _classifier_model is None:
        (
            _classifier_model,
            _classifier_meta,
            _classifier_transform,
            _device,
        ) = load_classifier(DEVICE)


def prepare_image(pil_image: Image.Image) -> dict:
    """
    Classify a single image.

    Args:
        pil_image: PIL.Image - raw uploaded image

    Returns a dict with:
    - label: str - predicted class, "unknown", "uncertain", or "rejected"
    - confidence: float - top-1 softmax probability
    - top3: list - top-3 (label, score) pairs
    """
    _init()

    result = classify(
        pil_image,
        model=_classifier_model,
        meta=_classifier_meta,
        transform=_classifier_transform,
        device=_device,
    )

    return {
        "label":      result["label"],
        "confidence": result["confidence"],
        "top3":       result["top3"],
    }