"""
Subsystem 2 - Object Classification
Loads the fine-tuned MobileNetV3-Small classifier and returns a class label + confidence.
Model: MobileNetV3-Small fine-tuned on MVTec AD surface classes.

Supported classes:
- carpet
- grid
- leather
- tile
- wood

Decision rules applied in order:
1. If top-1 confidence is below UNKNOWN_THRESHOLD, return "unknown"
2. If top-1 vs top-2 margin is below MARGIN_THRESHOLD, return "unknown"
3. If confidence is spread across 2-3 classes with no clear winner, return "uncertain"
4. If the Mahalanobis distance to the nearest class centroid exceeds the fitted threshold, return "unknown"
5. Otherwise return the top-1 class label
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np


BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "mobilenet_v3_small.pt"
EMBED_PATH = MODELS_DIR / "mobilenet_v3_small_embed.pt"
META_PATH  = Path(__file__).parent / "classes.json"

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Minimum softmax confidence for the top-1 prediction to be accepted
UNKNOWN_THRESHOLD = 0.80

# Minimum required gap between top-1 and top-2 confidence scores
MARGIN_THRESHOLD = 0.15

# Top-1 must be below this value to be considered an uncertain prediction
UNCERTAINTY_TOP1_CEILING = 0.55

# Top-2 must be at least this high to trigger the uncertainty check
UNCERTAINTY_TOP2_FLOOR = 0.10

# Top-3 must be at least this high to flag a three-way split
UNCERTAINTY_TOP3_FLOOR = 0.08

# Gap between top-1 and top-2 must be below this for uncertainty to be flagged
UNCERTAINTY_MARGIN_CEIL = 0.20


class MobileNetV3SmallClassifier(nn.Module):
    """
    MobileNetV3-Small fine-tuned for surface classification.
    Mirrors the architecture defined in the training notebook exactly.
    """

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)

        self.features = base.features
        self.avgpool  = base.avgpool
        in_features   = base.classifier[0].in_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled feature vector before the classifier head."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 1D feature vector."""
    norm = np.linalg.norm(x)
    return x / max(norm, eps)


def _mahalanobis_distance(feat: np.ndarray, centroid: np.ndarray,
                           inv_cov: np.ndarray) -> float:
    """Compute the Mahalanobis distance between a feature vector and a class centroid."""
    diff = feat - centroid
    return float(np.sqrt(diff @ inv_cov @ diff))


def _embedding_gate(feat: np.ndarray, embed_data: dict) -> bool:
    """
    Return True if the embedding passes the OOD gate, False if it should be rejected.

    Matches the notebook embedding_reject function exactly:
    - L2-normalize the feature vector first
    - Find the nearest class centroid by Mahalanobis distance
    - Reject if that minimum distance exceeds the threshold for the nearest class

    embed_data must contain keys: centroids, inv_covs, thresholds.
    """
    centroids  = embed_data["centroids"]
    inv_covs   = embed_data["inv_covs"]
    thresholds = embed_data["thresholds"]

    if not centroids:
        return True

    # L2-normalize to match how centroids were fitted in the notebook
    feat = _l2_normalize(feat)

    min_dist    = float("inf")
    nearest_idx = None

    for cls_idx, centroid in centroids.items():
        if cls_idx not in inv_covs:
            continue
        inv_cov = inv_covs[cls_idx]
        dist    = _mahalanobis_distance(feat, centroid, inv_cov)
        if dist < min_dist:
            min_dist    = dist
            nearest_idx = cls_idx

    if nearest_idx is None:
        return True

    return min_dist <= thresholds[nearest_idx]


def load_classifier(device: str = "cpu"):
    """
    Load the MobileNetV3-Small classifier, embedding gate data, and class metadata.

    Returns model, meta, embed_data, transform, device.
    model      - nn.Module in eval mode
    meta       - dict with classes, num_classes, etc.
    embed_data - dict with centroids, inv_covs, thresholds as numpy arrays
    transform  - torchvision preprocessing pipeline
    device     - str
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not EMBED_PATH.exists():
        raise FileNotFoundError(f"Embedding gate not found: {EMBED_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Class metadata not found: {META_PATH}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]

    model = MobileNetV3SmallClassifier(num_classes=num_classes, dropout=0.2)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Support both a raw state dict and a checkpoint dict with a "state_dict" key
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model = model.to(device).eval()

    raw_embed = torch.load(EMBED_PATH, map_location="cpu", weights_only=False)

    embed_data = {
        "centroids": {
            k: v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
            for k, v in raw_embed["centroids"].items()
        },
        "inv_covs": {
            k: v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
            for k, v in raw_embed["inv_covs"].items()
        },
        "thresholds": raw_embed["thresholds"],
    }

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return model, meta, embed_data, transform, device


def _is_uncertain(probs: np.ndarray) -> bool:
    """
    Return True when confidence is spread across 2-3 classes with no clear winner.

    Two scenarios are checked:

    Two-way split: top-1 is weak and top-2 is close behind.
    Example: wood=30%, leather=20% triggers manual inspection.

    Three-way split: top-1 is weak, top-2 is close, and top-3 is also meaningful.
    Example: tile=23%, wood=13%, grid=10% triggers manual inspection.
    """
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    top3 = float(sorted_probs[2]) if len(sorted_probs) > 2 else 0.0

    # If top-1 is confident enough there is no uncertainty
    if top1 >= UNCERTAINTY_TOP1_CEILING:
        return False

    margin_1_2 = top1 - top2

    # Two-way split check
    if top2 >= UNCERTAINTY_TOP2_FLOOR and margin_1_2 < UNCERTAINTY_MARGIN_CEIL:
        return True

    # Three-way split check
    if (top2 >= UNCERTAINTY_TOP2_FLOOR
            and top3 >= UNCERTAINTY_TOP3_FLOOR
            and margin_1_2 < UNCERTAINTY_MARGIN_CEIL):
        return True

    return False


def classify(
    pil_image: Image.Image,
    model=None,
    meta=None,
    embed_data=None,
    transform=None,
    device: str = "cpu",
) -> dict:
    """
    Classify a prepared object image using MobileNetV3-Small and the embedding gate.

    Returns a dict with:
    - label: final decision string used by the downstream pipeline
    - confidence: top-1 softmax probability
    - top3: list of (class_name, score) tuples for the top 3 predictions

    Decision rules applied in order:
    1. top-1 confidence below UNKNOWN_THRESHOLD returns "unknown"
    2. top-1 minus top-2 margin below MARGIN_THRESHOLD returns "unknown"
    3. Confidence spread across 2-3 classes returns "uncertain"
    4. Embedding Mahalanobis distance exceeds the fitted gate returns "unknown"
    5. All checks passed, return the top-1 class name
    """
    if model is None:
        model, meta, embed_data, transform, device = load_classifier(device)

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        feat   = model.get_embedding(image_tensor).squeeze(0).cpu().numpy()

    classes = meta["classes"]

    top3_indices = probs.argsort()[::-1][:3]
    top3 = [(classes[i], float(probs[i])) for i in top3_indices]

    top1_label, top1_conf = top3[0]
    top1_idx              = int(top3_indices[0])
    top2_conf             = top3[1][1] if len(top3) > 1 else 0.0

    # Rule 1 - low overall confidence
    if top1_conf < UNKNOWN_THRESHOLD:
        # Check for uncertain spread before returning unknown
        if _is_uncertain(probs):
            return {
                "label":      "uncertain",
                "confidence": top1_conf,
                "top3":       top3,
            }
        return {
            "label":      "unknown",
            "confidence": top1_conf,
            "top3":       top3,
        }

    # Rule 2 - top-1 and top-2 are too close together
    if (top1_conf - top2_conf) < MARGIN_THRESHOLD:
        return {
            "label":      "unknown",
            "confidence": top1_conf,
            "top3":       top3,
        }

    # Rule 3 - uncertainty check for flat distributions that passed rule 1
    if _is_uncertain(probs):
        return {
            "label":      "uncertain",
            "confidence": top1_conf,
            "top3":       top3,
        }

    # Rule 4 - embedding gate using Mahalanobis distance
    if embed_data is not None:
        accepted = _embedding_gate(feat, embed_data)
        if not accepted:
            return {
                "label":      "rejected",
                "confidence": top1_conf,
                "top3":       top3,
            }

    # Rule 5 - all checks passed, return the predicted class
    return {
        "label":      top1_label,
        "confidence": top1_conf,
        "top3":       top3,
    }
