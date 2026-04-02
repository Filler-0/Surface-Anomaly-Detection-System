"""
Subsystem 2 - Object Classification
Loads the fine-tuned EfficientNet-B0 classifier and returns a class label + confidence.
Model: EfficientNet-B0 fine-tuned on MVTec AD surface classes.

Supported classes:
- carpet
- grid
- leather
- tile
- wood

Decision rule:
1. Compute per-class probabilities via sigmoid (consistent with BCEWithLogitsLoss training).
2. If max sigmoid probability is below CONFIDENCE_THRESHOLD, return "unknown".
3. Otherwise return the top-1 class label.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np


BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "efficientnet_b0.pt"
META_PATH  = Path(__file__).parent / "classes.json"

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Minimum sigmoid confidence for the top-1 prediction to be accepted.
# Predictions below this threshold are returned as "unknown".
CONFIDENCE_THRESHOLD = 0.80


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned for surface classification.
    Mirrors the architecture defined in the training notebook exactly.
    Base layers frozen initially; full fine-tuning after warm-up.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3,
                 freeze_base: bool = True):
        super().__init__()
        self.base = timm.create_model(
            'efficientnet_b0', pretrained=False,
            num_classes=0, global_pool='avg'
        )
        in_features = self.base.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        return self.classifier(features)


def load_classifier(device: str = "cpu"):
    """
    Load the EfficientNet-B0 classifier and class metadata.

    Returns model, meta, transform, device.
    model     - nn.Module in eval mode
    meta      - dict with classes, num_classes, etc.
    transform - torchvision preprocessing pipeline
    device    - str
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Class metadata not found: {META_PATH}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]

    model = EfficientNetB0Classifier(num_classes=num_classes, dropout=0.3,
                                     freeze_base=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Support both a raw state dict and a checkpoint dict with a "state_dict" key
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return model, meta, transform, device


def classify(
    pil_image: Image.Image,
    model=None,
    meta=None,
    transform=None,
    device: str = "cpu",
) -> dict:
    """
    Classify a prepared object image using EfficientNet-B0.

    Probabilities are computed via sigmoid, consistent with BCEWithLogitsLoss
    training (independent per-class scores; values do not sum to 1).

    Returns a dict with:
    - label:      final decision string — top-1 class name or "unknown"
    - confidence: max sigmoid probability
    - top3:       list of (class_name, score) tuples for the top 3 predictions

    Decision rule:
    1. If max sigmoid probability is below CONFIDENCE_THRESHOLD, return "unknown".
    2. Otherwise return the top-1 class label.
    """
    if model is None:
        model, meta, transform, device = load_classifier(device)

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        # sigmoid: independent per-class probabilities, consistent with
        # BCEWithLogitsLoss used during training
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    classes = meta["classes"]

    top3_indices = probs.argsort()[::-1][:3]
    top3 = [(classes[i], float(probs[i])) for i in top3_indices]

    top1_label, top1_conf = top3[0]

    # Rule 1 — low overall confidence
    if top1_conf < CONFIDENCE_THRESHOLD:
        return {
            "label":      "unknown",
            "confidence": top1_conf,
            "top3":       top3,
        }

    # Rule 2 — all checks passed, return the predicted class
    return {
        "label":      top1_label,
        "confidence": top1_conf,
        "top3":       top3,
    }