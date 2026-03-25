"""
Subsystem 2 - Object Classification
Loads the fine-tuned CLIP classifier and returns a class label + confidence.
Model: CLIP ViT-L/14 + linear head, trained on MVTec AD classes.

Supported downstream anomaly classes:
- bottle
- carpet
- grid
- tile
- wood

Any other predicted object is forced to "unknown" because the anomaly
detection subsystem does not support it.
"""

import json
from pathlib import Path

import open_clip
import requests
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "classification_model.pt"
CLIP_MODEL_PATH = MODELS_DIR / "open_clip_model.safetensors"
META_PATH = Path(__file__).parent / "classification_metadata.json"

MODEL_URL = (
    "https://github.com/Filler-0/Surface-Anomaly-Detection-System/"
    "releases/download/v1.0/classification_model.pt"
)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMG_SIZE = 224

UNKNOWN_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15

ALLOWED_CLASSES = {"bottle", "carpet", "grid", "tile", "wood"}


def download_model_if_missing():
    """
    Download model from GitHub Release if it does not exist locally.
    """
    if MODEL_PATH.exists():
        return

    print("\nClassification model not found.")
    print("Downloading model from GitHub Release (~1.1GB)...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(MODEL_URL, stream=True) as response:
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    print("Model downloaded successfully.\n")


class CLIPClassifier(nn.Module):
    """
    CLIP visual backbone with a linear classification head.
    """

    def __init__(self, clip_model, num_classes, feature_dim, dropout=0.2):
        super().__init__()

        self.clip_visual = clip_model.visual

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        """
        Extract CLIP visual features and classify them.
        """
        with torch.no_grad():
            features = self.clip_visual(x).float()

        logits = self.classifier(features)
        return logits


def load_classifier(device="cpu"):
    """
    Load the classifier, metadata, and preprocessing pipeline.
    """
    download_model_if_missing()

    if not CLIP_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing CLIP backbone: {CLIP_MODEL_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing classifier metadata: {META_PATH}")

    with open(META_PATH, "r", encoding="utf-8") as file:
        meta = json.load(file)

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained=None
    )

    clip_state_dict = load_file(str(CLIP_MODEL_PATH))
    clip_model.load_state_dict(clip_state_dict)
    clip_model = clip_model.to(device).eval()

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    model = CLIPClassifier(
        clip_model=clip_model,
        num_classes=meta["num_classes"],
        feature_dim=meta["feature_dim"]
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    return model, meta, transform, device


def classify(
    pil_image: Image.Image,
    model=None,
    meta=None,
    transform=None,
    device="cpu"
) -> dict:
    """
    Classify a cropped object image.

    Returns:
    - final label used by the downstream pipeline
    - confidence of the top-1 prediction
    - top3 raw predictions from the classifier

    Final decision rules:
    1. If top-1 confidence is too low, return "unknown".
    2. If top-1 and top-2 are too close, return "unknown".
    3. If top-1 class is not supported by the anomaly detector, return "unknown".
    4. Otherwise return the top-1 class.
    """

    if model is None:
        model, meta, transform, device = load_classifier(device)

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top3_indices = probabilities.argsort()[::-1][:3]
    top3 = [(meta["classes"][i], float(probabilities[i])) for i in top3_indices]

    top1_label, top1_conf = top3[0]
    top2_conf = top3[1][1] if len(top3) > 1 else 0.0

    if top1_conf < UNKNOWN_THRESHOLD:
        final_label = "unknown"
    elif (top1_conf - top2_conf) < MARGIN_THRESHOLD:
        final_label = "unknown"
    elif top1_label not in ALLOWED_CLASSES:
        final_label = "unknown"
    else:
        final_label = top1_label

    return {
        "label": final_label,
        "confidence": top1_conf,
        "top3": top3,
    }