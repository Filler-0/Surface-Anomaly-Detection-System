"""
Subsystem 2 - Object Classification
Loads the fine-tuned CLIP classifier and returns a class label + confidence.
Model: CLIP ViT-L/14 + linear head, trained on MVTec AD 15 classes.

This version stores:
1. classification_model.pt in the subsystem folder
2. OpenAI CLIP pretrained weights in a local "clip_cache" folder
"""

import json
from pathlib import Path

import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import open_clip


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "classification_model.pt"
META_PATH = BASE_DIR / "classification_metadata.json"
CLIP_CACHE_DIR = BASE_DIR / "clip_cache"

MODEL_URL = (
    "https://github.com/Filler-0/Surface-Anomaly-Detection-System/"
    "releases/download/v1.0/classification_model.pt"
)

# ============================================================
# Constants
# ============================================================

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMG_SIZE = 224

UNKNOWN_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15


# ============================================================
# Download helpers
# ============================================================

def download_classifier_if_missing():
    """
    Download the trained classifier checkpoint from GitHub Release
    if it does not already exist locally.
    """
    if MODEL_PATH.exists():
        print("Classification checkpoint found locally.")
        return

    print("\nClassification model not found.")
    print("Downloading classification_model.pt from GitHub Release...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(MODEL_URL, stream=True, timeout=120) as response:
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    print("Classification model downloaded successfully.\n")


def ensure_clip_cache_dir():
    """
    Create the local CLIP cache directory if it does not exist.
    """
    CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Model definition
# ============================================================

class CLIPClassifier(nn.Module):
    """
    Frozen CLIP visual backbone + trainable linear classifier head.
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
        Extract CLIP image features using the frozen visual backbone,
        then classify them using the linear head.
        """
        with torch.no_grad():
            features = self.clip_visual(x).float()

        return self.classifier(features)


# ============================================================
# Loader
# ============================================================

def load_classifier(device="cpu"):
    """
    Load the classifier head and the CLIP backbone.

    Important:
    - The classifier checkpoint is stored locally as classification_model.pt
    - The CLIP pretrained OpenAI weights are cached inside ./clip_cache
    """

    download_classifier_if_missing()
    ensure_clip_cache_dir()

    with open(META_PATH, "r", encoding="utf-8") as file:
        meta = json.load(file)

    print("Loading CLIP backbone...")
    print(f"CLIP cache directory: {CLIP_CACHE_DIR}")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="openai",
        cache_dir=str(CLIP_CACHE_DIR)
    )

    clip_model = clip_model.to(device).eval()

    for param in clip_model.parameters():
        param.requires_grad = False

    model = CLIPClassifier(
        clip_model=clip_model,
        num_classes=meta["num_classes"],
        feature_dim=meta["feature_dim"]
    )

    print("Loading classifier checkpoint...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    print("Subsystem 2 classifier loaded successfully.\n")

    return model, meta, transform, device


# ============================================================
# Inference
# ============================================================

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
        {
            "label": predicted_label_or_unknown,
            "confidence": top1_confidence,
            "top3": [(label, prob), ...]
        }
    """

    if model is None:
        model, meta, transform, device = load_classifier(device)

    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    top3_idx = probs.argsort()[::-1][:3]
    top3 = [(meta["classes"][i], float(probs[i])) for i in top3_idx]

    top1_label, top1_conf = top3[0]
    top2_conf = top3[1][1] if len(top3) > 1 else 0.0

    if top1_conf < UNKNOWN_THRESHOLD or (top1_conf - top2_conf) < MARGIN_THRESHOLD:
        label = "unknown"
    else:
        label = top1_label

    return {
        "label": label,
        "confidence": float(top1_conf),
        "top3": top3,
    }
