"""
Subsystem 2 - Object Classification
Loads the fine-tuned CLIP classifier and returns a class label + confidence.
Model: CLIP ViT-L/14 + linear head, trained on MVTec AD 15 classes.
"""

import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import open_clip
import requests
from safetensors.torch import load_file

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "classification_model.pt"
CLIP_MODEL_PATH = MODELS_DIR / "open_clip_model.safetensors"
META_PATH = Path(__file__).parent / "classification_metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing classifier model: {MODEL_PATH}")

if not CLIP_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing CLIP backbone: {CLIP_MODEL_PATH}")

MODEL_URL = "https://github.com/Filler-0/Surface-Anomaly-Detection-System/releases/download/v1.0/classification_model.pt"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMG_SIZE = 224

UNKNOWN_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15


def download_model_if_missing():
    """
    Download model from GitHub Release if it does not exist locally.
    """
    if MODEL_PATH.exists():
        return

    print("\nClassification model not found.")
    print("Downloading model from GitHub Release (~1.1GB)...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("Model downloaded successfully.\n")


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes, feature_dim, dropout=0.2):
        super().__init__()

        self.clip_visual = clip_model.visual

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.clip_visual(x).float()

        return self.classifier(features)


def load_classifier(device="cpu"):
    download_model_if_missing()

    with open(META_PATH) as f:
        meta = json.load(f)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained=None
    )

    state_dict = load_file(str(CLIP_MODEL_PATH))
    clip_model.load_state_dict(state_dict)
    clip_model = clip_model.to(device).eval()

    for p in clip_model.parameters():
        p.requires_grad = False

    model = CLIPClassifier(
        clip_model,
        meta["num_classes"],
        meta["feature_dim"]
    )

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])

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
    """

    if model is None:
        model, meta, transform, device = load_classifier(device)

    img_t = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(img_t)

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
        "confidence": top1_conf,
        "top3": top3,
    }
