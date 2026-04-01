"""
Prototype Recognition - Subsystem entry point.
Mirrors the inference logic from add_new_class.ipynb exactly.

Input : PIL.Image
Output: dict with label, confidence, top3
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

BASE_DIR        = Path(__file__).resolve().parent
MODELS_DIR      = BASE_DIR / "models"
BACKBONE_PATH   = MODELS_DIR / "resnet18_backbone.pth"
BANK_PATH       = MODELS_DIR / "bank_resnet18_multi_proto.pth"

IMG_SIZE             = 224
EMBEDDING_DIM        = 128
DROPOUT              = 0.2
CONFIDENCE_THRESHOLD = 0.98
MARGIN_THRESHOLD     = 0.05

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------
# Exact copy of EmbeddingHead from notebook
# -----------------------------------------------------------------------
class EmbeddingHead(nn.Module):
    def __init__(self, in_features, embedding_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


# -----------------------------------------------------------------------
# Exact copy of PrototypeRecognizer from notebook
# -----------------------------------------------------------------------
class PrototypeRecognizer(nn.Module):
    def __init__(self, num_classes, embedding_dim=128, dropout=0.2):
        super().__init__()
        import torchvision.models as models
        bb = models.resnet18(weights=None)
        self.features       = nn.Sequential(*list(bb.children())[:-1])
        in_feat             = self._probe()
        self.embedding_head = EmbeddingHead(in_feat, embedding_dim, dropout)
        self.classifier     = nn.Linear(embedding_dim, num_classes)

    def _probe(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            return self.features(dummy).view(1, -1).shape[1]

    def forward(self, x, return_embedding=False):
        feat = self.features(x).view(x.size(0), -1)
        emb  = self.embedding_head(feat)
        if return_embedding:
            return emb
        return emb, self.classifier(emb)


# -----------------------------------------------------------------------
# Exact copy of PrototypeBank from notebook
# -----------------------------------------------------------------------
class PrototypeBank:
    def __init__(self, embedding_dim, class_to_idx, idx_to_class,
                 confidence_threshold=CONFIDENCE_THRESHOLD,
                 margin_threshold=MARGIN_THRESHOLD,
                 n_prototypes=3):
        self.embedding_dim        = embedding_dim
        self.class_to_idx         = class_to_idx
        self.idx_to_class         = idx_to_class
        self.confidence_threshold = confidence_threshold
        self.margin_threshold     = margin_threshold
        self.n_prototypes         = n_prototypes
        self.prototypes           = {}
        self.per_class_threshold  = {idx: confidence_threshold for idx in idx_to_class}

    @torch.no_grad()
    def predict(self, embedding):
        emb = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
        scores = {}
        for idx, protos in self.prototypes.items():
            sims = torch.mm(emb, protos.T)
            scores[idx] = sims.max().item()

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_idx, top_score = sorted_scores[0]
        top_name = self.idx_to_class[top_idx]
        top3     = [(self.idx_to_class[i], s) for i, s in sorted_scores[:3]]

        thresh = self.per_class_threshold.get(top_idx, self.confidence_threshold)
        if top_score < thresh:
            return 'unknown', top_score, top3
        if len(sorted_scores) > 1:
            margin = top_score - sorted_scores[1][1]
            if margin < self.margin_threshold:
                return 'ambiguous', top_score, top3
        return top_name, top_score, top3

    @classmethod
    def load(cls, path):
        d    = torch.load(path, map_location='cpu', weights_only=False)
        bank = cls(
            embedding_dim        = d['embedding_dim'],
            class_to_idx         = d['class_to_idx'],
            idx_to_class         = d['idx_to_class'],
            confidence_threshold = d['confidence_threshold'],
            margin_threshold     = d['margin_threshold'],
            n_prototypes         = d.get('n_prototypes', 3),
        )
        bank.prototypes          = d['prototypes']
        bank.per_class_threshold = d['per_class_threshold']
        return bank


# -----------------------------------------------------------------------
# Module-level singletons
# -----------------------------------------------------------------------
_model = None
_bank  = None


def _init():
    global _model, _bank
    if _model is not None:
        return

    ckpt = torch.load(BACKBONE_PATH, map_location=DEVICE, weights_only=False)
    _model = PrototypeRecognizer(
        num_classes   = ckpt.get("num_classes", 5),
        embedding_dim = ckpt.get("embedding_dim", EMBEDDING_DIM),
        dropout       = DROPOUT,
    ).to(DEVICE)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    for p in _model.parameters():
        p.requires_grad = False

    _bank = PrototypeBank.load(BANK_PATH)


def recognize(pil_image: Image.Image) -> dict:
    _init()

    tensor = _transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = _model(tensor, return_embedding=True).squeeze(0).cpu()

    label, confidence, top3 = _bank.predict(emb)

    return {
        "label":      label,
        "confidence": confidence,
        "top3":       top3,
    }