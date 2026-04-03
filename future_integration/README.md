# Adding a New Class to the Recognition System

## Running the Demo

Locate resnet18_backbone.pth file in project submission folder

Paste the file into directory:

syrface-anomaly-detection-system/fututre_integration/models/

Locate pipline.py file in syrface-anomaly-detection-system/

Change MODE to Recogniser:

MODE = Recogniser

## Overview

The recognition system uses a frozen ResNet18 backbone to extract feature embeddings, and a multi-prototype bank to classify objects. Adding a new class does **not** require retraining the backbone — it only computes new prototype centroids for the new class and appends them to the bank.

Two files may be updated after this process:
- `bank_resnet18_multi_proto.pth` — updated with new class prototypes
- `stfpm_<classname>.ckpt` — new anomaly detection model for the new class

The backbone `resnet18_backbone.pth` is **never modified**.

---

## Prerequisites

- Google Colab account (GPU runtime recommended: Runtime → Change runtime type → T4 GPU)
- Google Drive mounted in Colab
- Images of the new class in MVTec structure (see below)
- The two model files downloaded from the repository

### Dataset Structure

Your new class images must follow MVTec layout:
```
dataset/
└── <new_class_name>/
    ├── train/
    │   └── good/
    │       ├── image1.jpg
    │       └── ...
    └── test/
        └── good/
            ├── image2.jpg
            └── ...
    └── ground_truth/
```
Recommended number of images minimum 200, optimal - 400 images. For testing at least 20 defact images

---

## Step 1 — Download the Required Files

From the repository, download these two files to your local machine:

| File | Location |
|---|---|
| Notebook | `future_integration/training/add_new_class.ipynb` |
| Prototype bank | `future_integration/models/bank_resnet18_multi_proto.pth` |
| Backbone | `future_integration/models/resnet18_backbone.pth` |

If you cannot find the "resnet18_backbone.pth", please locate it in provided zip file, and paste inside:

future_integration/models/

Upload all three to your Google Drive before opening Colab.

---

## Step 2 — Run the Notebook

1. Open `add_new_class.ipynb` in Google Colab
2. In **STEP 2 – Configuration**, set the following paths:
```python
DATASET_ROOT    = '/content/drive/MyDrive/...'   # folder containing your new class images
CHECKPOINT_DIR  = '/content/drive/MyDrive/...'   # folder where the two .pth files are saved
STFPM_MODEL_DIR = '/content/drive/MyDrive/...'   # folder where the anomaly model will be saved
```

3. Run all cells from top to bottom
4. When prompted, enter the new class name — it must match the subfolder name in your dataset exactly

The notebook will:
- Extract embeddings for all new class images using the frozen backbone
- Compute K-Means centroids and append them to the bank
- Calibrate a confidence threshold for the new class
- Train an STFPM anomaly detection model for the new class
- Save both updated files back to your Drive

---

## Step 3 — Download the Updated Files from Drive

Once the notebook has finished, download from your Google Drive:

| File | What changed |
|---|---|
| `bank_resnet18_multi_proto.pth` | Now includes prototypes for the new class |
| `stfpm_<new_class_name>.ckpt` | New anomaly detection model |

The backbone `resnet18_backbone.pth` has **not changed** — do not re-download or replace it.

---

## Step 4 — Update the Repository

Replace the old bank file in the repository:
```
future_integration/models/bank_resnet18_multi_proto.pth   ← replace with updated file
```

Copy the new anomaly model into:
```
future_integration/models/stfpm/stfpm_<new_class_name>.ckpt
```

---

## Step 5 — Switch the Pipeline to Recognition Mode

Open `pipeline.py` and change the mode flag:
```python
# -----------------------------------------------------------------------
MODE = "Recognition"   # "Classifier"  or  "Recognition"
# -----------------------------------------------------------------------
```

Then run the program as normal.

---

## Classifier vs Recognition Mode

|                        | Classifier                     | Recognition |
|---                     |---                             |---|
| Add new classes        | Fixed architecture             |  Supported |
| OOD detection          | Mahalanobis distance (strict)  | Prototype similarity (flexible) |
| Misclassification rate | Lower                          | Higher |
| Use case               | Closed, known class set        | Expanding class set |

> **Note:** The recognition mode trades some classification strictness for extensibility. This is expected behaviour — the confidence threshold can be tuned per class in the notebook if needed.

---

## Demo

A `bottle` class is included in the repository as a working example of the full add-new-class workflow.

If you want to test the demo:

1. Copy paste the resnet18_backbone.pth file from provided models to the feature_integration/models directory
2. Change the mode in pipline.py 

Open `pipeline.py` and change the mode flag:
```python
# -----------------------------------------------------------------------
MODE = "Recognition"   # "Classifier"  or  "Recognition"
# -----------------------------------------------------------------------
```