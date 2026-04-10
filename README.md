# Surface Anomaly Detection System

Surface inspection application with a Streamlit UI, an EfficientNet-B0 image classifier, STFPM anomaly detection (Anomalib), and PostgreSQL-backed history.

## Current System Behavior

1. Upload one or more images in `app.py` (JPG/JPEG/PNG).
2. `pipeline.py` runs object classification via `object_localisation_classification/prepare_image.py`.
3. If the class is unsupported (`unknown`, `rejected`, or `uncertain`), anomaly detection is skipped and verdict is `UNSUPPORTED_FORMAT`.
4. For supported classes (`carpet`, `grid`, `leather`, `tile`, `wood`), `multi_detector.py` runs STFPM anomaly detection.
5. Final verdict is one of:
   - `NORMAL`
   - `ANOMALOUS`
   - `MANUAL_INSPECTION` (low anomaly certainty)
   - `UNSUPPORTED_FORMAT`
6. Results are stored in PostgreSQL (`inspections` table).
7. Streamlit `Dashboard` and `History` pages read from the same table.

## Architecture

```text
Streamlit UI
  app.py                -> Upload, run pipeline, show per-image result
  pages/dashboard.py    -> KPIs + charts from DB history
  pages/history.py      -> Filterable stored inspections

Core Pipeline
  pipeline.py
    -> Classification (prepare_image.py -> classifier.py)
    -> Verdict gate (unsupported vs supported)
    -> Subprocess anomaly call: python multi_detector.py <class> <temp_dir>
    -> Final verdict + metadata

ML Components
  object_localisation_classification/classifier.py
    -> EfficientNet-B0 classifier checkpoint: models/efficientnet_b0.pt
    -> Class metadata: object_localisation_classification/classes.json

  multi_detector.py
    -> Anomalib STFPM inference per class
    -> Checkpoints: models/stfpm_<class>.ckpt
    -> Fallback path: future_integration/models/new_stfpm/stfpm_<class>.ckpt

Persistence
  db.py + init_db.sql
    -> PostgreSQL table: inspections
```

## Tech Stack

- Python 3.11
- Streamlit
- PyTorch, Torchvision, timm
- Anomalib (STFPM)
- NumPy, Pandas, Pillow
- Matplotlib, scikit-learn, seaborn
- PostgreSQL + psycopg2-binary
- python-dotenv
- Docker + Docker Compose

## Runtime Model Files

### Classification

- `models/efficientnet_b0.pt`
- `object_localisation_classification/classes.json`

### Anomaly Detection (STFPM)

- `models/stfpm_carpet.ckpt`
- `models/stfpm_grid.ckpt`
- `models/stfpm_leather.ckpt`
- `models/stfpm_tile.ckpt`
- `models/stfpm_wood.ckpt`

## Project Structure

```text
Surface-Anomaly-Detection-System/
|-- app.py
|-- pipeline.py
|-- multi_detector.py
|-- db.py
|-- ui_styles.py
|-- requirements.txt
|-- init_db.sql
|-- Dockerfile
|-- docker-compose.yml
|-- .env.example
|-- models/
|   |-- efficientnet_b0.pt
|   |-- stfpm_carpet.ckpt
|   |-- stfpm_grid.ckpt
|   |-- stfpm_leather.ckpt
|   |-- stfpm_tile.ckpt
|   |-- stfpm_wood.ckpt
|-- object_localisation_classification/
|   |-- prepare_image.py
|   |-- classifier.py
|   |-- classes.json
|-- pages/
|   |-- dashboard.py
|   |-- history.py
|-- uploads/
|-- results/
|-- temp_runs/
|-- future_integration/
```

## Environment Variables

Set DB connection variables (same names used by `db.py`):

```env
DB_HOST=db
DB_PORT=5432
DB_NAME=sads_db
DB_USER=postgres
DB_PASSWORD=postgres
```

## Run With Docker

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

Services in `docker-compose.yml`:
- `app`: Streamlit application
- `db`: PostgreSQL 16

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start PostgreSQL and initialize schema:

```bash
psql -U <user> -d <db_name> -f init_db.sql
```

3. Set environment variables (for example via `.env`).

4. Start Streamlit:

```bash
streamlit run app.py
```

## Database Schema

`init_db.sql` creates table `inspections` with fields for:
- uploaded image metadata
- classification result and confidence/top-3
- anomaly score and final verdict
- raw pipeline output
- creation timestamp

## Notes

- `pipeline.py` currently runs with `MODE = "Classifier"`.
- `MODE = "Recognition"` exists for experimental future integration under `future_integration/`.

## Adding a New Class to the Recognition System

### Running the Demo

- Locate resnet18_backbone.pth file in project submission folder
- Paste the file into directory: `syrface-anomaly-detection-system/fututre_integration/models/`
- Locate `pipline.py` file in `syrface-anomaly-detection-system/`
- Change MODE to Recogniser: `MODE = Recogniser`

### Overview

The recognition system uses a frozen ResNet18 backbone to extract feature embeddings, and a multi-prototype bank to classify objects. Adding a new class does **not** require retraining the backbone — it only computes new prototype centroids for the new class and appends them to the bank.

## Authors

- **Zhyldyz Davydova** - [little-hawk](https://github.com/little-hawk)
- **Petr Vasilevskii** - [Filler-0](https://github.com/Filler-0)
- **Alena Bobyleva** - [AlisMori](https://github.com/AlisMori)
