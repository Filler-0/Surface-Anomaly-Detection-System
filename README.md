# Surface Anomaly Detection System

Surface inspection system with a Streamlit UI, PyTorch-based classification + anomaly detection, and PostgreSQL-backed history.

## What The System Does

1. User uploads one or more images in the Streamlit app.
2. A classifier predicts the surface class (`carpet`, `grid`, `leather`, `tile`, `wood`) and top-3 scores.
3. Unsupported/low-confidence cases are marked as `UNSUPPORTED_FORMAT` (no anomaly run).
4. For supported classes, STFPM anomaly detection runs on the image.
5. Final verdict is produced (`NORMAL`, `ANOMALOUS`, or `MANUAL_INSPECTION` for low anomaly certainty).
6. Result metadata and paths are saved to PostgreSQL.
7. Dashboard and History pages read the stored records.

## Current Architecture

```text
Streamlit App (app.py)
  |
  +-- Upload + preview images
  +-- For each image -> run_full_pipeline(...)
        |
        +-- Classification (object_localisation_classification/prepare_image.py)
        |     +-- MobileNetV3-Small classifier (models/mobilenet_v3_small.pt)
        |     +-- Embedding/OOD gate (models/mobilenet_v3_small_embed.pt)
        |
        +-- Verdict gate
        |     +-- unsupported/rejected/uncertain -> UNSUPPORTED_FORMAT
        |
        +-- Anomaly Detection (multi_detector.py via anomalib STFPM)
        |     +-- Per-class checkpoint: models/stfpm_<class>.ckpt
        |
        +-- Final verdict
              +-- NORMAL / ANOMALOUS / MANUAL_INSPECTION

Persistence Layer (db.py + PostgreSQL)
  +-- insert_inspection(...)
  +-- fetch_history(...)

UI Pages (Streamlit)
  +-- app.py (upload + inference)
  +-- pages/dashboard.py (KPIs/charts)
  +-- pages/history.py (filterable history)
```

## Tech Stack

- UI: Streamlit, Altair
- Core language/runtime: Python 3.11
- ML: PyTorch, Torchvision, Anomalib (STFPM)
- Data/processing: NumPy, Pandas, Pillow, Matplotlib, scikit-learn, seaborn
- Database: PostgreSQL + psycopg2
- Config: python-dotenv
- Containerization: Docker, Docker Compose

## Model Files In Repository

All required runtime model files are currently in Git (no separate large-model download step required).

### Classification models

- `models/mobilenet_v3_small.pt`
- `models/mobilenet_v3_small_embed.pt`

### Anomaly models (STFPM)

- `models/stfpm_carpet.ckpt`
- `models/stfpm_grid.ckpt`
- `models/stfpm_leather.ckpt`
- `models/stfpm_tile.ckpt`
- `models/stfpm_wood.ckpt`

Notes:
- `multi_detector.py` prefers `models/stfpm_<class>.ckpt` and falls back to `future_integration/models/new_stfpm/stfpm_<class>.ckpt` if needed.
- Main pipeline mode is currently `MODE = "Classifier"` in `pipeline.py`.

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
|   |-- mobilenet_v3_small.pt
|   |-- mobilenet_v3_small_embed.pt
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

## Run With Docker

1. Build and start:

```bash
docker compose up --build
```

2. Open:

```text
http://localhost:8501
```

Containers/services:
- `app`: Streamlit application
- `db`: PostgreSQL 16

## Environment Variables

Used by the app for PostgreSQL connection:

```env
DB_HOST=db
DB_PORT=5432
DB_NAME=sads_db
DB_USER=postgres
DB_PASSWORD=postgres
```

In Docker Compose these are already set for the `app` container.

## Run Locally (Without Docker)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start PostgreSQL and create the `inspections` table (or run `init_db.sql`).

3. Set environment variables (for example using `.env`).

4. Run the app:

```bash
streamlit run app.py
```

## Database Schema

`init_db.sql` creates one table:

- `inspections`
  - stores image info, classification output, anomaly output, verdict, raw output, and timestamp.

## Authors

- **Zhyldyz Davydova** - [little-hawk](https://github.com/little-hawk)
- **Petr Vasilevskii** - [Filler-0](https://github.com/Filler-0)
- **Alena Bobyleva** - [AlisMori](https://github.com/AlisMori)
