# Surface Anomaly Detection System

A deep-learning based system for **surface anomaly detection and product classification** with an interactive **Streamlit web interface**, **PostgreSQL database**, and **Docker deployment**.

The system allows users to upload an image of a surface, automatically detect the object type, run anomaly detection, and visualize the results.

---

# Project Overview

This project combines multiple machine learning components into a single pipeline:

1. **Object localisation** – crops the relevant region of the image.
2. **Product classification** – identifies the object type.
3. **Anomaly detection** – detects defects on the surface.
4. **Visualization** – displays anomaly heatmaps and prediction masks.
5. **Database storage** – saves inspection history and results.
6. **Web interface** – provides an interactive interface for users.

The system is designed so that **multiple object types and anomaly models can be added later**.

---

# System Architecture

```
User uploads image
        │
        ▼
Object localisation
        │
        ▼
Product classification
        │
        ▼
Anomaly detection
        │
        ▼
Result visualization
        │
        ▼
Results saved to PostgreSQL
        │
        ▼
Dashboard & History pages
```

---

# Features

- Image upload interface
- Automatic object classification
- Surface anomaly detection
- Visualization of anomaly heatmaps
- PostgreSQL inspection history
- KPI dashboard
- Docker deployment
- Modular ML pipeline

---

# Tech Stack

### Frontend
- Streamlit

### Backend
- Python
- PyTorch
- OpenCV
- NumPy

### Database
- PostgreSQL

### Deployment
- Docker
- Docker Compose

---

# Project Structure

```
Surface-Anomaly-Detection-System
│
├── models/                       # ML model weights (NOT stored in Git)
│
├── object_localisation_classification/
│   ├── classifier.py
│   ├── localiser.py
│   ├── prepare_image.py
│   └── classification_metadata.json
│
├── pages/
│   ├── dashboard.py
│   └── history.py
│
├── uploads/                      # Uploaded images
├── results/                      # Generated visualizations
├── temp_runs/                    # Temporary model outputs
│
├── app.py                        # Streamlit main app
├── pipeline.py                   # ML pipeline
├── wood_detector.py              # Anomaly detection
├── db.py                         # PostgreSQL connection
├── ui_styles.py                  # UI styling
│
├── requirements.txt
├── init_db.sql
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

---

# Required Model Files

Large model files are **not included in the repository**.

Before running the system, create a folder:

```
models/
```

and place the following files inside:

```
models/
├── classification_model.pt
├── open_clip_model.safetensors
├── stfpm.ckpt
```

These files are provided separately.

---

# Running the Project with Docker

### 1 Install Docker Desktop

Download and install:

https://www.docker.com/products/docker-desktop/

Make sure Docker is running.

---

### 2 Clone the repository

```bash
git clone https://github.com/Filler-0/Surface-Anomaly-Detection-System.git
cd Surface-Anomaly-Detection-System
```

---

### 3 Add model files

Create the folder:

```
models/
```

and place the model weights inside.

---

### 4 Start the system

Run:

```bash
docker compose up --build
```

The first build may take several minutes.

---

### 5 Open the application

After startup open:

```
http://localhost:8501
```

---

# Application Pages

## Upload Page

Allows the user to upload an image and run anomaly detection.

Displays:

- original image
- anomaly heatmap
- predicted mask
- classification results

---

## Dashboard

Displays system statistics:

- total inspections
- anomaly rate
- normal vs anomalous distribution
- average anomaly score

---

## History

Shows previously processed inspections stored in the database.

Includes:

- image previews
- prediction results
- anomaly score
- timestamps

---

# Database

The system automatically initializes a PostgreSQL database using:

```
init_db.sql
```

The database stores inspection metadata and prediction results.

---

# Environment Variables

Database configuration:

```
DB_HOST=db
DB_PORT=5432
DB_NAME=sads_db
DB_USER=postgres
DB_PASSWORD=postgres
```

These variables are configured automatically by Docker.

---

# Development Without Docker

If you want to run the project locally:

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start PostgreSQL

Ensure PostgreSQL is running locally.

### Run Streamlit

```bash
streamlit run app.py
```

---

# Future Improvements

Possible future extensions:

- support for multiple product types
- multiple anomaly detection models
- model registry
- improved visualization
- GPU inference
- model versioning

---

# Notes About Model Files

Large model files are excluded from Git because they significantly increase repository size.

Instead they are mounted into the container using Docker volumes:

```
./models:/app/models
```

This keeps the repository lightweight while still allowing full functionality.

---

# Authors
*   **Zhyldyz Davydova** - [little-hawk](https://github.com/little-hawk)
*   **Petr Vasilevskii** - [Filler-0](https://github.com/Filler-0)
*   **Alena Bobyleva**   - [AlisMori](https://github.com/AlisMori)
