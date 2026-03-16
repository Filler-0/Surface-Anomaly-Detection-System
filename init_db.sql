CREATE TABLE IF NOT EXISTS inspections (
    id SERIAL PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    image_path TEXT NOT NULL,
    crop_path TEXT,
    class_label TEXT,
    class_confidence DOUBLE PRECISION,
    top3_predictions TEXT,
    heatmap_path TEXT,
    result_image_path TEXT,
    anomaly_score DOUBLE PRECISION,
    verdict VARCHAR(50) NOT NULL,
    raw_output TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);