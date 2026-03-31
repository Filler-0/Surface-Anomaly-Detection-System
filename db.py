import os

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def test_connection():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT version();")
    version = cur.fetchone()
    cur.close()
    conn.close()
    return version


def insert_inspection(
    image_name,
    image_path,
    class_label,
    class_confidence,
    top3_predictions,
    heatmap_path,
    result_image_path,
    anomaly_score,
    verdict,
    raw_output,
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO inspections (
            image_name,
            image_path,
            class_label,
            class_confidence,
            top3_predictions,
            heatmap_path,
            result_image_path,
            anomaly_score,
            verdict,
            raw_output
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            image_name,
            image_path,
            class_label,
            class_confidence,
            top3_predictions,
            heatmap_path,
            result_image_path,
            anomaly_score,
            verdict,
            raw_output,
        ),
    )
    inspection_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return inspection_id


def fetch_history():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT
            id,
            image_name,
            image_path,
            class_label,
            class_confidence,
            top3_predictions,
            heatmap_path,
            result_image_path,
            anomaly_score,
            verdict,
            raw_output,
            created_at
        FROM inspections
        ORDER BY created_at DESC
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
