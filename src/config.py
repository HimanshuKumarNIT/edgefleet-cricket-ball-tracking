# code/config.py
"""
Central configuration file for EdgeFleet Cricket Ball Tracking project.
All paths and hyperparameters are defined here to ensure reproducibility.
"""

from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TEST_VIDEO_DIR = DATA_DIR / "test_videos"

MODEL_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_PATH = MODEL_DIR / "yolov8_cricket_ball.pt"

ANNOTATION_DIR = PROJECT_ROOT / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create output directories if they do not exist
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Detection Parameters
# -----------------------------
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45

# -----------------------------
# Tracking Parameters
# -----------------------------
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 1e-1

# -----------------------------
# Visualization Parameters
# -----------------------------
BALL_RADIUS = 6
TRAJECTORY_THICKNESS = 2
