"""
Main inference pipeline for EdgeFleet Cricket Ball Tracking.

This script:
1. Loads a test video
2. Detects cricket ball using YOLOv8
3. Tracks ball using Kalman Filter
4. Writes per-frame CSV annotations
5. Generates processed video with centroid + trajectory overlay
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config import (
    TEST_VIDEO_DIR,
    RESULTS_DIR,
    ANNOTATION_DIR
)
from src.detection.yolo_detector import YoloBallDetector
from src.tracking.kalman_tracker import BallKalmanTracker
from src.utils.video_io import open_video, create_video_writer
from src.utils.csv_writer import CSVWriter
from src.utils.drawing import TrajectoryDrawer


def run(video_path: Path):
    # -----------------------------
    # Setup
    # -----------------------------
    detector = YoloBallDetector()
    tracker = BallKalmanTracker()
    drawer = TrajectoryDrawer()

    cap, fps, width, height = open_video(video_path)

    output_video_path = RESULTS_DIR / f"{video_path.stem}_processed.mp4"
    writer = create_video_writer(output_video_path, fps, width, height)

    csv_path = ANNOTATION_DIR / f"{video_path.stem}.csv"
    csv_writer = CSVWriter(csv_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -----------------------------
    # Frame-by-frame processing
    # -----------------------------
    frame_idx = 0
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        det_x, det_y, det_visible = detector.detect(frame)

        # Tracking
        trk_x, trk_y, trk_visible = tracker.update(
            det_x, det_y, det_visible
        )

        # Save annotation
        csv_writer.add_record(
            frame_idx,
            trk_x,
            trk_y,
            trk_visible
        )

        # Draw trajectory
        drawer.update(trk_x, trk_y, trk_visible)
        frame = drawer.draw(frame)

        # Write frame
        writer.write(frame)

        frame_idx += 1

    # -----------------------------
    # Cleanup
    # -----------------------------
    cap.release()
    writer.release()
    csv_writer.save()

    print(f"[INFO] Processed video saved to: {output_video_path}")
    print(f"[INFO] Annotations saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EdgeFleet Cricket Ball Tracking Inference"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input test video"
    )
    args = parser.parse_args()

    video_file = Path(args.video)
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    run(video_file)
