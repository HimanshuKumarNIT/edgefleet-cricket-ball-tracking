"""
Main inference pipeline – motion first + YOLO fallback + continuous tracking
"""

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

from code.config import RESULTS_DIR, ANNOTATION_DIR
from code.detection.motion_detector import MotionBallDetector
from code.detection.yolo_detector import YoloBallDetector
from code.tracking.kalman_tracker import BallKalmanTracker
from code.utils.video_io import open_video, create_video_writer
from code.utils.csv_writer import CSVWriter
from code.utils.drawing import TrajectoryDrawer


def run(video_path: Path):
    motion_detector = MotionBallDetector()
    yolo_detector = YoloBallDetector()
    tracker = BallKalmanTracker()
    drawer = TrajectoryDrawer()

    cap, fps, width, height = open_video(video_path)

    writer = create_video_writer(
        RESULTS_DIR / f"{video_path.stem}_processed.mp4",
        fps, width, height
    )

    csv_writer = CSVWriter(
        ANNOTATION_DIR / f"{video_path.stem}.csv"
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        # Motion first
        bbox, centroid, visible = motion_detector.detect(frame)

        # YOLO fallback
        if not visible:
            bbox, centroid, visible = yolo_detector.detect(frame)

        trk_x, trk_y, trk_visible = tracker.update(centroid, visible)

        csv_writer.add_record(
            frame_idx,
            trk_x,
            trk_y,
            trk_visible
        )

        drawer.update(trk_x, trk_y, trk_visible)
        frame = drawer.draw(frame, bbox if visible else None)

        writer.write(frame)

    cap.release()
    writer.release()
    csv_writer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    run(Path(args.video))


