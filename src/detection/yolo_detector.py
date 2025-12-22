"""
YOLOv8-based cricket ball detector.
Returns the centroid of the detected ball if confidence threshold is met.
"""

from ultralytics import YOLO
import numpy as np
from src.config import YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD


class YoloBallDetector:
    def __init__(self):
        self.model = YOLO(str(YOLO_MODEL_PATH))

    def detect(self, frame):
        """
        Detect cricket ball in a single frame.

        Args:
            frame (np.ndarray): BGR image

        Returns:
            tuple: (x, y, visible)
        """
        results = self.model.predict(
            source=frame,
            conf=YOLO_CONF_THRESHOLD,
            verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            return -1, -1, 0

        boxes = results[0].boxes
        best_box = boxes[boxes.conf.argmax()]

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return float(cx), float(cy), 1
 
