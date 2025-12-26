"""
YOLOv8-based cricket ball detector.
Returns bounding box + centroid + visibility.
"""

from ultralytics import YOLO
from code.config import YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD


class YoloBallDetector:
    def __init__(self):
        self.model = YOLO(str(YOLO_MODEL_PATH))
        self.MIN_AREA = 30
        self.MAX_AREA = 3000

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=YOLO_CONF_THRESHOLD,
            verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            return None, None, 0

        candidates = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)

            if self.MIN_AREA <= area <= self.MAX_AREA:
                candidates.append((box.conf.item(), x1, y1, x2, y2))

        if not candidates:
            return None, None, 0

        _, x1, y1, x2, y2 = max(candidates, key=lambda x: x[0])

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        bbox = (int(x1), int(y1), int(x2), int(y2))
        centroid = (float(cx), float(cy))

        return bbox, centroid, 1



 
