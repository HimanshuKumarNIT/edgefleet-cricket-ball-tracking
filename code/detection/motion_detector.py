"""
Motion-based candidate detector for cricket ball.

Purpose:
- Detect small fast-moving objects using frame differencing
- Generate candidate centroid + bbox
- Robust to blur and tiny object size
"""

import cv2
import numpy as np


class MotionBallDetector:
    def __init__(
        self,
        min_area=15,
        max_area=400,
        min_radius=2,
        max_radius=12
    ):
        self.prev_gray = None
        self.min_area = min_area
        self.max_area = max_area
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, frame):
        """
        Returns:
            bbox: (x1, y1, x2, y2) or None
            centroid: (cx, cy) or None
            visible: 0 or 1
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None, None, 0

        # Frame difference
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3)))

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_candidate = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            if area > best_area:
                best_area = area
                x, y, w, h = cv2.boundingRect(cnt)
                best_candidate = (x, y, x + w, y + h)

        if best_candidate is None:
            return None, None, 0

        x1, y1, x2, y2 = best_candidate
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return (x1, y1, x2, y2), (cx, cy), 1
