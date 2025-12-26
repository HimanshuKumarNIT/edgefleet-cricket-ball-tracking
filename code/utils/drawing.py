"""
Draw bounding box, centroid, and continuous trajectory.
"""

import cv2
from collections import deque
from code.config import BALL_RADIUS, TRAJECTORY_THICKNESS


class TrajectoryDrawer:
    def __init__(self, max_length=200):
        self.points = deque(maxlen=max_length)

    def update(self, x, y, visible):
        if x >= 0 and y >= 0:
            self.points.append((int(x), int(y)))
        else:
            self.points.append(None)

    def draw(self, frame, bbox=None):
        # Trajectory
        for i in range(1, len(self.points)):
            if self.points[i - 1] is None or self.points[i] is None:
                continue
            cv2.line(
                frame,
                self.points[i - 1],
                self.points[i],
                (255, 0, 0),
                TRAJECTORY_THICKNESS
            )

        # Bounding box (only when detected)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Centroid (prediction or detection)
        if self.points and self.points[-1] is not None:
            cv2.circle(
                frame,
                self.points[-1],
                BALL_RADIUS,
                (0, 0, 255),
                -1
            )

        return frame



