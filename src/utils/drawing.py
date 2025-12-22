"""
Utility functions for drawing ball centroid and trajectory on frames.
"""

import cv2
from collections import deque
from src.config import BALL_RADIUS, TRAJECTORY_THICKNESS


class TrajectoryDrawer:
    def __init__(self, max_length=200):
        self.points = deque(maxlen=max_length)

    def update(self, x, y, visible):
        """
        Update trajectory points.
        """
        if visible:
            self.points.append((int(x), int(y)))
        else:
            self.points.append(None)

    def draw(self, frame):
        """
        Draw centroid and trajectory on frame.
        """
        # Draw trajectory
        for i in range(1, len(self.points)):
            if self.points[i - 1] is None or self.points[i] is None:
                continue
            cv2.line(
                frame,
                self.points[i - 1],
                self.points[i],
                (0, 0, 255),
                TRAJECTORY_THICKNESS
            )

        # Draw current point
        if self.points and self.points[-1] is not None:
            cv2.circle(
                frame,
                self.points[-1],
                BALL_RADIUS,
                (0, 255, 0),
                -1
            )

        return frame
