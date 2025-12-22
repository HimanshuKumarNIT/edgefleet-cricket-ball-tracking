"""
Kalman Filter based tracker for cricket ball centroid tracking.
Handles missed detections using motion prediction.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from src.config import (
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE
)


class BallKalmanTracker:
    def __init__(self):
        # State: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Initial uncertainty
        self.kf.P *= 1000.

        # Noise matrices
        self.kf.R = np.eye(2) * KALMAN_MEASUREMENT_NOISE
        self.kf.Q = np.eye(4) * KALMAN_PROCESS_NOISE

        self.initialized = False

    def update(self, x, y, visible):
        """
        Update tracker with detection or predict if missing.

        Returns:
            tuple: (x, y, visible)
        """
        if visible:
            z = np.array([x, y])
            if not self.initialized:
                self.kf.x = np.array([x, y, 0, 0])
                self.initialized = True
            self.kf.predict()
            self.kf.update(z)
            return x, y, 1

        if self.initialized:
            self.kf.predict()
            pred_x, pred_y = self.kf.x[0], self.kf.x[1]
            return float(pred_x), float(pred_y), 0

        return -1, -1, 0
