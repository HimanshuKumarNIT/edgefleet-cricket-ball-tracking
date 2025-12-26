"""
Kalman tracker with continuous prediction.
Prediction is drawn even when detection is missing.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from code.config import KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE


class BallKalmanTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.kf.P *= 500.0
        self.kf.R = np.eye(2) * KALMAN_MEASUREMENT_NOISE
        self.kf.Q = np.eye(4) * KALMAN_PROCESS_NOISE

        self.initialized = False
        self.MAX_JUMP = 80

    def update(self, centroid, visible):
        # Init
        if not self.initialized:
            if visible:
                x, y = centroid
                self.kf.x = np.array([x, y, 0, 0])
                self.initialized = True
                return x, y, 1
            return -1, -1, 0

        # Always predict
        self.kf.predict()
        pred_x, pred_y = float(self.kf.x[0]), float(self.kf.x[1])

        # Update only if detection is valid
        if visible:
            x, y = centroid
            dist = np.hypot(x - pred_x, y - pred_y)
            if dist < self.MAX_JUMP:
                self.kf.update(np.array([x, y]))
                return float(self.kf.x[0]), float(self.kf.x[1]), 1

        # Prediction continues but visibility = 0
        return pred_x, pred_y, 0



