"""
Utility functions for video reading and writing.
"""

import cv2


def open_video(video_path):
    """
    Open video file for reading.

    Returns:
        cap (cv2.VideoCapture)
        fps (float)
        width (int)
        height (int)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, width, height


def create_video_writer(output_path, fps, width, height):
    """
    Create video writer for output video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )
    return writer
