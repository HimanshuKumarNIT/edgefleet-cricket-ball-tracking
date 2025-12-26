```
# Cricket Ball Detection and Tracking – EdgeFleet.AI Assessment

## Overview
This project implements a complete computer vision system to detect and track a cricket ball in videos recorded from a single, fixed camera. The system detects the ball centroid in each visible frame, applies temporal tracking to maintain continuity, and generates both per-frame annotation files and processed videos with trajectory overlays.

---

## System Capabilities
- Cricket ball centroid detection per frame
- Visibility handling using a binary visibility flag
- Motion-based detection with YOLO fallback
- Kalman-filter–based continuous tracking
- Processed video generation with centroid and trajectory overlay
- Per-frame CSV annotation output
- Fully reproducible inference pipeline

---

## Input
- Cricket videos captured from a **single static camera**

---

## Output
1. **Processed Video**
   - MP4 video with ball centroid and trajectory overlay
   - Saved in the `results/` directory

2. **Annotation File**
   - CSV file containing per-frame detections
   - Saved in the `annotations/` directory

### Annotation CSV Format
```csv
frame,x,y,visible
0,512.3,298.1,1
1,518.7,305.4,1
2,-1,-1,0

```

visible = 1 → ball detected

visible = 0 → ball not visible (x = -1, y = -1)

---

Repository Structure
```
code/            # Detection, tracking, inference, utilities
annotations/     # Per-frame CSV annotation files
results/         # Processed videos with trajectory overlay
models/          # Model file required to reproduce results
README.md
requirements.txt
report.pdf

```
---
Setup

Install all required dependencies using
```
pip install -r requirements.txt

```

---
Running Inference

Run the complete detection and tracking pipeline using
```
python -m code.inference.run_pipeline --video path/to/input_video.mp4

```
This command will:

Detect and track the cricket ball

Generate a processed video in results/

Generate a CSV annotation file in annotations/

---

Model Information

A pretrained YOLOv8-based detector is used for cricket ball detection

The model file is included in the repository to allow full reproducibility

No training is performed on the provided evaluation dataset

---

Dataset Usage Declaration

The dataset provided by EdgeFleet.AI is used strictly for testing and evaluation purposes only and is not used for training the model.

---

Assumptions

Input videos are captured from a single fixed camera

The cricket ball may be temporarily occluded or missed due to motion or lighting

The tracking module maintains trajectory continuity during short detection failures

---

Notes

All paths and configuration values are centralized in code/config.py for reproducibility

The pipeline is modular and easy to extend or modify

Detailed modelling decisions, fallback logic, assumptions, and performance improvements are documented in report.pdf