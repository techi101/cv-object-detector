# Real-Time Object Detection & Tracking

> A Python-based object detection system using YOLOv8 (PyTorch) and OpenCV that detects and classifies 80 object classes in real-time from webcam feeds, video files, or images. Includes performance benchmarking, confidence analysis, and automated reporting.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?style=flat-square)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=flat-square)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-purple?style=flat-square)](https://docs.ultralytics.com)

---

## What It Does

This system takes any visual input (webcam, video, or image) and:

1. **Detects objects** — Draws bounding boxes around every recognized object (person, car, dog, bottle, etc.)
2. **Classifies objects** — Labels each detection with its class name and confidence score
3. **Tracks counts** — Displays a real-time HUD showing per-class object counts
4. **Measures performance** — Tracks FPS, generates confidence histograms, and writes benchmark reports

It uses **YOLOv8** (You Only Look Once, version 8) — the same deep learning architecture used in autonomous vehicles, security cameras, and industrial quality inspection.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on webcam (live detection)
python detector.py

# Run on a video file
python detector.py --source video.mp4

# Run on an image
python detector.py --source photo.jpg

# Save annotated output
python detector.py --source video.mp4 --save

# Adjust confidence threshold (only show high-confidence detections)
python detector.py --confidence 0.6

# Run performance benchmark on a video
python benchmark.py --source video.mp4

# Run tests
pytest tests/ -v
```

---

## How YOLO Works (Simple Explanation)

Traditional object detection (R-CNN) works in two steps:
1. First, scan the image to find "regions" that might contain objects
2. Then, classify each region separately

**YOLO** does it in **one step** (hence "You Only Look Once"):
1. Divide the image into a grid (e.g., 13×13)
2. Each grid cell simultaneously predicts bounding boxes AND class probabilities
3. Filter out low-confidence predictions

This makes YOLO extremely fast — fast enough for real-time video processing.

### YOLOv8 Architecture
```
Input Image (640×640)
    ↓
[Backbone: CSPDarknet] → Extracts visual features at multiple scales
    ↓
[Neck: PANet/FPN] → Combines features from different scales
    ↓
[Head: Decoupled] → Predicts (bounding box, confidence, class) for each anchor
    ↓
[NMS] → Filters overlapping boxes, keeps the best ones
    ↓
Output: List of (x1, y1, x2, y2, class, confidence)
```

---

## Project Structure

```
cv-object-detector/
├── detector.py          ← Core detection engine (YOLOv8 + OpenCV visualization)
├── benchmark.py         ← Performance analysis (FPS, confidence, per-class stats)
├── tests/
│   └── test_detector.py ← 16 pytest tests (model init, output structure, drawing)
├── results/             ← Auto-generated outputs
│   ├── *_detected.jpg       (annotated images)
│   ├── detected_output.mp4  (annotated video)
│   ├── performance_analysis.png (benchmark charts)
│   └── benchmark_report.txt    (technical report)
├── requirements.txt
└── README.md
```

---

## Features

| Feature | Description |
|:---|:---|
| **Multi-source input** | Webcam, video file, or single image |
| **80 object classes** | Full COCO dataset (person, car, dog, chair, phone, etc.) |
| **Real-time HUD** | FPS counter, object count, per-class breakdown overlay |
| **Confidence filtering** | Adjustable threshold (0.0 – 1.0) via CLI |
| **Model size selection** | nano/small/medium/large/xlarge (`--model n/s/m/l/x`) |
| **Performance benchmarking** | FPS analysis, confidence distribution, detection charts |
| **Automated reporting** | Text-based technical report with full statistics |
| **Video output** | Save annotated video with `--save` flag |

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Deep Learning Model | YOLOv8 (Ultralytics) |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Scientific Computing | NumPy, Matplotlib |
| Testing | pytest |
| CLI | argparse (stdlib) |
