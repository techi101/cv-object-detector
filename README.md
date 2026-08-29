# Real-Time Object Detection & Tracking

> A full-stack computer vision application using YOLOv8, FastAPI, and OpenCV that detects and classifies 80 object classes in real-time. Features a deployed web interface with live webcam detection and image upload analysis.

[![Live Demo](https://img.shields.io/badge/🔗_Live_Demo-cv--object--detector-00b4d8?style=for-the-badge)](https://cv-object-detector-zrgy.onrender.com)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?style=flat-square)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=flat-square)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-purple?style=flat-square)](https://docs.ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square)](https://docker.com)

---

## What It Does

This system takes any visual input (live webcam, uploaded image, or video file) and:

1. **Detects objects** — Draws bounding boxes around every recognized object (person, car, dog, bottle, etc.)
2. **Classifies objects** — Labels each detection with its class name and confidence score (e.g., `person 92%`)
3. **Tracks counts** — Displays a real-time HUD showing per-class object counts
4. **Measures performance** — Tracks FPS, generates confidence histograms, and writes benchmark reports

It uses **YOLOv8** (You Only Look Once, version 8) — the same deep learning architecture used in autonomous vehicles, security cameras, and industrial quality inspection.

### 🌐 Deployed Web Application

The project includes a **full-stack web application** that runs entirely in the browser:

- **Live Webcam Detection** — Enable your camera and run continuous real-time object detection with bounding boxes and labels drawn directly on the video feed
- **Image Upload Analysis** — Drag & drop any image for instant YOLOv8 analysis with a detailed breakdown of detected objects
- **Modern Dark UI** — Premium design with glassmorphism, animated gradients, and micro-animations

> **Try it live:** [cv-object-detector-zrgy.onrender.com](https://cv-object-detector-zrgy.onrender.com)

---

## Quick Start

### Option 1: Web Application (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web server
uvicorn app:app --reload

# Open http://127.0.0.1:8000 in your browser
```

### Option 2: Command Line (Local)

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

### Option 3: Docker

```bash
# Build the container
docker build -t cv-object-detector .

# Run it
docker run -p 8000:8000 cv-object-detector

# Open http://localhost:8000
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

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Browser (Client)                    │
│  ┌──────────────┐  ┌──────────────────────────────────┐ │
│  │  Webcam API   │  │  Drag & Drop Image Upload        │ │
│  │  (MediaDevices)│  │  (File API)                      │ │
│  └──────┬───────┘  └──────────────┬───────────────────┘ │
│         │                          │                     │
│         └──────────┬───────────────┘                     │
│                    ▼                                     │
│         POST /detect (multipart)                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Server (app.py)                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │  YOLOv8s Model (detector.py)                     │    │
│  │  ┌───────────┐  ┌────────────┐  ┌────────────┐  │    │
│  │  │ Inference  │→ │ Draw Boxes │→ │ Base64 Enc │  │    │
│  │  └───────────┘  └────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                                │
│              JSON Response: {image, counts}              │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
cv-object-detector/
├── app.py               ← FastAPI web server (serves UI + /detect API)
├── detector.py          ← Core detection engine (YOLOv8 + OpenCV visualization)
├── benchmark.py         ← Performance analysis (FPS, confidence, per-class stats)
├── Dockerfile           ← Container config for cloud deployment
├── static/
│   ├── index.html       ← Web UI (dark mode, glassmorphism)
│   ├── style.css        ← Premium stylesheet (animations, responsive)
│   └── script.js        ← Webcam capture, live detection loop, drag & drop
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
| **🌐 Live Web App** | Deployed on Render with a modern dark-mode UI |
| **📷 Browser Webcam** | Real-time continuous detection using browser camera API |
| **📤 Image Upload** | Drag & drop image analysis with detection breakdown |
| **🎯 80 object classes** | Full COCO dataset (person, car, dog, chair, phone, etc.) |
| **📊 Real-time HUD** | FPS counter, object count, per-class breakdown overlay |
| **⚙️ Confidence filtering** | Adjustable threshold (0.0 – 1.0) via CLI |
| **📦 Docker support** | Containerized for one-command cloud deployment |
| **🧪 Tested** | 16 pytest tests covering model init, output structure, drawing |
| **📈 Benchmarking** | FPS analysis, confidence distribution, detection charts |

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Deep Learning Model | YOLOv8s (Ultralytics) |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Web Backend | FastAPI + Uvicorn |
| Web Frontend | HTML, CSS (Glassmorphism), JavaScript |
| Containerization | Docker |
| Deployment | Render (Free Tier) |
| Scientific Computing | NumPy, Matplotlib |
| Testing | pytest |

---

## Deployment

This project is deployed on [Render](https://render.com) using Docker. Every push to `main` triggers an automatic redeploy.

To deploy your own instance:
1. Fork this repository
2. Create a new **Web Service** on Render
3. Connect your GitHub repo
4. Render auto-detects the `Dockerfile` and deploys

---
