# YOLOv8 Object Detection — CLI and Web App

> A full-stack computer vision application using a pretrained YOLOv8 model, FastAPI, and OpenCV that detects and classifies the 80 COCO object classes. Runs in real time locally on a laptop CPU (about 13 FPS with the nano model) and ships a deployed web interface with image upload and webcam detection.

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
3. **Counts objects** — Displays a HUD with per-class object counts for the current frame
4. **Measures performance** — Tracks FPS, generates confidence histograms, and writes benchmark reports

It uses **YOLOv8** (You Only Look Once, version 8) with Ultralytics' COCO-pretrained weights, used as released (no training or fine-tuning).

> **No tracking.** Objects are detected independently in every frame. Counts are
> per frame; a video summary adds them up across frames, so one bus visible for
> 60 frames is counted 60 times. Persistent object IDs would need a tracker
> (e.g. Ultralytics `model.track` with ByteTrack), which is not implemented.

### 🌐 Deployed Web Application

The project includes a **full-stack web application** that runs entirely in the browser:

- **Live Webcam Detection** — Enable your camera and run continuous detection: each frame is uploaded, annotated on the server, and shown with boxes and labels. The frame rate is limited by the server round trip (see [Performance](#performance-measured))
- **Image Upload Analysis** — Drag & drop any image for instant YOLOv8 analysis with a detailed breakdown of detected objects
- **Modern Dark UI** — Premium design with glassmorphism, animated gradients, and micro-animations

> **Try it live:** [cv-object-detector-zrgy.onrender.com](https://cv-object-detector-zrgy.onrender.com)
> The free Render instance sleeps when idle (first load ≈ 1 minute) and runs inference on a small shared CPU, so each detection takes about a minute there. It is a functional demo, not real-time. Run it locally for real-time speed.

---

## Quick Start

### Option 1: Web Application (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web server
uvicorn app:app --reload

# Open http://127.0.0.1:8000 in your browser
# (MODEL_SIZE=n uvicorn app:app selects the faster nano model; default is small)
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

Two-stage detectors (the R-CNN family) work in two steps:
1. First, propose "regions" that might contain objects
2. Then, classify each region separately

**YOLO** does it in **one pass** (hence "You Only Look Once"): a single network
predicts boxes and classes for the whole image at once, which makes it fast.

### YOLOv8 Architecture
```
Input image, letterboxed to 640 on the long side (grey padding, value 114)
    ↓
[Backbone: CSPDarknet-style, C2f blocks] → features at strides 8, 16, 32
    ↓
[Neck: FPN/PAN] → combines fine and coarse features across scales
    ↓
[Head: decoupled, anchor-free] → at every position of three grids
     (80×80, 40×40, 20×20 for a 640×640 input = 8,400 positions):
     4 box values + 80 class scores  → output tensor 84 × 8,400
    ↓
[Confidence filter (0.4 here) + NMS (IoU 0.7)] → removes low scores and duplicates
    ↓
Output: list of (x1, y1, x2, y2, class, confidence) in original-image pixels
```

YOLOv8 is **anchor-free** (it predicts distances from each grid point to the box
edges, rather than adjusting preset anchor boxes) and has **no separate
objectness score**: the best class score is the confidence.

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
│   ├── test_detector.py ← 20 tests (model init, output structure, drawing, known image)
│   └── test_api.py      ← 5 tests (status codes for good, bad and oversized uploads)
├── results/             ← Created when you run the tools (not committed)
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
| **📷 Browser Webcam** | Continuous frame-by-frame detection using the browser camera API |
| **📤 Image Upload** | Drag & drop image analysis with detection breakdown |
| **🎯 80 object classes** | Full COCO dataset (person, car, dog, chair, phone, etc.) |
| **📊 HUD overlay** | FPS counter, object count, per-class breakdown for the current frame |
| **⚙️ Confidence filtering** | Adjustable threshold (0.0 – 1.0) via CLI |
| **📦 Docker support** | Containerized for one-command cloud deployment |
| **🧪 Tested** | 25 pytest tests: model init, output structure, drawing, a known image, and API status codes |
| **📈 Benchmarking** | FPS analysis, confidence distribution, detection charts |

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Deep Learning Model | YOLOv8s in the web app, YOLOv8n for CLI/benchmark/tests (Ultralytics, COCO-pretrained) |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Web Backend | FastAPI + Uvicorn |
| Web Frontend | HTML, CSS (Glassmorphism), JavaScript |
| Containerization | Docker |
| Deployment | Render (Free Tier) |
| Scientific Computing | NumPy, Matplotlib |
| Testing | pytest |

---

## Performance (measured)

Measured on 26 Sep 2026. Laptop: CPU only, 8 threads, PyTorch 2.13. Image: Ultralytics' `bus.jpg` (810×1080).

| Setting | Result |
|---|---|
| YOLOv8n, single image, median of warm runs | ~77–80 ms (~13 FPS) |
| YOLOv8s, single image, median of warm runs | ~169 ms |
| YOLOv8n, 60-frame 640×480 video, whole loop (read + detect + draw) | 12.2 FPS average |
| Deployed on Render free tier, `POST /detect`, warm | ~57–65 s per image |
| Deployed on Render free tier, first page load after sleep | ~56 s |

Model sizes: YOLOv8n has 3.16 M parameters (6.5 MB), YOLOv8s 11.17 M (22.6 MB).
On bus.jpg at confidence 0.4, nano finds the bus and 3 people; small finds the bus and 4.

**Accuracy is not measured in this project.** The weights are Ultralytics'
COCO-pretrained releases; Ultralytics reports COCO val mAP50-95 of about 37 (n)
and 45 (s). `benchmark.py` measures speed and confidence, not accuracy.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **Deployed speed** | About a minute per image on Render's free CPU; live webcam mode there updates roughly once a minute. |
| **No tracking** | Counts are per frame; video summaries add detections across frames. |
| **80 COCO classes only** | Pretrained weights, no fine-tuning; objects outside COCO are not detected. |
| **No accuracy evaluation** | No mAP / precision / recall measured on any labelled data. |
| **Server-side drawing** | The API returns a re-encoded base64 image (~110 KB) instead of box coordinates, which is heavy for live video. |
| **HEIC photos** | Accepted by the picker but only decodable in browsers that support HEIC (mainly Safari). |

---

## Deployment

This project is deployed on [Render](https://render.com) using Docker. Every push to `main` triggers an automatic redeploy.

To deploy your own instance:
1. Fork this repository
2. Create a new **Web Service** on Render
3. Connect your GitHub repo
4. Render auto-detects the `Dockerfile` and deploys

---
