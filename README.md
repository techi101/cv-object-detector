# YOLOv8 Object Detection — CLI and Web App

> A full-stack computer vision application using a pretrained YOLOv8 model, FastAPI, OpenCV and ONNX Runtime Web that detects and classifies the 80 COCO object classes. The web app runs the model **in the visitor's browser** (about 0.24 s per image on a laptop), with the server as a fallback. Accuracy is measured on COCO val2017: **mAP50-95 36.5** for the browser model.

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

A **"Run on: This device / Server"** switch picks where the model runs:

- **This device (default)** — the page downloads YOLOv8n as an ONNX file (12.5 MB, once) and runs it with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) in WebAssembly. Images never leave the device. Pre- and post-processing (letterbox, confidence filter, per-class NMS) are re-implemented in `static/browser_detector.js` and tested against Ultralytics.
- **Server** — the image is uploaded to `POST /detect` and YOLOv8s runs on the server with PyTorch. Used automatically if the browser cannot load the model.

Features: image upload with a per-class breakdown and timing, live webcam detection with an FPS readout, and a dark responsive UI.

> **Try it live:** [cv-object-detector-zrgy.onrender.com](https://cv-object-detector-zrgy.onrender.com)
> The free Render instance sleeps when idle (first page load ≈ 1 minute). After that, on-device detection is fast; the "Server" option still takes about a minute per image on Render's small shared CPU.

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

**On-device path (default):**

```
Browser
  photo / webcam frame
    → browser_detector.js: letterbox to 640×640 (grey 114), RGBA → Float32 CHW tensor
    → ONNX Runtime Web (WebAssembly, up to 4 threads): yolov8n.onnx → 1×84×8400
    → browser_detector.js: confidence ≥ 0.4, per-class NMS (IoU 0.7), undo letterbox
    → draw boxes on a canvas + per-class counts
Server only serves files (index.html, JS, the .onnx model) and adds the
COOP/COEP headers that let the browser use several threads.
```

**Server path ("Server" switch, and the CLI):**

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
├── eval_map.py          ← Accuracy: mAP on COCO val2017 images (downloads to datasets/)
├── eval/map_results.json ← Saved accuracy results
├── Dockerfile           ← Container config for cloud deployment
├── static/
│   ├── index.html       ← Web UI (dark mode, engine switch)
│   ├── style.css        ← Stylesheet (animations, responsive)
│   ├── script.js        ← Upload, webcam, live loop; device and server engines
│   ├── browser_detector.js ← In-browser pre/post-processing (letterbox, NMS, drawing)
│   └── models/yolov8n.onnx ← YOLOv8n exported to ONNX for the browser
├── tests/
│   ├── test_detector.py ← 20 tests (model init, output structure, drawing, known image)
│   ├── test_api.py      ← 5 tests (status codes for good, bad and oversized uploads)
│   └── test_browser_detector.py ← 2 tests (JS post-processing vs Ultralytics on the same ONNX)
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
| **⚡ In-browser inference** | YOLOv8n via ONNX Runtime Web; no upload, ~0.24 s per image on a laptop |
| **📷 Browser Webcam** | Continuous frame-by-frame detection using the browser camera API |
| **📤 Image Upload** | Drag & drop image analysis with detection breakdown |
| **🎯 80 object classes** | Full COCO dataset (person, car, dog, chair, phone, etc.) |
| **📊 HUD overlay** | FPS counter, object count, per-class breakdown for the current frame |
| **⚙️ Confidence filtering** | Adjustable threshold (0.0 – 1.0) via CLI |
| **📦 Docker support** | Containerized for one-command cloud deployment |
| **🧪 Tested** | 27 pytest tests: model init, output structure, drawing, a known image, API status codes, and browser post-processing vs Ultralytics |
| **🎯 Accuracy measured** | mAP on 500 COCO val2017 images (`eval_map.py`) |
| **📈 Benchmarking** | FPS analysis, confidence distribution, detection charts |

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Deep Learning Model | YOLOv8n (ONNX) in the browser, YOLOv8s on the server, YOLOv8n for CLI/benchmark/tests (Ultralytics, COCO-pretrained) |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Web Backend | FastAPI + Uvicorn |
| Web Frontend | HTML, CSS (Glassmorphism), JavaScript |
| In-browser inference | ONNX Runtime Web 1.30 (WebAssembly), model exported with `model.export(format="onnx")` |
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
| **In-browser** (headless Edge, 4 threads), bus.jpg, warm | ~218–242 ms per image (pre + model + post) |
| In-browser, single thread (no COOP/COEP headers) | ~386–428 ms per image |
| In-browser, model download + load + warm-up | 1.7–2.6 s (3 MB runtime + 12.5 MB model) |
| In-browser live webcam (headless Edge, fake camera) | ~3.4–4.1 FPS |

Model sizes: YOLOv8n has 3.16 M parameters (6.5 MB), YOLOv8s 11.17 M (22.6 MB).
On bus.jpg at confidence 0.4, nano finds the bus and 3 people; small finds the bus and 4.

### Accuracy (measured)

`python eval_map.py` downloads the COCO labels and a fixed random 500 of the
5,000 val2017 images (seed 0; 3,778 labelled objects), which the models never
trained on, and runs Ultralytics validation at 640 px on CPU.

| Model | mAP50-95 | mAP50 | Precision | Recall | Ultralytics' published mAP50-95 (full val) |
|---|---|---|---|---|---|
| YOLOv8n, PyTorch | 37.4 | 52.5 | 62.2 | 46.2 | 37.3 |
| YOLOv8s, PyTorch (server default) | 45.1 | 62.3 | 61.1 | 60.4 | 44.9 |
| YOLOv8n, ONNX (browser model) | 36.5 | 51.8 | 61.1 | 47.1 | — |

- The subset reproduces the published numbers within 0.2, so the evaluation is sound.
- The ONNX model scores 0.9 lower only because it always takes a 640×640 square
  input. Validating the PyTorch model with square padding too (`rect=False`)
  gives exactly 36.5 / 51.8, so the export itself loses nothing.
- Validation keeps boxes down to confidence 0.001, as COCO scoring requires;
  the app shows only boxes ≥ 0.4. Precision and recall are at Ultralytics'
  best-F1 threshold.
- These are COCO numbers. Accuracy on your own images (other cameras, classes,
  lighting) would need your own labelled test set.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **Server speed** | The "Server" option takes about a minute per image on Render's free CPU. The default on-device path avoids it. |
| **Device speed varies** | On-device speed depends on the visitor's hardware; weak phones will be slower. First visit downloads ~15.5 MB. Safari stays single-threaded (no `credentialless` support). |
| **Browser uses nano** | The browser runs YOLOv8n (mAP50-95 36.5), not the server's YOLOv8s (45.1), to stay fast. |
| **No tracking** | Counts are per frame; video summaries add detections across frames. |
| **80 COCO classes only** | Pretrained weights, no fine-tuning; objects outside COCO are not detected. |
| **Server-side drawing** | The server API returns a re-encoded base64 image (~110 KB) instead of box coordinates. (The on-device path draws locally.) |
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
