import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from detector import ObjectDetector

app = FastAPI(title="YOLOv8 Object Detector API")

# Initialize TWO detectors:
#   - nano  → fast, for live webcam (speed > accuracy)
#   - small → accurate, for image uploads (accuracy > speed)
print("Initializing detectors...")
detector_nano = ObjectDetector(model_size="n", confidence=0.4)
detector_small = ObjectDetector(model_size="s", confidence=0.4)
print("Both detectors ready.")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found."}


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    mode: str = Query(default="upload", description="'upload' for accurate (small model), 'live' for fast (nano model)")
):
    """
    Accepts an uploaded image, runs YOLOv8 detection, and returns the
    annotated image (base64) along with detection statistics.

    mode=upload  → uses YOLOv8s (more accurate)
    mode=live    → uses YOLOv8n (faster for real-time)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # Pick the right detector based on mode
        detector = detector_nano if mode == "live" else detector_small

        detections = detector.detect_frame(frame)
        annotated_frame = detector.draw_detections(frame, detections)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        class_counts = {}
        for class_name in detections["class_names"]:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return {
            "success": True,
            "image": encoded_image,
            "count": detections["count"],
            "class_counts": class_counts
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
