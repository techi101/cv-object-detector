import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from detector import ObjectDetector

app = FastAPI(title="YOLOv8 Object Detector API")

# Use nano model — only viable option for Render free tier (0.1 CPU, 512MB RAM)
print("Initializing YOLOv8n detector...")
detector = ObjectDetector(model_size="n", confidence=0.4)
print("Detector ready.")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def resize_image(frame, max_dim=1280):
    """
    Resize large images (e.g., phone photos) to a reasonable size.
    YOLOv8 internally resizes to 640x640 anyway, so sending a 12MP
    image just wastes time on encoding/decoding.
    """
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found."}


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs YOLOv8 detection, and returns the
    annotated image (base64) along with detection statistics.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # Resize large images (phone photos can be 12MP+) for faster processing
        frame = resize_image(frame)

        detections = detector.detect_frame(frame)
        annotated_frame = detector.draw_detections(frame, detections)

        # Encode with reduced quality for faster transfer
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
