import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from detector import ObjectDetector

app = FastAPI(title="YOLOv8 Object Detector API")

# Initialize a single detector using the small model for better accuracy
# (free tier only has 512MB RAM — can't load two models)
print("Initializing YOLOv8s detector...")
detector = ObjectDetector(model_size="s", confidence=0.4)
print("Detector ready.")

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
