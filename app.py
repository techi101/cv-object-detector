import cv2
import numpy as np
import base64
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from detector import ObjectDetector

app = FastAPI(title="YOLOv8 Object Detector API")
log = logging.getLogger("detector-api")

# Small model by default for accuracy on uploads. MODEL_SIZE=n selects nano,
# about 2x faster on CPU (measured locally: ~80 ms vs ~169 ms per image).
MODEL_SIZE = os.getenv("MODEL_SIZE", "s")

# Reject uploads above this size before decoding: a decoded 12 MP photo is
# ~36 MB of pixels, a large share of a 512 MB free-tier instance.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

print(f"Initializing YOLOv8{MODEL_SIZE} detector...")
detector = ObjectDetector(model_size=MODEL_SIZE, confidence=0.4)
print("Detector ready.")

@app.middleware("http")
async def cross_origin_isolation(request, call_next):
    """
    Make the page "cross-origin isolated" so the in-browser model can use
    several CPU threads (browsers only allow the shared memory that
    multi-threaded WebAssembly needs on isolated pages). "credentialless"
    still lets the page load Google Fonts and the ONNX Runtime CDN files.
    Browsers without support simply stay single-threaded.
    """
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    return response


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def resize_image(frame, max_dim=640):
    """
    Resize large images before inference.
    YOLOv8 letterboxes to 640 on the long side anyway, so pre-shrinking costs
    almost nothing: on bus.jpg the same 4 objects were found with confidences
    within 0.01 of the full-size run. It cuts memory and decode time a lot for
    phone photos (12MP → ~0.3MP).
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
def detect_objects(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs YOLOv8 detection, and returns the
    annotated image (base64) along with detection statistics.

    Declared with plain `def`, not `async def`: inference is CPU-bound, and in
    an async route it would block the event loop so the server could not answer
    any other request (even the homepage) until it finished. FastAPI runs plain
    `def` routes in a thread pool.
    """
    # Be lenient with content type — mobile browsers often send wrong MIME types
    # We'll validate the actual image data instead when we try to decode it
    if file.content_type and not file.content_type.startswith("image/") and file.content_type != "application/octet-stream":
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image is larger than 15 MB.")

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # Resize to 640px max — the size YOLO works at internally
        frame = resize_image(frame)

        detections = detector.detect_frame(frame)
        annotated_frame = detector.draw_detections(frame, detections)

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

    except HTTPException:
        # Our own 400/413 responses must pass through unchanged. Without this
        # clause the catch-all below turned them into 500s.
        raise
    except Exception:
        log.exception("detection failed")
        raise HTTPException(status_code=500, detail="Detection failed on the server.")
