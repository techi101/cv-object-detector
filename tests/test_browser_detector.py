"""
Check that the browser post-processing (static/browser_detector.js) gives the
same boxes as Ultralytics in Python.

The ONNX model is run here with onnxruntime (same model file the browser
loads), its raw output is handed to the JavaScript postprocess() under Node,
and the result is compared with Ultralytics predict on the same ONNX file.
Skipped when Node or onnxruntime is not installed.
"""
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(ROOT, "static", "models", "yolov8n.onnx")
JS_PATH = os.path.join(ROOT, "static", "browser_detector.js")

ort = pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

NODE_SCRIPT = """
const bd = require(process.argv[2]);
const fs = require('fs');
const buf = fs.readFileSync(process.argv[3]);
const out = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
const lb = bd.letterboxParams(+process.argv[4], +process.argv[5]);
console.log(JSON.stringify(bd.postprocess(out, lb, { conf: 0.25 })));
"""


def bus_image():
    from ultralytics.utils import ASSETS
    return cv2.imread(str(ASSETS / "bus.jpg"))


def letterbox(img):
    """Ultralytics' own letterbox, so only the JS postprocess is under test."""
    from ultralytics.data.augment import LetterBox
    lb = LetterBox((640, 640), auto=False)(image=img)
    rgb = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return np.ascontiguousarray(rgb)


def js_postprocess(raw, w, h):
    with tempfile.TemporaryDirectory() as d:
        bin_path = os.path.join(d, "out.bin")
        script = os.path.join(d, "run.js")
        raw.astype(np.float32).tofile(bin_path)
        with open(script, "w") as f:
            f.write(NODE_SCRIPT)
        res = subprocess.run(["node", script, JS_PATH, bin_path, str(w), str(h)],
                             capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def test_letterbox_params_match_ultralytics():
    res = subprocess.run(
        ["node", "-e",
         f"const bd=require({json.dumps(JS_PATH)});"
         "console.log(JSON.stringify([bd.letterboxParams(810,1080),bd.letterboxParams(1920,1080)]))"],
        capture_output=True, text=True, check=True)
    tall, wide = json.loads(res.stdout)
    assert (tall["newW"], tall["newH"], tall["padX"], tall["padY"]) == (480, 640, 80, 0)
    assert (wide["newW"], wide["newH"], wide["padX"], wide["padY"]) == (640, 360, 0, 140)


def test_js_postprocess_matches_ultralytics():
    from ultralytics import YOLO

    img = bus_image()
    h, w = img.shape[:2]
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    raw = sess.run(None, {"images": letterbox(img)})[0]
    assert raw.shape == (1, 84, 8400)
    js = js_postprocess(raw, w, h)

    # Reference: Ultralytics running the same ONNX file (fixed 640x640 input;
    # the .pt predictor pads to 640x480 instead, which shifts scores slightly).
    ref = YOLO(ONNX_PATH, task="detect").predict(img, conf=0.25, verbose=False)[0].boxes
    ref_cls = sorted(int(c) for c in ref.cls)
    assert sorted(d["classId"] for d in js) == ref_cls

    # Every Python box has a same-class JS box within 2 px and 0.01 score.
    for box, cls, score in zip(ref.xyxy.numpy(), ref.cls.numpy(), ref.conf.numpy()):
        diffs = [np.abs(np.array(d["box"]) - box).max() for d in js if d["classId"] == cls]
        best = int(np.argmin(diffs))
        match = [d for d in js if d["classId"] == cls][best]
        assert diffs[best] < 2.0
        assert abs(match["score"] - score) < 0.01
