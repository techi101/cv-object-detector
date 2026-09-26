"""
tests/test_api.py
-----------------
Tests for the FastAPI app: status codes for good and bad uploads.
Uses the nano model (MODEL_SIZE=n) so the tests need only yolov8n.pt.
"""

import os
import sys

import pytest

os.environ.setdefault("MODEL_SIZE", "n")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
import ultralytics

import app as app_module

BUS_JPG = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


def test_homepage(client):
    assert client.get("/").status_code == 200


def test_detect_real_image(client):
    with open(BUS_JPG, "rb") as f:
        r = client.post("/detect", files={"file": ("bus.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["class_counts"].get("bus") == 1
    assert data["image"].startswith("/9j/")        # base64 of a JPEG


def test_undecodable_bytes_are_400_not_500(client):
    r = client.post("/detect", files={"file": ("x.bin", b"not an image", "application/octet-stream")})
    assert r.status_code == 400
    assert r.json()["detail"] == "Could not decode image."


def test_wrong_content_type_is_400(client):
    r = client.post("/detect", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_oversized_upload_is_413(client):
    big = b"0" * (app_module.MAX_UPLOAD_BYTES + 1)
    r = client.post("/detect", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413
