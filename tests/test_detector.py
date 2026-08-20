"""
tests/test_detector.py
----------------------
Pytest suite for the Object Detector module.

Tests cover:
  - Model initialization
  - Detection on synthetic images
  - Confidence threshold filtering
  - Drawing functions
  - Color assignment consistency
  - Detection output structure
"""

import numpy as np
import pytest
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from detector import ObjectDetector, get_color


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector():
    """Create a shared detector instance (model loads once)."""
    return ObjectDetector(model_size="n", confidence=0.3)


@pytest.fixture
def blank_frame():
    """Create a blank 640x480 black frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def noisy_frame():
    """Create a random noise frame (simulates a busy scene)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ── Model Initialization Tests ───────────────────────────────────────────────

class TestModelInit:
    def test_model_loads(self, detector):
        """Model should load without errors."""
        assert detector.model is not None

    def test_class_names_populated(self, detector):
        """COCO dataset has 80 classes."""
        assert len(detector.class_names) == 80

    def test_class_names_include_person(self, detector):
        """'person' should be class 0 in COCO."""
        assert detector.class_names[0] == "person"

    def test_class_names_include_car(self, detector):
        """'car' should be in the class list."""
        assert "car" in detector.class_names.values()

    def test_confidence_threshold_set(self, detector):
        """Confidence threshold should match initialization value."""
        assert detector.confidence_threshold == 0.3


# ── Detection Output Structure ───────────────────────────────────────────────

class TestDetectionOutput:
    def test_output_is_dict(self, detector, blank_frame):
        """Detection output should be a dictionary."""
        result = detector.detect_frame(blank_frame)
        assert isinstance(result, dict)

    def test_output_has_required_keys(self, detector, blank_frame):
        """Output must contain all required keys."""
        result = detector.detect_frame(blank_frame)
        required = {"boxes", "confidences", "class_ids", "class_names", "count"}
        assert required.issubset(result.keys())

    def test_blank_frame_no_detections(self, detector, blank_frame):
        """A pure black frame should produce zero detections."""
        result = detector.detect_frame(blank_frame)
        assert result["count"] == 0
        assert len(result["boxes"]) == 0

    def test_lists_same_length(self, detector, noisy_frame):
        """All output lists should have the same length."""
        result = detector.detect_frame(noisy_frame)
        n = result["count"]
        assert len(result["boxes"]) == n
        assert len(result["confidences"]) == n
        assert len(result["class_ids"]) == n
        assert len(result["class_names"]) == n

    def test_confidences_in_range(self, detector, noisy_frame):
        """All confidence scores must be between 0 and 1."""
        result = detector.detect_frame(noisy_frame)
        for conf in result["confidences"]:
            assert 0.0 <= conf <= 1.0


# ── Drawing Tests ────────────────────────────────────────────────────────────

class TestDrawing:
    def test_draw_returns_image(self, detector, blank_frame):
        """draw_detections should return a valid image array."""
        detections = detector.detect_frame(blank_frame)
        annotated = detector.draw_detections(blank_frame, detections)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == blank_frame.shape

    def test_draw_does_not_modify_original(self, detector, blank_frame):
        """Original frame should not be modified."""
        original_copy = blank_frame.copy()
        detections = detector.detect_frame(blank_frame)
        detector.draw_detections(blank_frame, detections)
        assert np.array_equal(blank_frame, original_copy)

    def test_draw_with_fps(self, detector, blank_frame):
        """Drawing with FPS overlay should not crash."""
        detections = detector.detect_frame(blank_frame)
        annotated = detector.draw_detections(blank_frame, detections, fps=30.0)
        assert annotated is not None


# ── Color Assignment Tests ───────────────────────────────────────────────────

class TestColors:
    def test_color_is_tuple(self):
        """Colors should be BGR tuples."""
        color = get_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_color_consistency(self):
        """Same class ID should always get the same color."""
        assert get_color(5) == get_color(5)

    def test_different_classes_different_colors(self):
        """Adjacent class IDs should get different colors."""
        assert get_color(0) != get_color(1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
