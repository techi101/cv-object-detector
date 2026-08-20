"""
detector.py
-----------
Real-Time Object Detection & Tracking using YOLOv8 and OpenCV.

Supports:
  - Live webcam detection
  - Video file detection
  - Saving annotated output video
  - Real-time FPS overlay
  - Confidence threshold filtering
  - Per-class object counting

Usage:
  python detector.py                          # Webcam (live)
  python detector.py --source video.mp4       # Video file
  python detector.py --source video.mp4 --save  # Save annotated output
  python detector.py --confidence 0.5         # Set confidence threshold
"""

import cv2
import numpy as np
import argparse
import time
import os
from collections import defaultdict
from ultralytics import YOLO


# ── Color Palette ────────────────────────────────────────────────────────────
# Visually distinct colors for different object classes (BGR format for OpenCV)
COLORS = [
    (255, 107, 107),   # Red
    (78, 205, 196),    # Teal
    (255, 195, 0),     # Gold
    (106, 176, 76),    # Green
    (199, 125, 255),   # Purple
    (255, 154, 162),   # Pink
    (0, 180, 216),     # Cyan
    (255, 183, 77),    # Orange
    (144, 190, 109),   # Lime
    (108, 142, 191),   # Steel Blue
]


def get_color(class_id: int) -> tuple:
    """Return a consistent color for a given class ID."""
    return COLORS[class_id % len(COLORS)]


class ObjectDetector:
    """
    YOLOv8-based object detector with OpenCV visualization.

    Attributes:
        model: YOLOv8 model instance
        confidence_threshold: Minimum confidence to display a detection
        frame_count: Total frames processed
        detection_log: Per-frame detection counts
    """

    def __init__(self, model_size: str = "n", confidence: float = 0.4):
        """
        Initialize the detector.

        Args:
            model_size: YOLOv8 model variant ('n'=nano, 's'=small, 'm'=medium)
            confidence: Minimum confidence threshold (0.0 to 1.0)
        """
        model_name = f"yolov8{model_size}.pt"
        print(f"  Loading YOLOv8 model: {model_name} ...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence
        self.class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}
        self.frame_count = 0
        self.detection_log = []  # List of per-frame detection dicts
        self.fps_history = []
        print(f"  Model loaded. {len(self.class_names)} object classes available.")
        print(f"  Confidence threshold: {confidence:.0%}")

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Run YOLOv8 inference on a single frame.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            dict with keys:
                'boxes': list of [x1, y1, x2, y2] bounding boxes
                'confidences': list of float confidence scores
                'class_ids': list of int class IDs
                'class_names': list of str class names
                'count': total detections in this frame
        """
        # Run inference (verbose=False suppresses ultralytics output)
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = {
            "boxes": [],
            "confidences": [],
            "class_ids": [],
            "class_names": [],
            "count": 0,
        }

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]

                detections["boxes"].append([x1, y1, x2, y2])
                detections["confidences"].append(confidence)
                detections["class_ids"].append(class_id)
                detections["class_names"].append(class_name)

        detections["count"] = len(detections["boxes"])
        return detections

    def draw_detections(self, frame: np.ndarray, detections: dict,
                        fps: float = None) -> np.ndarray:
        """
        Draw bounding boxes, labels, and HUD overlay on the frame.

        Args:
            frame: Original BGR frame
            detections: Output from detect_frame()
            fps: Current FPS to display (optional)

        Returns:
            Annotated frame (copy)
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Count objects per class for the HUD
        class_counts = defaultdict(int)

        for i in range(detections["count"]):
            x1, y1, x2, y2 = detections["boxes"][i]
            confidence = detections["confidences"][i]
            class_id = detections["class_ids"][i]
            class_name = detections["class_names"][i]
            color = get_color(class_id)

            class_counts[class_name] += 1

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name} {confidence:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 4),
                (x1 + label_w + 4, y1),
                color, -1  # Filled
            )
            # Draw label text
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # ── HUD Overlay (top-left) ──────────────────────────────────────
        hud_y = 30

        # FPS counter
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            hud_y += 30

        # Total detections
        total_text = f"Objects: {detections['count']}"
        cv2.putText(annotated, total_text, (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        hud_y += 25

        # Per-class breakdown
        for cls_name, count in sorted(class_counts.items()):
            cls_text = f"  {cls_name}: {count}"
            cv2.putText(annotated, cls_text, (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            hud_y += 20

        return annotated

    def process_source(self, source=0, save_output: bool = False,
                       output_dir: str = "results"):
        """
        Run detection on a video source (webcam or file).

        Args:
            source: 0 for webcam, or path to video file
            save_output: Whether to save annotated video
            output_dir: Directory for output files
        """
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  ERROR: Cannot open video source: {source}")
            return

        # Get video properties
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        is_webcam = isinstance(source, int)

        source_name = "Webcam" if is_webcam else os.path.basename(str(source))
        print(f"\n  Source: {source_name}")
        print(f"  Resolution: {frame_w}x{frame_h}")
        if not is_webcam:
            print(f"  Total frames: {total_frames}")
            print(f"  Input FPS: {input_fps:.1f}")

        # Setup video writer if saving
        writer = None
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "detected_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, input_fps,
                                     (frame_w, frame_h))
            print(f"  Saving output to: {out_path}")

        print(f"\n  Running detection ... (Press 'q' to stop)\n")

        # Cumulative stats
        total_detections = 0
        all_class_counts = defaultdict(int)
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Detect
            detections = self.detect_frame(frame)
            total_detections += detections["count"]

            # Track per-class totals
            for name in detections["class_names"]:
                all_class_counts[name] += 1

            # Log detections for this frame
            self.detection_log.append({
                "frame": self.frame_count,
                "count": detections["count"],
                "classes": dict(defaultdict(int)),
            })

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time
            self.fps_history.append(fps)

            # Draw
            annotated = self.draw_detections(frame, detections, fps=fps)

            # Save frame
            if writer is not None:
                writer.write(annotated)

            # Display (skip if no display available)
            try:
                cv2.imshow("YOLOv8 Object Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("  User pressed 'q'. Stopping.")
                    break
            except cv2.error:
                # No display available (e.g., headless server)
                pass

            # Progress indicator for video files
            if not is_webcam and self.frame_count % 100 == 0:
                pct = (self.frame_count / max(total_frames, 1)) * 100
                print(f"    Processed {self.frame_count}/{total_frames}"
                      f" frames ({pct:.0f}%) | FPS: {fps:.1f}")

        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        # Print summary
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        print(f"\n  {'='*50}")
        print(f"  DETECTION SUMMARY")
        print(f"  {'='*50}")
        print(f"  Frames processed:    {self.frame_count}")
        print(f"  Total detections:    {total_detections}")
        print(f"  Avg detections/frame: {total_detections/max(self.frame_count,1):.1f}")
        print(f"  Average FPS:         {avg_fps:.1f}")
        print(f"  {'─'*50}")
        print(f"  Objects detected by class:")
        for cls, count in sorted(all_class_counts.items(),
                                  key=lambda x: -x[1]):
            print(f"    {cls:20s}  {count}")
        print(f"  {'='*50}")

        return {
            "frames": self.frame_count,
            "total_detections": total_detections,
            "avg_fps": avg_fps,
            "class_counts": dict(all_class_counts),
        }


def run_image_detection(detector: ObjectDetector, image_path: str,
                        output_dir: str = "results") -> dict:
    """
    Run detection on a single image and save the annotated result.

    Args:
        detector: ObjectDetector instance
        image_path: Path to input image
        output_dir: Directory for output files

    Returns:
        Detection results dict
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ERROR: Cannot read image: {image_path}")
        return {}

    print(f"\n  Detecting objects in: {os.path.basename(image_path)}")
    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")

    detections = detector.detect_frame(frame)
    annotated = detector.draw_detections(frame, detections)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}_detected.jpg")
    cv2.imwrite(out_path, annotated)

    print(f"  Objects found: {detections['count']}")
    for i in range(detections["count"]):
        name = detections["class_names"][i]
        conf = detections["confidences"][i]
        print(f"    [{i+1}] {name} ({conf:.0%})")
    print(f"  Saved: {out_path}")

    return detections


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Real-Time Object Detection with OpenCV"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video file path or image path. Defaults to webcam (0)."
    )
    parser.add_argument(
        "--confidence", type=float, default=0.4,
        help="Minimum confidence threshold (0.0 to 1.0). Default: 0.4"
    )
    parser.add_argument(
        "--model", type=str, default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(tra). Default: n"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output video/image to results/"
    )
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("  YOLOv8 REAL-TIME OBJECT DETECTION")
    print("=" * 55)

    detector = ObjectDetector(model_size=args.model,
                              confidence=args.confidence)

    if args.source and args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        # Image mode
        run_image_detection(detector, args.source)
    else:
        # Video mode (webcam or video file)
        source = int(args.source) if args.source and args.source.isdigit() else (args.source or 0)
        detector.process_source(source=source, save_output=args.save)


if __name__ == "__main__":
    main()
