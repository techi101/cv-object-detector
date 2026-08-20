"""
benchmark.py
------------
Performance benchmarking and analysis for the object detector.

Generates:
  - Detection statistics across a sample video
  - FPS performance chart
  - Confidence distribution histogram
  - Per-class detection breakdown chart
  - Technical report (text file)

Usage:
  python benchmark.py --source video.mp4
  python benchmark.py --source sample.jpg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict
from detector import ObjectDetector


def benchmark_on_video(source: str, detector: ObjectDetector,
                       max_frames: int = 500) -> dict:
    """
    Run detection on a video and collect performance metrics.

    Args:
        source: Path to video file
        detector: ObjectDetector instance
        max_frames: Maximum frames to process

    Returns:
        dict with benchmark results
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {source}")
        return {}

    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    print(f"  Benchmarking on {total_frames} frames ...")

    fps_log = []
    detections_per_frame = []
    all_confidences = []
    all_class_counts = defaultdict(int)
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        detections = detector.detect_frame(frame)
        elapsed = time.time() - start

        fps = 1.0 / max(elapsed, 1e-9)
        fps_log.append(fps)
        detections_per_frame.append(detections["count"])
        all_confidences.extend(detections["confidences"])

        for name in detections["class_names"]:
            all_class_counts[name] += 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    Frame {frame_idx}/{total_frames} | FPS: {fps:.1f}")

    cap.release()

    results = {
        "frames_processed": frame_idx,
        "fps_log": fps_log,
        "avg_fps": np.mean(fps_log),
        "min_fps": np.min(fps_log),
        "max_fps": np.max(fps_log),
        "detections_per_frame": detections_per_frame,
        "avg_detections": np.mean(detections_per_frame),
        "all_confidences": all_confidences,
        "avg_confidence": np.mean(all_confidences) if all_confidences else 0,
        "class_counts": dict(all_class_counts),
        "total_detections": sum(detections_per_frame),
    }

    return results


def benchmark_on_image(source: str, detector: ObjectDetector) -> dict:
    """
    Run detection on a single image for benchmarking.

    Args:
        source: Path to image file
        detector: ObjectDetector instance

    Returns:
        dict with benchmark results
    """
    frame = cv2.imread(source)
    if frame is None:
        print(f"  ERROR: Cannot read {source}")
        return {}

    print(f"  Benchmarking on image: {os.path.basename(source)}")
    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")

    # Run detection multiple times to get stable FPS measurement
    fps_log = []
    all_confidences = []
    all_class_counts = defaultdict(int)
    detections = None

    n_runs = 50
    print(f"  Running {n_runs} inference passes for FPS measurement ...")

    for i in range(n_runs):
        start = time.time()
        detections = detector.detect_frame(frame)
        elapsed = time.time() - start
        fps = 1.0 / max(elapsed, 1e-9)
        fps_log.append(fps)

    # Use last detection for statistics
    all_confidences = detections["confidences"]
    for name in detections["class_names"]:
        all_class_counts[name] += 1

    # Save annotated image
    os.makedirs("results", exist_ok=True)
    annotated = detector.draw_detections(frame, detections, fps=np.mean(fps_log))
    base = os.path.splitext(os.path.basename(source))[0]
    out_path = os.path.join("results", f"{base}_detected.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"  Saved annotated image: {out_path}")

    results = {
        "frames_processed": n_runs,
        "fps_log": fps_log,
        "avg_fps": np.mean(fps_log),
        "min_fps": np.min(fps_log),
        "max_fps": np.max(fps_log),
        "detections_per_frame": [detections["count"]] * n_runs,
        "avg_detections": detections["count"],
        "all_confidences": all_confidences,
        "avg_confidence": np.mean(all_confidences) if all_confidences else 0,
        "class_counts": dict(all_class_counts),
        "total_detections": detections["count"],
    }

    return results


def generate_plots(results: dict, output_dir: str = "results"):
    """Generate performance analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    fig.suptitle("Object Detection Performance Analysis",
                 fontsize=14, fontweight="bold")

    # 1. FPS over time
    ax = axes[0, 0]
    ax.plot(results["fps_log"], color="#2563EB", linewidth=0.8, alpha=0.7)
    ax.axhline(y=results["avg_fps"], color="#EF4444", linestyle="--",
               label=f'Avg: {results["avg_fps"]:.1f} FPS')
    ax.set_xlabel("Frame")
    ax.set_ylabel("FPS")
    ax.set_title("Inference Speed (FPS)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Confidence distribution
    ax = axes[0, 1]
    if results["all_confidences"]:
        ax.hist(results["all_confidences"], bins=20, color="#10B981",
                edgecolor="white", alpha=0.8)
        ax.axvline(x=results["avg_confidence"], color="#EF4444",
                   linestyle="--",
                   label=f'Avg: {results["avg_confidence"]:.0%}')
        ax.legend()
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Detection Confidence Distribution")
    ax.grid(True, alpha=0.3)

    # 3. Detections per frame
    ax = axes[1, 0]
    ax.plot(results["detections_per_frame"], color="#8B5CF6",
            linewidth=0.8, alpha=0.7)
    ax.axhline(y=results["avg_detections"], color="#EF4444",
               linestyle="--",
               label=f'Avg: {results["avg_detections"]:.1f}')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Detections")
    ax.set_title("Objects Detected per Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Per-class breakdown
    ax = axes[1, 1]
    if results["class_counts"]:
        sorted_classes = sorted(results["class_counts"].items(),
                                key=lambda x: -x[1])
        names = [c[0] for c in sorted_classes[:10]]
        counts = [c[1] for c in sorted_classes[:10]]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax.barh(names, counts, color=colors, edgecolor="white")
        ax.invert_yaxis()
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(count), va="center", fontsize=9)
    ax.set_xlabel("Total Detections")
    ax.set_title("Top Detected Object Classes")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(output_dir, "performance_analysis.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


def generate_report(results: dict, output_dir: str = "results"):
    """Generate a text-based technical report."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_report.txt")

    with open(path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("  OBJECT DETECTION BENCHMARK REPORT\n")
        f.write("=" * 55 + "\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model:            YOLOv8n (nano)\n")
        f.write(f"  Framework:        PyTorch + Ultralytics\n")
        f.write(f"  Visualization:    OpenCV\n\n")

        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Frames processed: {results['frames_processed']}\n")
        f.write(f"  Average FPS:      {results['avg_fps']:.1f}\n")
        f.write(f"  Min FPS:          {results['min_fps']:.1f}\n")
        f.write(f"  Max FPS:          {results['max_fps']:.1f}\n\n")

        f.write("DETECTION STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total detections:     {results['total_detections']}\n")
        f.write(f"  Avg detections/frame: {results['avg_detections']:.1f}\n")
        f.write(f"  Avg confidence:       {results['avg_confidence']:.1%}\n\n")

        f.write("PER-CLASS BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for cls, count in sorted(results["class_counts"].items(),
                                  key=lambda x: -x[1]):
            f.write(f"  {cls:20s}  {count}\n")

        f.write("\n" + "=" * 55 + "\n")

    print(f"  [SAVED] {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv8 Object Detection Performance"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to video file or image for benchmarking"
    )
    parser.add_argument(
        "--max-frames", type=int, default=500,
        help="Max frames to process from video. Default: 500"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.4,
        help="Confidence threshold. Default: 0.4"
    )
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("  OBJECT DETECTION BENCHMARK")
    print("=" * 55)

    detector = ObjectDetector(model_size="n", confidence=args.confidence)

    # Detect if source is an image or video
    is_image = args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_image:
        results = benchmark_on_image(args.source, detector)
    else:
        results = benchmark_on_video(args.source, detector,
                                     max_frames=args.max_frames)

    if not results:
        return

    print("\n  Generating analysis plots ...")
    generate_plots(results)
    generate_report(results)

    print(f"\n  Done! Check the results/ folder for outputs.")


if __name__ == "__main__":
    main()
