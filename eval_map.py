"""
Measure detection accuracy (mAP) on COCO val2017 images.

The models were trained on COCO train2017, so val2017 is data they never saw.
This script:
  1. downloads the COCO labels (YOLO format, 48 MB) and a fixed random subset
     of val2017 images (default 500 of 5000, seed 0) into datasets/
  2. runs Ultralytics validation for each model at 640 px
  3. writes eval/map_results.json and prints a table

Usage:
    python eval_map.py                  # 500 images, yolov8n, yolov8s, browser ONNX
    python eval_map.py --n 5000         # the full val2017 set (slower)

mAP50-95 is the COCO headline number: average precision averaged over IoU
thresholds 0.50, 0.55, ..., 0.95. mAP50 uses only IoU >= 0.5 (looser).
Validation keeps every box down to confidence 0.001, as COCO scoring expects;
the app itself shows only boxes >= 0.4.
"""
import argparse
import json
import os
import random
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "datasets")
LABELS_ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/{}"

MODELS = {
    "yolov8n (server, PyTorch)": "yolov8n.pt",
    "yolov8s (server default, PyTorch)": "yolov8s.pt",
    "yolov8n (browser, ONNX)": os.path.join(ROOT, "static", "models", "yolov8n.onnx"),
}

# Published by Ultralytics for the full val2017 set, for comparison.
PUBLISHED = {"yolov8n.pt": 37.3, "yolov8s.pt": 44.9}


def download(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)


def build_subset(n, seed):
    """datasets/coco-val{n}/images/val2017/*.jpg + labels/val2017/*.txt"""
    out = os.path.join(DATA, f"coco-val{n}")
    img_dir = os.path.join(out, "images", "val2017")
    lbl_dir = os.path.join(out, "labels", "val2017")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    zip_path = os.path.join(DATA, "coco2017labels.zip")
    print("Downloading COCO labels (48 MB, once)...")
    download(LABELS_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        val_list = z.read("coco/val2017.txt").decode().split()
        names = sorted(os.path.basename(p) for p in val_list)
        chosen = sorted(random.Random(seed).sample(names, n)) if n < len(names) else names
        for name in chosen:
            stem = os.path.splitext(name)[0]
            member = f"coco/labels/val2017/{stem}.txt"
            dst = os.path.join(lbl_dir, stem + ".txt")
            try:
                with z.open(member) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
            except KeyError:
                open(dst, "w").close()   # image with no objects

    print(f"Downloading {len(chosen)} val2017 images...")
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda nm: download(IMAGE_URL.format(nm), os.path.join(img_dir, nm)), chosen))

    yaml_path = os.path.join(out, "data.yaml")
    from ultralytics.utils import ROOT as ULTRA_ROOT
    import yaml
    with open(ULTRA_ROOT / "cfg" / "datasets" / "coco.yaml", encoding="utf-8") as f:
        names_map = yaml.safe_load(f)["names"]
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": out, "train": "images/val2017", "val": "images/val2017",
                        "names": names_map}, f)
    return yaml_path, len(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of val2017 images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    yaml_path, n = build_subset(args.n, args.seed)
    results = []
    for label, weights in MODELS.items():
        print(f"\nValidating {label} ...")
        m = YOLO(weights, task="detect")
        r = m.val(data=yaml_path, imgsz=640, batch=1, device="cpu", plots=False, verbose=False)
        row = {
            "model": label,
            "weights": os.path.basename(weights),
            "images": n,
            "mAP50_95": round(float(r.box.map) * 100, 1),
            "mAP50": round(float(r.box.map50) * 100, 1),
            "precision": round(float(r.box.mp) * 100, 1),
            "recall": round(float(r.box.mr) * 100, 1),
            "inference_ms_per_image": round(r.speed["inference"], 1),
            "published_mAP50_95_full_val": PUBLISHED.get(os.path.basename(weights)),
        }
        results.append(row)

    os.makedirs(os.path.join(ROOT, "eval"), exist_ok=True)
    out = os.path.join(ROOT, "eval", "map_results.json")
    with open(out, "w") as f:
        json.dump({"dataset": f"COCO val2017, {n} images (seed {args.seed})",
                   "imgsz": 640, "results": results}, f, indent=2)

    print(f"\n{'model':36} {'mAP50-95':>9} {'mAP50':>7} {'P':>6} {'R':>6} {'ms/img':>7}")
    for r in results:
        print(f"{r['model']:36} {r['mAP50_95']:9.1f} {r['mAP50']:7.1f} "
              f"{r['precision']:6.1f} {r['recall']:6.1f} {r['inference_ms_per_image']:7.1f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
