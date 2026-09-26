/*
 * In-browser YOLOv8 detection: the maths around the model.
 *
 * The model itself (static/models/yolov8n.onnx) is run by ONNX Runtime Web.
 * This file does the two jobs the model does not do for you:
 *   1. preprocess  - turn any image into the 1x3x640x640 float tensor it expects
 *   2. postprocess - turn its raw 1x84x8400 output into a short list of boxes
 * It mirrors what Ultralytics does in Python (letterbox, confidence filter,
 * per-class NMS), so browser and server results match (checked by
 * tests/test_browser_detector.py).
 *
 * No DOM access except in drawLetterboxed/drawDetections, so the maths can be
 * run under Node for testing.
 */
(function (root) {
    'use strict';

    const INPUT_SIZE = 640;     // the model was exported for 640x640 input
    const PAD_VALUE = 114;      // grey used by Ultralytics for letterbox padding

    const CLASS_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush',
    ];

    // Same palette as detector.py COLORS (which stores them as BGR for OpenCV).
    const COLORS = ['#FF6B6B', '#4ECDC4', '#FFC300', '#6AB04C', '#C77DFF',
                    '#FF9AA2', '#00B4D8', '#FFB74D', '#90BE6D', '#6C8EBF'];

    /**
     * Letterbox geometry: scale the image to fit 640x640 without distortion
     * and centre it, padding the rest. Same rounding as Ultralytics LetterBox.
     */
    function letterboxParams(srcW, srcH) {
        const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
        const newW = Math.round(srcW * scale);
        const newH = Math.round(srcH * scale);
        const padX = Math.round((INPUT_SIZE - newW) / 2 - 0.1);
        const padY = Math.round((INPUT_SIZE - newH) / 2 - 0.1);
        return { scale, newW, newH, padX, padY, srcW, srcH };
    }

    /**
     * RGBA pixels of the 640x640 letterboxed image -> Float32 tensor data in
     * CHW order (all reds, then all greens, then all blues), scaled to 0..1.
     */
    function rgbaToTensor(rgba) {
        const area = INPUT_SIZE * INPUT_SIZE;
        const out = new Float32Array(3 * area);
        for (let i = 0; i < area; i++) {
            out[i] = rgba[i * 4] / 255;
            out[area + i] = rgba[i * 4 + 1] / 255;
            out[2 * area + i] = rgba[i * 4 + 2] / 255;
        }
        return out;
    }

    function iou(a, b) {
        const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
        const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = (a[2] - a[0]) * (a[3] - a[1]);
        const areaB = (b[2] - b[0]) * (b[3] - b[1]);
        return inter / (areaA + areaB - inter + 1e-9);
    }

    /**
     * Raw model output -> final detections in original-image pixels.
     *
     * output is 84 rows x 8400 columns, stored row by row: rows 0-3 are the box
     * (centre x, centre y, width, height in 640-space), rows 4-83 are the 80
     * class scores. Each column is one candidate box.
     */
    function postprocess(output, lb, opts) {
        const conf = (opts && opts.conf) != null ? opts.conf : 0.4;
        const iouThr = (opts && opts.iou) != null ? opts.iou : 0.7;  // Ultralytics default
        const maxDet = (opts && opts.maxDet) || 300;
        const numClasses = CLASS_NAMES.length;
        const n = output.length / (4 + numClasses);

        // 1. Keep candidates whose best class score clears the threshold.
        const cands = [];
        for (let i = 0; i < n; i++) {
            let best = 0, cls = -1;
            for (let c = 0; c < numClasses; c++) {
                const s = output[(4 + c) * n + i];
                if (s > best) { best = s; cls = c; }
            }
            if (best < conf) continue;
            const cx = output[i], cy = output[n + i];
            const w = output[2 * n + i], h = output[3 * n + i];
            cands.push({ classId: cls, score: best,
                         box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] });
        }

        // 2. Non-maximum suppression, per class: highest score wins, and any
        //    same-class box overlapping it by more than iouThr is dropped.
        cands.sort((a, b) => b.score - a.score);
        const kept = [];
        for (const c of cands) {
            if (kept.length >= maxDet) break;
            if (kept.every(k => k.classId !== c.classId || iou(k.box, c.box) <= iouThr)) {
                kept.push(c);
            }
        }

        // 3. Undo the letterbox: remove padding, divide by scale, clip to image.
        const clip = (v, max) => Math.min(Math.max(v, 0), max);
        return kept.map(k => ({
            classId: k.classId,
            name: CLASS_NAMES[k.classId],
            score: k.score,
            box: [
                clip((k.box[0] - lb.padX) / lb.scale, lb.srcW),
                clip((k.box[1] - lb.padY) / lb.scale, lb.srcH),
                clip((k.box[2] - lb.padX) / lb.scale, lb.srcW),
                clip((k.box[3] - lb.padY) / lb.scale, lb.srcH),
            ],
        }));
    }

    function countByClass(dets) {
        const counts = {};
        for (const d of dets) counts[d.name] = (counts[d.name] || 0) + 1;
        return counts;
    }

    // ── Browser-only helpers ───────────────────────────────

    /** Draw source (img/video/canvas) letterboxed onto a 640x640 canvas. */
    function drawLetterboxed(source, srcW, srcH, canvas) {
        const lb = letterboxParams(srcW, srcH);
        canvas.width = INPUT_SIZE;
        canvas.height = INPUT_SIZE;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
        ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
        ctx.drawImage(source, lb.padX, lb.padY, lb.newW, lb.newH);
        const rgba = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
        return { lb, tensorData: rgbaToTensor(rgba) };
    }

    /** Boxes and "name 92%" labels, styled like the server's OpenCV drawing. */
    function drawDetections(ctx, dets) {
        // Scale with the image so labels stay readable when it is shown small.
        const longSide = Math.max(ctx.canvas.width, ctx.canvas.height);
        const lineW = Math.max(2, Math.round(longSide / 300));
        const font = Math.max(14, Math.round(longSide / 32));
        ctx.lineWidth = lineW;
        ctx.font = `600 ${font}px Inter, system-ui, sans-serif`;
        ctx.textBaseline = 'top';
        for (const d of dets) {
            const [x1, y1, x2, y2] = d.box;
            const color = COLORS[d.classId % COLORS.length];
            ctx.strokeStyle = color;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            const label = `${d.name} ${Math.round(d.score * 100)}%`;
            const tw = ctx.measureText(label).width + 8;
            const th = font + 6;
            const ty = y1 - th >= 0 ? y1 - th : y1;
            ctx.fillStyle = color;
            ctx.fillRect(x1 - lineW / 2, ty, tw, th);
            ctx.fillStyle = '#fff';
            ctx.fillText(label, x1 + 4 - lineW / 2, ty + 3);
        }
    }

    const api = { INPUT_SIZE, CLASS_NAMES, COLORS, letterboxParams, rgbaToTensor,
                  postprocess, countByClass, drawLetterboxed, drawDetections };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.BrowserDetector = api;
})(typeof self !== 'undefined' ? self : this);
