document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultImage = document.getElementById('result-image');
    const totalCount = document.getElementById('total-count');
    const classList = document.getElementById('class-list');

    let webcamStream = null;
    let liveDetectionRunning = false;
    let liveDetectionAbort = null;

    // ── Inference engine ────────────────────────────────
    // 'device': the YOLOv8n ONNX model runs in this browser (ONNX Runtime Web).
    // 'server': the image is uploaded to /detect and YOLOv8 runs on the server.
    // Device is the default: on the free hosting the server takes about a
    // minute per image, the browser a fraction of a second.
    const MODEL_URL = '/static/models/yolov8n.onnx';
    const ORT_WASM_PATH = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
    const CONFIDENCE = 0.4;     // same threshold as the server (app.py)
    let engine = 'device';
    let session = null;
    let modelLoading = null;    // promise, so parallel callers share one load
    const engineStatus = document.getElementById('engine-status');
    const inputCanvas = document.createElement('canvas');  // 640x640 model input

    function loadModel() {
        if (modelLoading) return modelLoading;
        modelLoading = (async () => {
            if (typeof ort === 'undefined') throw new Error('ONNX Runtime did not load');
            ort.env.wasm.wasmPaths = ORT_WASM_PATH;
            // Multi-threading needs a cross-origin-isolated page (see app.py).
            ort.env.wasm.numThreads = self.crossOriginIsolated
                ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
            const t0 = performance.now();
            session = await ort.InferenceSession.create(MODEL_URL, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
            });
            // Warm-up: the first run is slow while buffers are allocated.
            const blank = new ort.Tensor('float32', new Float32Array(3 * 640 * 640), [1, 3, 640, 640]);
            await session.run({ [session.inputNames[0]]: blank });
            const threads = ort.env.wasm.numThreads;
            engineStatus.textContent = `Model ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`
                + ` · runs in your browser (${threads} thread${threads > 1 ? 's' : ''})`;
            return session;
        })();
        modelLoading.catch(err => {
            console.warn('In-browser model failed, falling back to server:', err);
            document.getElementById('engine-device').disabled = true;
            setEngine('server');
            engineStatus.textContent = 'In-browser model unavailable · using server';
        });
        return modelLoading;
    }

    window.setEngine = function(mode) {
        const wasLive = liveDetectionRunning;
        if (wasLive) stopLiveDetection();
        engine = mode;
        document.getElementById('engine-device').classList.toggle('active', mode === 'device');
        document.getElementById('engine-server').classList.toggle('active', mode === 'server');
        if (mode === 'server') {
            engineStatus.textContent = 'Uploads to the server · about 1 min per image on free hosting';
        } else {
            engineStatus.textContent = session ? 'Model ready · runs in your browser' : 'Loading model…';
            loadModel();
        }
        if (wasLive) startLiveDetection();
    };

    /** Run YOLOv8n in the browser on an image, video or canvas. */
    async function detectOnDevice(source, width, height) {
        await loadModel();
        const t0 = performance.now();
        const { lb, tensorData } = BrowserDetector.drawLetterboxed(source, width, height, inputCanvas);
        const tensor = new ort.Tensor('float32', tensorData, [1, 3, 640, 640]);
        const out = await session.run({ [session.inputNames[0]]: tensor });
        const dets = BrowserDetector.postprocess(out[session.outputNames[0]].data, lb, { conf: CONFIDENCE });
        return { dets, ms: performance.now() - t0 };
    }

    /** Draw source + boxes on a canvas; return data in the /detect response shape. */
    function renderDeviceResult(source, width, height, dets, canvas) {
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(source, 0, 0, width, height);
        BrowserDetector.drawDetections(ctx, dets);
        return { count: dets.length, class_counts: BrowserDetector.countByClass(dets) };
    }

    loadModel();

    // ── Mode Switching ──────────────────────────────────
    window.switchMode = function(mode) {
        document.getElementById('tab-upload').classList.toggle('active', mode === 'upload');
        document.getElementById('tab-webcam').classList.toggle('active', mode === 'webcam');
        document.getElementById('mode-upload').classList.toggle('hidden', mode !== 'upload');
        document.getElementById('mode-webcam').classList.toggle('hidden', mode !== 'webcam');
        results.classList.add('hidden');
        loading.classList.add('hidden');

        if (mode === 'upload') {
            stopLiveDetection();
            stopWebcam();
        }
    };

    // ── Drag & Drop ─────────────────────────────────────
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
        uploadZone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });

    ['dragenter', 'dragover'].forEach(evt => {
        uploadZone.addEventListener(evt, () => uploadZone.classList.add('dragover'));
    });

    ['dragleave', 'drop'].forEach(evt => {
        uploadZone.addEventListener(evt, () => uploadZone.classList.remove('dragover'));
    });

    uploadZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function() { handleFiles(this.files); });

    function handleFiles(files) {
        if (!files.length) return;
        const file = files[0];
        // Check MIME type OR file extension (Android often doesn't set MIME type)
        const validExts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.heif'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        const isImage = file.type.startsWith('image/') || validExts.includes(ext);
        if (!isImage) {
            alert('Please upload an image file (JPG, PNG, BMP).');
            return;
        }
        if (engine === 'device') detectFileOnDevice(file);
        // Resize in browser before uploading (large photos crash the server)
        else resizeAndUpload(file);
    }

    function detectFileOnDevice(file) {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = async () => {
            showLoading(session ? 'Detecting on your device…'
                                : 'Loading the model (13 MB, first time only)…');
            try {
                const w = img.naturalWidth, h = img.naturalHeight;
                const { dets, ms } = await detectOnDevice(img, w, h);
                // Draw at most 1600 px on the long side; boxes scale with it.
                const k = Math.min(1, 1600 / Math.max(w, h));
                const scaled = dets.map(d => ({ ...d, box: d.box.map(v => v * k) }));
                const canvas = document.createElement('canvas');
                const data = renderDeviceResult(img, Math.round(w * k), Math.round(h * k), scaled, canvas);
                resultImage.src = canvas.toDataURL('image/jpeg', 0.9);
                showResults(data, `${Math.round(ms)} ms on your device`);
            } catch (err) {
                alert('Error: ' + err.message);
                resetUI();
            } finally {
                URL.revokeObjectURL(url);
            }
        };
        img.onerror = () => {
            URL.revokeObjectURL(url);
            alert('This browser could not open that image. Please use JPG, PNG or WebP.');
        };
        img.src = url;
    }

    function resizeAndUpload(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                const MAX_DIM = 640;
                let w = img.width;
                let h = img.height;

                // Only resize if larger than MAX_DIM
                if (Math.max(w, h) > MAX_DIM) {
                    const scale = MAX_DIM / Math.max(w, h);
                    w = Math.round(w * scale);
                    h = Math.round(h * scale);
                }

                const canvas = document.createElement('canvas');
                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, w, h);

                canvas.toBlob(blob => {
                    const resizedFile = new File([blob], file.name, { type: 'image/jpeg' });
                    sendToAPI(resizedFile);
                }, 'image/jpeg', 0.85);
            };
            // Most browsers other than Safari cannot decode HEIC; without this
            // handler the upload silently never happened.
            img.onerror = function() {
                alert('This browser could not open that image. Please use JPG, PNG or WebP.');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // ── Webcam ──────────────────────────────────────────
    window.startWebcam = async function() {
        try {
            const video = document.getElementById('webcam-video');
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            video.srcObject = webcamStream;
            document.getElementById('webcam-overlay').style.display = 'none';
            document.getElementById('capture-btn').style.display = 'flex';
            document.getElementById('live-btn').style.display = 'flex';
        } catch (err) {
            alert('Could not access camera: ' + err.message);
        }
    };

    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
        }
        const overlay = document.getElementById('webcam-overlay');
        const captureBtn = document.getElementById('capture-btn');
        const liveBtn = document.getElementById('live-btn');
        if (overlay) overlay.style.display = 'flex';
        if (captureBtn) captureBtn.style.display = 'none';
        if (liveBtn) liveBtn.style.display = 'none';
    }

    // Single frame capture
    window.captureFrame = async function() {
        const video = document.getElementById('webcam-video');
        if (engine === 'device') {
            showLoading('Detecting on your device…');
            try {
                const w = video.videoWidth, h = video.videoHeight;
                const { dets, ms } = await detectOnDevice(video, w, h);
                const canvas = document.createElement('canvas');
                const data = renderDeviceResult(video, w, h, dets, canvas);
                resultImage.src = canvas.toDataURL('image/jpeg', 0.9);
                showResults(data, `${Math.round(ms)} ms on your device`);
            } catch (err) {
                alert('Error: ' + err.message);
                resetUI();
            }
            return;
        }
        const canvas = document.getElementById('webcam-canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        canvas.toBlob(blob => {
            sendToAPI(new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' }));
        }, 'image/jpeg', 0.85);
    };

    // ── LIVE CONTINUOUS DETECTION ────────────────────────
    window.toggleLiveDetection = function() {
        if (liveDetectionRunning) {
            stopLiveDetection();
        } else {
            startLiveDetection();
        }
    };

    function startLiveDetection() {
        liveDetectionRunning = true;
        const liveBtn = document.getElementById('live-btn');
        liveBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="6" y="6" width="12" height="12" rx="2"/></svg> Stop Detection`;
        liveBtn.classList.add('live-active');

        document.getElementById('live-results').classList.remove('hidden');
        document.getElementById('capture-btn').style.display = 'none';

        document.getElementById('webcam-video').style.opacity = '0';
        document.getElementById('live-canvas').classList.remove('hidden');

        if (engine === 'device') runLiveLoopOnDevice();
        else runLiveLoop();
    }

    function stopLiveDetection() {
        liveDetectionRunning = false;
        if (liveDetectionAbort) {
            liveDetectionAbort.abort();
            liveDetectionAbort = null;
        }
        const liveBtn = document.getElementById('live-btn');
        if (liveBtn) {
            liveBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16" fill="currentColor"/></svg> Live Detection`;
            liveBtn.classList.remove('live-active');
        }

        const liveResults = document.getElementById('live-results');
        if (liveResults) liveResults.classList.add('hidden');

        const captureBtn = document.getElementById('capture-btn');
        if (captureBtn && webcamStream) captureBtn.style.display = 'flex';

        const video = document.getElementById('webcam-video');
        if (video) video.style.opacity = '1';
        const liveCanvas = document.getElementById('live-canvas');
        if (liveCanvas) liveCanvas.classList.add('hidden');
    }

    async function runLiveLoop() {
        const video = document.getElementById('webcam-video');
        const captureCanvas = document.getElementById('webcam-canvas');
        const liveCanvas = document.getElementById('live-canvas');
        const liveImg = new Image();

        while (liveDetectionRunning && webcamStream) {
            try {
                captureCanvas.width = video.videoWidth;
                captureCanvas.height = video.videoHeight;
                captureCanvas.getContext('2d').drawImage(video, 0, 0);

                const blob = await new Promise(resolve => {
                    captureCanvas.toBlob(resolve, 'image/jpeg', 0.75);
                });

                if (!liveDetectionRunning) break;

                const formData = new FormData();
                formData.append('file', new File([blob], 'frame.jpg', { type: 'image/jpeg' }));

                liveDetectionAbort = new AbortController();
                const res = await fetch('/detect', {
                    method: 'POST',
                    body: formData,
                    signal: liveDetectionAbort.signal
                });

                if (!liveDetectionRunning) break;

                if (!res.ok) {
                    // Show the server's error instead of silently retrying.
                    let msg = 'server returned ' + res.status;
                    try { msg = (await res.json()).detail || msg; } catch (e) { /* not JSON */ }
                    const liveCount = document.getElementById('live-count');
                    if (liveCount) liveCount.textContent = '!';
                    console.warn('Live detection error:', msg);
                    await new Promise(r => setTimeout(r, 1000));
                    continue;
                }

                const data = await res.json();

                if (data.success) {
                    liveImg.src = 'data:image/jpeg;base64,' + data.image;
                    await new Promise(resolve => { liveImg.onload = resolve; });

                    liveCanvas.width = liveImg.width;
                    liveCanvas.height = liveImg.height;
                    liveCanvas.getContext('2d').drawImage(liveImg, 0, 0);

                    updateLiveStats(data);
                }
            } catch (err) {
                if (err.name === 'AbortError') break;
                await new Promise(r => setTimeout(r, 1000));
            }
        }
    }

    // Live detection in the browser: detect, draw, repeat as fast as the device allows.
    async function runLiveLoopOnDevice() {
        const video = document.getElementById('webcam-video');
        const liveCanvas = document.getElementById('live-canvas');
        const fpsEl = document.getElementById('live-fps');
        let smoothed = null;
        while (liveDetectionRunning && webcamStream && engine === 'device') {
            try {
                const t0 = performance.now();
                const w = video.videoWidth, h = video.videoHeight;
                if (!w) { await nextFrame(); continue; }
                const { dets } = await detectOnDevice(video, w, h);
                if (!liveDetectionRunning) break;
                updateLiveStats(renderDeviceResult(video, w, h, dets, liveCanvas));
                const fps = 1000 / (performance.now() - t0);
                smoothed = smoothed == null ? fps : 0.8 * smoothed + 0.2 * fps;
                if (fpsEl) fpsEl.textContent = smoothed.toFixed(1);
                await nextFrame();  // let the page repaint between frames
            } catch (err) {
                console.warn('Live detection error:', err);
                await new Promise(r => setTimeout(r, 1000));
            }
        }
    }

    function nextFrame() {
        return new Promise(r => requestAnimationFrame(() => r()));
    }

    function updateLiveStats(data) {
        const liveCount = document.getElementById('live-count');
        const liveClasses = document.getElementById('live-classes');
        if (liveCount) liveCount.textContent = data.count;

        if (liveClasses) {
            const sorted = Object.entries(data.class_counts).sort((a, b) => b[1] - a[1]);
            liveClasses.replaceChildren(...sorted.map(([name, count]) => {
                const tag = document.createElement('span');
                tag.className = 'live-tag';
                tag.append(name + ' ');
                const strong = document.createElement('strong');
                strong.textContent = count;
                tag.append(strong);
                return tag;
            }));
        }
    }

    // ── API Call (single image) ─────────────────────────
    async function sendToAPI(file) {
        showLoading('Running YOLOv8 on the server… (about a minute on free hosting)');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const t0 = performance.now();
            const res = await fetch('/detect', { method: 'POST', body: formData });

            if (!res.ok) {
                // Try to parse error message from server
                let errMsg = 'Detection failed (server returned ' + res.status + ')';
                try {
                    const errData = await res.json();
                    errMsg = errData.detail || errMsg;
                } catch (e) { /* ignore parse errors */ }
                throw new Error(errMsg);
            }

            const data = await res.json();
            resultImage.src = 'data:image/jpeg;base64,' + data.image;
            showResults(data, `${((performance.now() - t0) / 1000).toFixed(1)} s via server`);
        } catch (err) {
            alert('Error: ' + err.message);
            resetUI();
        }
    }

    // ── Display Results ─────────────────────────────────
    function showLoading(text) {
        document.getElementById('mode-upload').classList.add('hidden');
        document.getElementById('mode-webcam').classList.add('hidden');
        document.querySelector('.mode-tabs').classList.add('hidden');
        document.querySelector('.engine-bar').classList.add('hidden');
        results.classList.add('hidden');
        document.getElementById('loading-text').textContent = text;
        loading.classList.remove('hidden');
    }

    // The caller sets resultImage.src; data carries count and class_counts.
    function showResults(data, timing) {
        loading.classList.add('hidden');
        document.getElementById('timing').textContent = timing || '';
        totalCount.textContent = data.count;

        classList.innerHTML = '';
        const sorted = Object.entries(data.class_counts).sort((a, b) => b[1] - a[1]);

        if (sorted.length === 0) {
            classList.innerHTML = '<li class="class-item"><span class="class-name">No objects detected</span></li>';
        } else {
            sorted.forEach(([name, count]) => {
                const li = document.createElement('li');
                li.className = 'class-item';
                const nameEl = document.createElement('span');
                nameEl.className = 'class-name';
                nameEl.textContent = name;
                const countEl = document.createElement('span');
                countEl.className = 'class-count';
                countEl.textContent = count;
                li.append(nameEl, countEl);
                classList.appendChild(li);
            });
        }

        results.classList.remove('hidden');
    }

    // ── Reset ───────────────────────────────────────────
    window.resetUI = function() {
        results.classList.add('hidden');
        loading.classList.add('hidden');
        document.querySelector('.mode-tabs').classList.remove('hidden');
        document.querySelector('.engine-bar').classList.remove('hidden');

        const activeTab = document.querySelector('.tab.active');
        if (activeTab && activeTab.id === 'tab-webcam') {
            document.getElementById('mode-webcam').classList.remove('hidden');
        } else {
            document.getElementById('mode-upload').classList.remove('hidden');
        }
        fileInput.value = '';
    };
});
