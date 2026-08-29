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
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG, BMP).');
            return;
        }
        sendToAPI(file);
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
    window.captureFrame = function() {
        const video = document.getElementById('webcam-video');
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

        runLiveLoop();
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

    function updateLiveStats(data) {
        const liveCount = document.getElementById('live-count');
        const liveClasses = document.getElementById('live-classes');
        if (liveCount) liveCount.textContent = data.count;

        if (liveClasses) {
            const sorted = Object.entries(data.class_counts).sort((a, b) => b[1] - a[1]);
            liveClasses.innerHTML = sorted.map(([name, count]) =>
                `<span class="live-tag">${name} <strong>${count}</strong></span>`
            ).join('');
        }
    }

    // ── API Call (single image) ─────────────────────────
    async function sendToAPI(file) {
        document.getElementById('mode-upload').classList.add('hidden');
        document.getElementById('mode-webcam').classList.add('hidden');
        document.querySelectorAll('.mode-tabs')[0].classList.add('hidden');
        results.classList.add('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
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
            showResults(data);
        } catch (err) {
            alert('Error: ' + err.message);
            resetUI();
        }
    }

    // ── Display Results ─────────────────────────────────
    function showResults(data) {
        loading.classList.add('hidden');

        resultImage.src = 'data:image/jpeg;base64,' + data.image;
        totalCount.textContent = data.count;

        classList.innerHTML = '';
        const sorted = Object.entries(data.class_counts).sort((a, b) => b[1] - a[1]);

        if (sorted.length === 0) {
            classList.innerHTML = '<li class="class-item"><span class="class-name">No objects detected</span></li>';
        } else {
            sorted.forEach(([name, count]) => {
                const li = document.createElement('li');
                li.className = 'class-item';
                li.innerHTML = `<span class="class-name">${name}</span><span class="class-count">${count}</span>`;
                classList.appendChild(li);
            });
        }

        results.classList.remove('hidden');
    }

    // ── Reset ───────────────────────────────────────────
    window.resetUI = function() {
        results.classList.add('hidden');
        loading.classList.add('hidden');
        document.querySelectorAll('.mode-tabs')[0].classList.remove('hidden');

        const activeTab = document.querySelector('.tab.active');
        if (activeTab && activeTab.id === 'tab-webcam') {
            document.getElementById('mode-webcam').classList.remove('hidden');
        } else {
            document.getElementById('mode-upload').classList.remove('hidden');
        }
        fileInput.value = '';
    };
});
