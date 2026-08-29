document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultImage = document.getElementById('result-image');
    const totalCount = document.getElementById('total-count');
    const classList = document.getElementById('class-list');

    // ── Mode Switching ──────────────────────────────────
    window.switchMode = function(mode) {
        document.getElementById('tab-upload').classList.toggle('active', mode === 'upload');
        document.getElementById('tab-webcam').classList.toggle('active', mode === 'webcam');
        document.getElementById('mode-upload').classList.toggle('hidden', mode !== 'upload');
        document.getElementById('mode-webcam').classList.toggle('hidden', mode !== 'webcam');
        results.classList.add('hidden');
        loading.classList.add('hidden');

        // Stop webcam if switching away
        if (mode === 'upload') {
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

    uploadZone.addEventListener('drop', e => {
        handleFiles(e.dataTransfer.files);
    });

    uploadZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

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
    let webcamStream = null;

    window.startWebcam = async function() {
        try {
            const video = document.getElementById('webcam-video');
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            video.srcObject = webcamStream;
            document.getElementById('webcam-overlay').style.display = 'none';
            document.getElementById('capture-btn').style.display = 'flex';
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
        if (overlay) overlay.style.display = 'flex';
        if (captureBtn) captureBtn.style.display = 'none';
    }

    window.captureFrame = function() {
        const video = document.getElementById('webcam-video');
        const canvas = document.getElementById('webcam-canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        canvas.toBlob(blob => {
            const file = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
            sendToAPI(file);
        }, 'image/jpeg', 0.92);
    };

    // ── API Call ─────────────────────────────────────────
    async function sendToAPI(file) {
        // Show loading
        document.getElementById('mode-upload').classList.add('hidden');
        document.getElementById('mode-webcam').classList.add('hidden');
        document.querySelectorAll('.mode-tabs')[0].classList.add('hidden');
        results.classList.add('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/detect', { method: 'POST', body: formData });
            const data = await res.json();

            if (!res.ok) throw new Error(data.detail || 'Detection failed.');

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
