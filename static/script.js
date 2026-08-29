document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsSection = document.getElementById('results-section');
    const resultImage = document.getElementById('result-image');
    const totalObjects = document.getElementById('total-objects');
    const classBreakdown = document.getElementById('class-breakdown');
    const resetBtn = document.getElementById('reset-btn');

    // Drag and drop event listeners
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // Handle click to upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        uploadImage(file);
    }

    async function uploadImage(file) {
        // UI transitions
        uploadZone.style.display = 'none';
        resultsSection.style.display = 'none';
        loadingSpinner.style.display = 'flex';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'An error occurred during detection.');
            }

            displayResults(data);

        } catch (error) {
            alert(`Error: ${error.message}`);
            // Reset UI
            loadingSpinner.style.display = 'none';
            uploadZone.style.display = 'block';
        }
    }

    function displayResults(data) {
        loadingSpinner.style.display = 'none';
        
        // Update image
        resultImage.src = `data:image/jpeg;base64,${data.image}`;
        
        // Update stats
        totalObjects.textContent = data.count;
        
        // Update class breakdown
        classBreakdown.innerHTML = '';
        if (Object.keys(data.class_counts).length === 0) {
            classBreakdown.innerHTML = '<li class="class-item"><span class="class-name">No objects detected</span></li>';
        } else {
            // Sort by count descending
            const sortedClasses = Object.entries(data.class_counts)
                .sort((a, b) => b[1] - a[1]);
                
            sortedClasses.forEach(([className, count]) => {
                const li = document.createElement('li');
                li.className = 'class-item';
                li.innerHTML = `
                    <span class="class-name">${className}</span>
                    <span class="class-count">${count}</span>
                `;
                classBreakdown.appendChild(li);
            });
        }
        
        // Show results with display grid
        resultsSection.style.display = 'grid';
    }

    // Reset button
    resetBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        uploadZone.style.display = 'block';
        fileInput.value = ''; // Clear input
    });
});
