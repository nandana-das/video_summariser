// Simple Video Summarizer JavaScript

console.log('JavaScript loaded successfully!');

// Global variables
let currentFile = null;
let currentUrl = null;
let results = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing app...');
    initializeApp();
});

function initializeApp() {
    console.log('Initializing app...');
    
    // Setup event listeners
    setupEventListeners();
    
    // Load platforms
    loadPlatforms();
    
    // Update range value
    updateRangeValue();
    
    console.log('App initialized successfully!');
}

function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Tab switching
    const tabButtons = document.querySelectorAll('.tab-btn');
    console.log('Found tab buttons:', tabButtons.length);
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', function(e) {
            console.log('Tab clicked:', e.target.dataset.tab);
            switchTab(e.target.dataset.tab);
        });
    });

    // File upload
    const fileInput = document.getElementById('videoFile');
    const uploadArea = document.getElementById('uploadArea');
    
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
        console.log('File input listener added');
    }
    
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        console.log('Upload area listeners added');
    }

    // Range slider
    const rangeSlider = document.getElementById('maxSentences');
    if (rangeSlider) {
        rangeSlider.addEventListener('input', updateRangeValue);
        console.log('Range slider listener added');
    }

    // URL input
    const urlInput = document.getElementById('videoUrl');
    if (urlInput) {
        urlInput.addEventListener('input', validateUrl);
        console.log('URL input listener added');
    }

    // Process button
    const processBtn = document.getElementById('processBtn');
    if (processBtn) {
        processBtn.addEventListener('click', processVideo);
        console.log('Process button listener added');
    }
}

function switchTab(tabName) {
    console.log('Switching to tab:', tabName);
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    const activeContent = document.getElementById(`${tabName}-tab`);
    if (activeContent) {
        activeContent.classList.add('active');
    }

    // Reset form
    resetForm();
}

function handleFileSelect(event) {
    console.log('File selected:', event.target.files[0]);
    const file = event.target.files[0];
    if (file) {
        currentFile = file;
        displayFileInfo(file);
        updateProcessButton();
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

function handleDragLeave(event) {
    event.currentTarget.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('video/')) {
            currentFile = file;
            displayFileInfo(file);
            updateProcessButton();
        } else {
            showError('Please select a video file.');
        }
    }
}

function displayFileInfo(file) {
    console.log('Displaying file info for:', file.name);
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);
    if (fileInfo) fileInfo.style.display = 'block';
    
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) uploadArea.style.display = 'none';
}

function removeFile() {
    console.log('Removing file');
    currentFile = null;
    const fileInfo = document.getElementById('fileInfo');
    const uploadArea = document.getElementById('uploadArea');
    
    if (fileInfo) fileInfo.style.display = 'none';
    if (uploadArea) uploadArea.style.display = 'block';
    
    const fileInput = document.getElementById('videoFile');
    if (fileInput) fileInput.value = '';
    
    updateProcessButton();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateUrl() {
    const urlInput = document.getElementById('videoUrl');
    const url = urlInput ? urlInput.value.trim() : '';
    console.log('Validating URL:', url);
    
    if (url) {
        currentUrl = url;
    } else {
        currentUrl = null;
    }
    updateProcessButton();
}

function updateRangeValue() {
    const range = document.getElementById('maxSentences');
    const valueDisplay = document.querySelector('.range-value');
    
    if (range && valueDisplay) {
        const value = range.value;
        valueDisplay.textContent = `${value} sentence${value !== '1' ? 's' : ''}`;
    }
}

function updateProcessButton() {
    const processBtn = document.getElementById('processBtn');
    const canProcess = currentFile || currentUrl;
    
    if (processBtn) {
        processBtn.disabled = !canProcess;
        console.log('Process button state:', canProcess ? 'enabled' : 'disabled');
    }
}

async function loadPlatforms() {
    try {
        console.log('Loading platforms...');
        const response = await fetch('/api/platforms');
        const data = await response.json();
        displayPlatforms(data.platforms);
        console.log('Platforms loaded:', data.platforms.length);
    } catch (error) {
        console.error('Error loading platforms:', error);
    }
}

function displayPlatforms(platforms) {
    const container = document.getElementById('platformCards');
    if (container) {
        container.innerHTML = platforms.map(platform => `
            <div class="platform-card" style="border-color: ${platform.color}">
                <div class="platform-icon" style="color: ${platform.color}">${platform.icon}</div>
                <div class="platform-name">${platform.name}</div>
                <div class="platform-desc">${platform.description}</div>
            </div>
        `).join('');
        console.log('Platforms displayed');
    }
}

async function processVideo() {
    console.log('Processing video...');
    
    if (!currentFile && !currentUrl) {
        showError('Please select a video file or enter a URL.');
        return;
    }

    showLoading();

    try {
        let response;
        
        if (currentFile) {
            console.log('Processing file:', currentFile.name);
            response = await processFile();
        } else {
            console.log('Processing URL:', currentUrl);
            response = await processUrl();
        }

        if (response.ok) {
            const data = await response.json();
            console.log('Processing successful:', data);
            displayResults(data.data);
        } else {
            const error = await response.json();
            console.error('Processing failed:', error);
            showError(error.error || 'Processing failed.');
        }
    } catch (error) {
        console.error('Error processing video:', error);
        showError('An error occurred while processing the video.');
    } finally {
        hideLoading();
    }
}

async function processFile() {
    const formData = new FormData();
    formData.append('video', currentFile);
    formData.append('max_sentences', document.getElementById('maxSentences').value);
    formData.append('processing_mode', document.getElementById('processingMode').value);

    return fetch('/api/process-video', {
        method: 'POST',
        body: formData
    });
}

async function processUrl() {
    const data = {
        url: currentUrl,
        max_sentences: parseInt(document.getElementById('maxSentences').value),
        processing_mode: document.getElementById('processingMode').value
    };

    return fetch('/api/process-url', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
}

function showLoading() {
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    
    if (loadingSection) loadingSection.style.display = 'block';
    if (resultsSection) resultsSection.style.display = 'none';
}

function hideLoading() {
    const loadingSection = document.getElementById('loadingSection');
    if (loadingSection) loadingSection.style.display = 'none';
}

function displayResults(data) {
    console.log('Displaying results:', data);
    results = data;
    
    // Update summary
    const summaryText = document.getElementById('summaryText');
    if (summaryText) summaryText.textContent = data.summary;
    
    // Update action items
    const actionItems = document.getElementById('actionItems');
    if (actionItems) {
        if (data.action_items && data.action_items.length > 0) {
            actionItems.innerHTML = data.action_items.map((item, index) => `
                <div class="action-item">
                    <strong>${index + 1}.</strong> ${item}
                </div>
            `).join('');
        } else {
            actionItems.innerHTML = '<p>No action items found in the content.</p>';
        }
    }
    
    // Update keywords
    const keywords = document.getElementById('keywords');
    if (keywords) {
        if (data.keywords && data.keywords.length > 0) {
            keywords.innerHTML = data.keywords.map(keyword => `
                <span class="keyword-tag">${keyword}</span>
            `).join('');
        } else {
            keywords.innerHTML = '<p>No keywords extracted.</p>';
        }
    }
    
    // Update transcript
    const transcriptText = document.getElementById('transcriptText');
    if (transcriptText) transcriptText.value = data.transcript || 'No transcript available.';
    
    // Update stats
    const metadata = data.metadata || {};
    const summaryCount = document.getElementById('summaryCount');
    const keywordCount = document.getElementById('keywordCount');
    const actionCount = document.getElementById('actionCount');
    const compressionRatio = document.getElementById('compressionRatio');
    
    if (summaryCount) summaryCount.textContent = metadata.summary_sentence_count || 0;
    if (keywordCount) keywordCount.textContent = metadata.keyword_count || 0;
    if (actionCount) actionCount.textContent = metadata.action_item_count || 0;
    if (compressionRatio) compressionRatio.textContent = 
        metadata.compression_ratio ? `${Math.round(metadata.compression_ratio * 100)}%` : '0%';
    
    // Show results
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

function resetForm() {
    currentFile = null;
    currentUrl = null;
    results = null;
    
    const fileInput = document.getElementById('videoFile');
    const urlInput = document.getElementById('videoUrl');
    const fileInfo = document.getElementById('fileInfo');
    const uploadArea = document.getElementById('uploadArea');
    const resultsSection = document.getElementById('resultsSection');
    
    if (fileInput) fileInput.value = '';
    if (urlInput) urlInput.value = '';
    if (fileInfo) fileInfo.style.display = 'none';
    if (uploadArea) uploadArea.style.display = 'block';
    if (resultsSection) resultsSection.style.display = 'none';
    
    updateProcessButton();
}

function showError(message) {
    alert(`Error: ${message}`);
}

function showSuccess(message) {
    alert(message);
}

// Make functions globally available for onclick handlers
window.removeFile = removeFile;
window.processVideo = processVideo;
