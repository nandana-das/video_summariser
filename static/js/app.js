// Video Summarizer App JavaScript

class VideoSummarizer {
    constructor() {
        this.currentFile = null;
        this.currentUrl = null;
        this.results = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadPlatforms();
        this.updateRangeValue();
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // File upload
        const fileInput = document.getElementById('videoFile');
        const uploadArea = document.getElementById('uploadArea');
        
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Range slider
        document.getElementById('maxSentences').addEventListener('input', (e) => {
            this.updateRangeValue();
        });

        // URL input
        document.getElementById('videoUrl').addEventListener('input', (e) => {
            this.validateUrl();
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        // Reset form
        this.resetForm();
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.currentFile = file;
            this.displayFileInfo(file);
            this.updateProcessButton();
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.currentTarget.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('video/')) {
                this.currentFile = file;
                this.displayFileInfo(file);
                this.updateProcessButton();
            } else {
                this.showError('Please select a video file.');
            }
        }
    }

    displayFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        
        fileName.textContent = file.name;
        fileSize.textContent = this.formatFileSize(file.size);
        
        fileInfo.style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
    }

    removeFile() {
        this.currentFile = null;
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('uploadArea').style.display = 'block';
        document.getElementById('videoFile').value = '';
        this.updateProcessButton();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    validateUrl() {
        const url = document.getElementById('videoUrl').value.trim();
        if (url) {
            this.currentUrl = url;
            this.updateProcessButton();
        } else {
            this.currentUrl = null;
            this.updateProcessButton();
        }
    }

    updateRangeValue() {
        const range = document.getElementById('maxSentences');
        const value = range.value;
        const valueDisplay = document.querySelector('.range-value');
        valueDisplay.textContent = `${value} sentence${value !== '1' ? 's' : ''}`;
    }

    updateProcessButton() {
        const processBtn = document.getElementById('processBtn');
        const canProcess = this.currentFile || this.currentUrl;
        processBtn.disabled = !canProcess;
    }

    async loadPlatforms() {
        try {
            const response = await fetch('/api/platforms');
            const data = await response.json();
            this.displayPlatforms(data.platforms);
        } catch (error) {
            console.error('Error loading platforms:', error);
        }
    }

    displayPlatforms(platforms) {
        const container = document.getElementById('platformCards');
        container.innerHTML = platforms.map(platform => `
            <div class="platform-card" style="border-color: ${platform.color}">
                <div class="platform-icon" style="color: ${platform.color}">${platform.icon}</div>
                <div class="platform-name">${platform.name}</div>
                <div class="platform-desc">${platform.description}</div>
            </div>
        `).join('');
    }

    async processVideo() {
        if (!this.currentFile && !this.currentUrl) {
            this.showError('Please select a video file or enter a URL.');
            return;
        }

        this.showLoading();

        try {
            let response;
            
            if (this.currentFile) {
                response = await this.processFile();
            } else {
                response = await this.processUrl();
            }

            if (response.ok) {
                const data = await response.json();
                this.displayResults(data.data);
            } else {
                const error = await response.json();
                this.showError(error.error || 'Processing failed.');
            }
        } catch (error) {
            console.error('Error processing video:', error);
            this.showError('An error occurred while processing the video.');
        } finally {
            this.hideLoading();
        }
    }

    async processFile() {
        const formData = new FormData();
        formData.append('video', this.currentFile);
        formData.append('max_sentences', document.getElementById('maxSentences').value);
        formData.append('processing_mode', document.getElementById('processingMode').value);

        return fetch('/api/process-video', {
            method: 'POST',
            body: formData
        });
    }

    async processUrl() {
        const data = {
            url: this.currentUrl,
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

    showLoading() {
        document.getElementById('loadingSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
    }

    hideLoading() {
        document.getElementById('loadingSection').style.display = 'none';
    }

    displayResults(data) {
        this.results = data;
        
        // Update summary
        document.getElementById('summaryText').textContent = data.summary;
        
        // Update action items
        const actionItems = document.getElementById('actionItems');
        if (data.action_items && data.action_items.length > 0) {
            actionItems.innerHTML = data.action_items.map((item, index) => `
                <div class="action-item">
                    <strong>${index + 1}.</strong> ${item}
                </div>
            `).join('');
        } else {
            actionItems.innerHTML = '<p>No action items found in the content.</p>';
        }
        
        // Update keywords
        const keywords = document.getElementById('keywords');
        if (data.keywords && data.keywords.length > 0) {
            keywords.innerHTML = data.keywords.map(keyword => `
                <span class="keyword-tag">${keyword}</span>
            `).join('');
        } else {
            keywords.innerHTML = '<p>No keywords extracted.</p>';
        }
        
        // Update transcript
        document.getElementById('transcriptText').value = data.transcript || 'No transcript available.';
        
        // Update stats
        const metadata = data.metadata || {};
        document.getElementById('summaryCount').textContent = metadata.summary_sentence_count || 0;
        document.getElementById('keywordCount').textContent = metadata.keyword_count || 0;
        document.getElementById('actionCount').textContent = metadata.action_item_count || 0;
        document.getElementById('compressionRatio').textContent = 
            metadata.compression_ratio ? `${Math.round(metadata.compression_ratio * 100)}%` : '0%';
        
        // Show results
        document.getElementById('resultsSection').style.display = 'block';
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ 
            behavior: 'smooth' 
        });
    }

    downloadResults() {
        if (!this.results) return;
        
        const data = {
            summary: this.results.summary,
            action_items: this.results.action_items || [],
            keywords: this.results.keywords || [],
            transcript: this.results.transcript || '',
            metadata: this.results.metadata || {}
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `video_summary_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    shareResults() {
        if (!this.results) return;
        
        const summary = this.results.summary;
        const url = window.location.href;
        
        if (navigator.share) {
            navigator.share({
                title: 'Video Summary',
                text: summary,
                url: url
            });
        } else {
            // Fallback: copy to clipboard
            const text = `Video Summary:\n\n${summary}\n\nView full results: ${url}`;
            navigator.clipboard.writeText(text).then(() => {
                this.showSuccess('Summary copied to clipboard!');
            });
        }
    }

    resetForm() {
        this.currentFile = null;
        this.currentUrl = null;
        this.results = null;
        
        document.getElementById('videoFile').value = '';
        document.getElementById('videoUrl').value = '';
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('uploadArea').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        
        this.updateProcessButton();
    }

    showError(message) {
        // Simple error display - you can enhance this with a proper modal
        alert(`Error: ${message}`);
    }

    showSuccess(message) {
        // Simple success display - you can enhance this with a proper notification
        alert(message);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VideoSummarizer();
});
