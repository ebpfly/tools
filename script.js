// Spectrum Explorer JavaScript
class SpectrumExplorer {
    constructor() {
        this.canvas = document.getElementById('spectrumCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.data = [];
        this.viewBox = { minX: 0, maxX: 100, minY: 0, maxY: 100 };
        this.isDragging = false;
        this.lastMousePos = { x: 0, y: 0 };
        this.padding = { top: 40, right: 40, bottom: 60, left: 70 };

        // Touch support
        this.touches = [];
        this.lastTouchDistance = 0;

        this.setupCanvas();
        this.setupEventListeners();
    }

    setupCanvas() {
        // Set canvas size
        this.canvas.width = this.canvas.offsetWidth * window.devicePixelRatio;
        this.canvas.height = 500 * window.devicePixelRatio;
        this.canvas.style.width = this.canvas.offsetWidth + 'px';
        this.canvas.style.height = '500px';
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

        // Set actual drawing dimensions
        this.width = this.canvas.offsetWidth;
        this.height = 500;
    }

    setupEventListeners() {
        // File input
        document.getElementById('dataInput').addEventListener('change', (e) => this.loadFile(e));

        // Buttons
        document.getElementById('loadSample').addEventListener('click', () => this.loadSampleData());
        document.getElementById('resetZoom').addEventListener('click', () => this.resetZoom());
        document.getElementById('exportData').addEventListener('click', () => this.exportData());

        // Checkboxes
        document.getElementById('showGrid').addEventListener('change', () => this.draw());
        document.getElementById('showMarkers').addEventListener('change', () => this.draw());

        // Canvas interactions - Mouse
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mouseup', () => this.handleMouseUp());
        this.canvas.addEventListener('mouseleave', () => this.handleMouseUp());
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e));

        // Canvas interactions - Touch
        this.canvas.addEventListener('touchstart', (e) => this.handleTouchStart(e), { passive: false });
        this.canvas.addEventListener('touchmove', (e) => this.handleTouchMove(e), { passive: false });
        this.canvas.addEventListener('touchend', (e) => this.handleTouchEnd(e), { passive: false });
        this.canvas.addEventListener('touchcancel', (e) => this.handleTouchEnd(e), { passive: false });

        // Window resize
        window.addEventListener('resize', () => {
            this.setupCanvas();
            this.draw();
        });
    }

    loadSampleData() {
        // Generate sample reflectance spectrum data (visible range)
        this.data = [];
        for (let wavelength = 380; wavelength <= 780; wavelength += 5) {
            // Create a sample spectrum with some interesting features
            let reflectance = 50;

            // Add some peaks and valleys
            if (wavelength > 450 && wavelength < 500) {
                reflectance += 20 * Math.sin((wavelength - 450) / 50 * Math.PI);
            }
            if (wavelength > 600 && wavelength < 650) {
                reflectance -= 15;
            }

            // Add some noise
            reflectance += (Math.random() - 0.5) * 5;

            // Clamp between 0 and 100
            reflectance = Math.max(0, Math.min(100, reflectance));

            this.data.push({ wavelength, reflectance });
        }

        this.resetZoom();
        this.updateStatistics();
        this.draw();
    }

    loadFile(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            this.parseCSV(text);
        };
        reader.readAsText(file);
    }

    parseCSV(text) {
        const lines = text.trim().split('\n');
        this.data = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();

            // Skip empty lines and potential headers
            if (!line || line.toLowerCase().includes('wavelength')) continue;

            const [wavelength, reflectance] = line.split(/[,\t]/).map(v => parseFloat(v.trim()));

            if (!isNaN(wavelength) && !isNaN(reflectance)) {
                this.data.push({ wavelength, reflectance });
            }
        }

        if (this.data.length > 0) {
            this.resetZoom();
            this.updateStatistics();
            this.draw();
        } else {
            alert('No valid data found in file. Expected format: wavelength,reflectance');
        }
    }

    resetZoom() {
        if (this.data.length === 0) return;

        const wavelengths = this.data.map(d => d.wavelength);
        const reflectances = this.data.map(d => d.reflectance);

        this.viewBox.minX = Math.min(...wavelengths);
        this.viewBox.maxX = Math.max(...wavelengths);
        this.viewBox.minY = Math.min(...reflectances) - 5;
        this.viewBox.maxY = Math.max(...reflectances) + 5;

        this.draw();
    }

    updateStatistics() {
        if (this.data.length === 0) {
            document.getElementById('dataCount').textContent = '0';
            document.getElementById('minReflectance').textContent = '-';
            document.getElementById('maxReflectance').textContent = '-';
            document.getElementById('meanReflectance').textContent = '-';
            document.getElementById('wavelengthRange').textContent = '-';
            return;
        }

        const reflectances = this.data.map(d => d.reflectance);
        const wavelengths = this.data.map(d => d.wavelength);

        const min = Math.min(...reflectances).toFixed(2);
        const max = Math.max(...reflectances).toFixed(2);
        const mean = (reflectances.reduce((a, b) => a + b, 0) / reflectances.length).toFixed(2);
        const range = `${Math.min(...wavelengths)} - ${Math.max(...wavelengths)}`;

        document.getElementById('dataCount').textContent = this.data.length;
        document.getElementById('minReflectance').textContent = min;
        document.getElementById('maxReflectance').textContent = max;
        document.getElementById('meanReflectance').textContent = mean;
        document.getElementById('wavelengthRange').textContent = range;
    }

    draw() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.width, this.height);

        // Draw grid
        if (document.getElementById('showGrid').checked) {
            this.drawGrid();
        }

        // Draw axes
        this.drawAxes();

        // Draw spectrum
        if (this.data.length > 0) {
            this.drawSpectrum();
        }
    }

    drawGrid() {
        this.ctx.strokeStyle = '#e0e0e0';
        this.ctx.lineWidth = 0.5;

        const plotWidth = this.width - this.padding.left - this.padding.right;
        const plotHeight = this.height - this.padding.top - this.padding.bottom;

        // Vertical grid lines
        for (let i = 0; i <= 10; i++) {
            const x = this.padding.left + (plotWidth * i / 10);
            this.ctx.beginPath();
            this.ctx.moveTo(x, this.padding.top);
            this.ctx.lineTo(x, this.height - this.padding.bottom);
            this.ctx.stroke();
        }

        // Horizontal grid lines
        for (let i = 0; i <= 10; i++) {
            const y = this.padding.top + (plotHeight * i / 10);
            this.ctx.beginPath();
            this.ctx.moveTo(this.padding.left, y);
            this.ctx.lineTo(this.width - this.padding.right, y);
            this.ctx.stroke();
        }
    }

    drawAxes() {
        const plotWidth = this.width - this.padding.left - this.padding.right;
        const plotHeight = this.height - this.padding.top - this.padding.bottom;

        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        this.ctx.fillStyle = '#333';
        this.ctx.font = '12px Arial';

        // X-axis
        this.ctx.beginPath();
        this.ctx.moveTo(this.padding.left, this.height - this.padding.bottom);
        this.ctx.lineTo(this.width - this.padding.right, this.height - this.padding.bottom);
        this.ctx.stroke();

        // Y-axis
        this.ctx.beginPath();
        this.ctx.moveTo(this.padding.left, this.padding.top);
        this.ctx.lineTo(this.padding.left, this.height - this.padding.bottom);
        this.ctx.stroke();

        // X-axis labels
        this.ctx.textAlign = 'center';
        for (let i = 0; i <= 10; i++) {
            const x = this.padding.left + (plotWidth * i / 10);
            const value = this.viewBox.minX + (this.viewBox.maxX - this.viewBox.minX) * i / 10;
            this.ctx.fillText(value.toFixed(0), x, this.height - this.padding.bottom + 20);
        }

        // Y-axis labels
        this.ctx.textAlign = 'right';
        for (let i = 0; i <= 10; i++) {
            const y = this.padding.top + (plotHeight * i / 10);
            const value = this.viewBox.maxY - (this.viewBox.maxY - this.viewBox.minY) * i / 10;
            this.ctx.fillText(value.toFixed(1), this.padding.left - 10, y + 4);
        }

        // Axis titles
        this.ctx.textAlign = 'center';
        this.ctx.font = '14px Arial';
        this.ctx.fillText('Wavelength (nm)', this.width / 2, this.height - 10);

        this.ctx.save();
        this.ctx.translate(15, this.height / 2);
        this.ctx.rotate(-Math.PI / 2);
        this.ctx.fillText('Reflectance (%)', 0, 0);
        this.ctx.restore();
    }

    drawSpectrum() {
        const plotWidth = this.width - this.padding.left - this.padding.right;
        const plotHeight = this.height - this.padding.top - this.padding.bottom;

        // Draw line
        this.ctx.strokeStyle = '#2196F3';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();

        this.data.forEach((point, index) => {
            const x = this.padding.left + ((point.wavelength - this.viewBox.minX) / (this.viewBox.maxX - this.viewBox.minX)) * plotWidth;
            const y = this.height - this.padding.bottom - ((point.reflectance - this.viewBox.minY) / (this.viewBox.maxY - this.viewBox.minY)) * plotHeight;

            if (index === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        });

        this.ctx.stroke();

        // Draw markers
        if (document.getElementById('showMarkers').checked) {
            this.ctx.fillStyle = '#2196F3';
            this.data.forEach(point => {
                const x = this.padding.left + ((point.wavelength - this.viewBox.minX) / (this.viewBox.maxX - this.viewBox.minX)) * plotWidth;
                const y = this.height - this.padding.bottom - ((point.reflectance - this.viewBox.minY) / (this.viewBox.maxY - this.viewBox.minY)) * plotHeight;

                this.ctx.beginPath();
                this.ctx.arc(x, y, 3, 0, Math.PI * 2);
                this.ctx.fill();
            });
        }
    }

    handleMouseMove(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        if (this.isDragging) {
            const dx = x - this.lastMousePos.x;
            const dy = y - this.lastMousePos.y;
            this.pan(dx, dy);
            this.lastMousePos = { x, y };
        } else {
            this.updateCrosshair(x, y);
        }
    }

    handleMouseDown(event) {
        const rect = this.canvas.getBoundingClientRect();
        this.isDragging = true;
        this.lastMousePos = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
        this.canvas.style.cursor = 'grabbing';
    }

    handleMouseUp() {
        this.isDragging = false;
        this.canvas.style.cursor = 'default';
    }

    handleWheel(event) {
        event.preventDefault();
        const delta = event.deltaY;
        const zoomFactor = delta > 0 ? 1.1 : 0.9;

        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const plotWidth = this.width - this.padding.left - this.padding.right;
        const centerRatio = (x - this.padding.left) / plotWidth;

        this.zoom(zoomFactor, centerRatio);
    }

    zoom(factor, centerRatio = 0.5) {
        const rangeX = this.viewBox.maxX - this.viewBox.minX;
        const centerX = this.viewBox.minX + rangeX * centerRatio;
        const newRangeX = rangeX * factor;

        this.viewBox.minX = centerX - newRangeX * centerRatio;
        this.viewBox.maxX = centerX + newRangeX * (1 - centerRatio);

        this.draw();
    }

    pan(dx, dy) {
        const plotWidth = this.width - this.padding.left - this.padding.right;
        const plotHeight = this.height - this.padding.top - this.padding.bottom;

        const rangeX = this.viewBox.maxX - this.viewBox.minX;
        const rangeY = this.viewBox.maxY - this.viewBox.minY;

        const deltaX = -(dx / plotWidth) * rangeX;
        const deltaY = (dy / plotHeight) * rangeY;

        this.viewBox.minX += deltaX;
        this.viewBox.maxX += deltaX;
        this.viewBox.minY += deltaY;
        this.viewBox.maxY += deltaY;

        this.draw();
    }

    updateCrosshair(mouseX, mouseY) {
        if (this.data.length === 0) return;

        const plotWidth = this.width - this.padding.left - this.padding.right;
        const plotHeight = this.height - this.padding.top - this.padding.bottom;

        // Check if mouse is in plot area
        if (mouseX < this.padding.left || mouseX > this.width - this.padding.right ||
            mouseY < this.padding.top || mouseY > this.height - this.padding.bottom) {
            document.getElementById('currentWavelength').textContent = '-';
            document.getElementById('currentReflectance').textContent = '-';
            return;
        }

        // Convert mouse position to data coordinates
        const wavelength = this.viewBox.minX + ((mouseX - this.padding.left) / plotWidth) * (this.viewBox.maxX - this.viewBox.minX);

        // Find nearest data point
        let nearest = this.data[0];
        let minDist = Math.abs(nearest.wavelength - wavelength);

        for (const point of this.data) {
            const dist = Math.abs(point.wavelength - wavelength);
            if (dist < minDist) {
                minDist = dist;
                nearest = point;
            }
        }

        document.getElementById('currentWavelength').textContent = nearest.wavelength.toFixed(1);
        document.getElementById('currentReflectance').textContent = nearest.reflectance.toFixed(2);
    }

    exportData() {
        if (this.data.length === 0) {
            alert('No data to export');
            return;
        }

        let csv = 'wavelength,reflectance\n';
        this.data.forEach(point => {
            csv += `${point.wavelength},${point.reflectance}\n`;
        });

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'spectrum_data.csv';
        a.click();
        URL.revokeObjectURL(url);
    }

    // Touch event handlers
    handleTouchStart(event) {
        event.preventDefault();
        this.touches = Array.from(event.touches);

        if (this.touches.length === 1) {
            // Single touch - start dragging
            const rect = this.canvas.getBoundingClientRect();
            this.isDragging = true;
            this.lastMousePos = {
                x: this.touches[0].clientX - rect.left,
                y: this.touches[0].clientY - rect.top
            };
        } else if (this.touches.length === 2) {
            // Two touches - prepare for pinch zoom
            this.isDragging = false;
            this.lastTouchDistance = this.getTouchDistance(this.touches[0], this.touches[1]);
        }
    }

    handleTouchMove(event) {
        event.preventDefault();
        this.touches = Array.from(event.touches);

        if (this.touches.length === 1 && this.isDragging) {
            // Single touch - pan
            const rect = this.canvas.getBoundingClientRect();
            const x = this.touches[0].clientX - rect.left;
            const y = this.touches[0].clientY - rect.top;

            const dx = x - this.lastMousePos.x;
            const dy = y - this.lastMousePos.y;
            this.pan(dx, dy);
            this.lastMousePos = { x, y };
        } else if (this.touches.length === 2) {
            // Two touches - pinch zoom
            const currentDistance = this.getTouchDistance(this.touches[0], this.touches[1]);

            if (this.lastTouchDistance > 0) {
                const zoomFactor = this.lastTouchDistance / currentDistance;

                // Calculate center point between two touches
                const rect = this.canvas.getBoundingClientRect();
                const centerX = ((this.touches[0].clientX + this.touches[1].clientX) / 2) - rect.left;
                const plotWidth = this.width - this.padding.left - this.padding.right;
                const centerRatio = (centerX - this.padding.left) / plotWidth;

                this.zoom(zoomFactor, Math.max(0, Math.min(1, centerRatio)));
            }

            this.lastTouchDistance = currentDistance;
        }
    }

    handleTouchEnd(event) {
        event.preventDefault();
        this.touches = Array.from(event.touches);

        if (this.touches.length === 0) {
            this.isDragging = false;
            this.lastTouchDistance = 0;
        } else if (this.touches.length === 1) {
            // Switch back to single touch mode
            const rect = this.canvas.getBoundingClientRect();
            this.lastMousePos = {
                x: this.touches[0].clientX - rect.left,
                y: this.touches[0].clientY - rect.top
            };
            this.isDragging = true;
            this.lastTouchDistance = 0;
        }
    }

    getTouchDistance(touch1, touch2) {
        const dx = touch1.clientX - touch2.clientX;
        const dy = touch1.clientY - touch2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }
}

// Initialize the explorer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const explorer = new SpectrumExplorer();
    explorer.loadSampleData();
});
