# GattoNero AI Assistant - Photoshop Integration
## Chapter 3: CEP Panel Interface

> **Status:** 📝 AKTYWNY  
> **Poprzedni:** [Chapter 2 - CEP Extension Setup](./gatto-WORKING-04-photoshop-chapter2.md)  
> **Następny:** [Chapter 4 - ExtendScript Core Functions](./gatto-WORKING-04-photoshop-chapter4.md)  
> **Spis treści:** [gatto-WORKING-04-photoshop-toc.md](./gatto-WORKING-04-photoshop-toc.md)

---

## 🖥️ CEP PANEL INTERFACE

### Main Panel HTML Structure (index.html)

> **✅ POPRAWKA:** Główny plik to `index.html`, nie `client.jsx`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GattoNero Color Matching</title>
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/components.css">
</head>
<body>
    <div id="app" class="container">
        <!-- Header -->
        <header class="header">
            <div class="logo">
                <img src="assets/icons/gattonero-icon.png" alt="GattoNero" width="24" height="24">
                <h1>GattoNero</h1>
            </div>
            <div class="status-indicator" id="serverStatus">
                <span class="status-dot offline"></span>
                <span class="status-text">Connecting...</span>
            </div>
        </header>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Function Selection -->
            <section class="function-section">
                <h2>Color Functions</h2>
                <div class="function-grid">
                    <div class="function-card" data-function="color-matching">
                        <h3>🎨 Color Matching</h3>
                        <p>Match colors between layers using AI algorithms</p>
                    </div>
                    <div class="function-card" data-function="palette-analysis">
                        <h3>🎭 Palette Analysis</h3>
                        <p>Extract and analyze color palettes from layers</p>
                    </div>
                    <div class="function-card" data-function="color-harmony">
                        <h3>🎵 Color Harmony</h3>
                        <p>Generate harmonious color schemes</p>
                    </div>
                    <div class="function-card" data-function="batch-processing">
                        <h3>⚡ Batch Processing</h3>
                        <p>Process multiple layers simultaneously</p>
                    </div>
                </div>
            </section>

            <!-- Color Matching Panel -->
            <section class="panel" id="colorMatchingPanel" style="display: none;">
                <h2>Color Matching</h2>
                
                <!-- Layer Selection -->
                <div class="layer-selection">
                    <label for="sourceLayer">Source Layer:</label>
                    <select id="sourceLayer" class="layer-dropdown">
                        <option value="">Select source layer...</option>
                    </select>
                    
                    <label for="targetLayer">Target Layer:</label>
                    <select id="targetLayer" class="layer-dropdown">
                        <option value="">Select target layer...</option>
                    </select>
                </div>

                <!-- Method Selection -->
                <div class="method-selection">
                    <label for="matchingMethod">Matching Method:</label>
                    <select id="matchingMethod">
                        <option value="histogram">Histogram Matching</option>
                        <option value="lab_delta">LAB Delta E</option>
                        <option value="neural_transfer">Neural Transfer</option>
                        <option value="statistical">Statistical Transfer</option>
                    </select>
                </div>

                <!-- Parameters -->
                <div class="method-params">
                    <label for="strength">Strength:</label>
                    <div class="slider-container">
                        <input type="range" id="strength" min="0" max="100" value="80">
                        <span class="slider-value">80%</span>
                    </div>
                    
                    <label for="preserveDetails">Preserve Details:</label>
                    <div class="slider-container">
                        <input type="range" id="preserveDetails" min="0" max="100" value="60">
                        <span class="slider-value">60%</span>
                    </div>
                </div>

                <!-- Actions -->
                <div class="actions">
                    <button class="btn-primary" id="processColorMatching">
                        Process Color Matching
                    </button>
                    <button class="btn-secondary" id="previewColorMatching">
                        Preview Changes
                    </button>
                </div>
            </section>

            <!-- Palette Analysis Panel -->
            <section class="panel" id="paletteAnalysisPanel" style="display: none;">
                <h2>Palette Analysis</h2>
                
                <!-- Layer Selection -->
                <div class="layer-selection">
                    <label for="analysisLayer">Layer to Analyze:</label>
                    <select id="analysisLayer" class="layer-dropdown">
                        <option value="">Select layer...</option>
                    </select>
                </div>

                <!-- Palette Parameters -->
                <div class="palette-params">
                    <label for="colorCount">Number of Colors:</label>
                    <div class="slider-container">
                        <input type="range" id="colorCount" min="2" max="20" value="8">
                        <span class="slider-value">8</span>
                    </div>
                    
                    <label for="tolerance">Color Tolerance:</label>
                    <div class="slider-container">
                        <input type="range" id="tolerance" min="1" max="50" value="10">
                        <span class="slider-value">10</span>
                    </div>
                </div>

                <!-- Actions -->
                <div class="actions">
                    <button class="btn-primary" id="analyzePalette">
                        Analyze Palette
                    </button>
                    <button class="btn-secondary" id="savePalette">
                        Save Palette
                    </button>
                </div>

                <!-- Palette Display -->
                <div class="palette-display" id="paletteResults" style="display: none;">
                    <!-- Colors will be populated here -->
                </div>
            </section>

            <!-- Progress & Status -->
            <section class="status-section">
                <div class="progress-container" id="progressContainer" style="display: none;">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <div class="progress-text" id="progressText">Processing...</div>
                </div>
                
                <div class="log-container" id="logContainer">
                    <div class="log-header">
                        <h3>Activity Log</h3>
                        <button class="btn-small" id="clearLog">Clear</button>
                    </div>
                    <div class="log-content" id="logContent">
                        <!-- Log entries will appear here -->
                    </div>
                </div>
            </section>
        </main>
    </div>

    <!-- JavaScript -->
    <script src="js/main.js"></script>
    <script src="js/ui.js"></script>
    <script src="js/api.js"></script>
    <script src="js/utils.js"></script>
</body>
</html>
```

---

## 🎨 CSS STYLING

### Main Styles (css/main.css)

```css
/* CSS Variables for Dark Theme */
:root {
    --primary-color: #2196F3;
    --secondary-color: #FF5722;
    --background: #2b2b2b;
    --surface: #383838;
    --text-primary: #ffffff;
    --text-secondary: #b3b3b3;
    --border: #555555;
    --success: #4CAF50;
    --warning: #FF9800;
    --error: #F44336;
    --shadow: rgba(0, 0, 0, 0.3);
}

/* Reset and Base */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--background);
    color: var(--text-primary);
    font-size: 12px;
    line-height: 1.4;
    overflow-x: hidden;
}

.container {
    width: 100%;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Header */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background-color: var(--surface);
    border-bottom: 1px solid var(--border);
    box-shadow: 0 2px 4px var(--shadow);
}

.logo {
    display: flex;
    align-items: center;
    gap: 8px;
}

.logo h1 {
    font-size: 14px;
    font-weight: 600;
    color: var(--primary-color);
}

/* Status Indicator */
.status-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--error);
    transition: background-color 0.3s ease;
}

.status-dot.online {
    background-color: var(--success);
}

.status-dot.offline {
    background-color: var(--error);
}

.status-dot.connecting {
    background-color: var(--warning);
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Main Content */
.main-content {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
}

/* Function Grid */
.function-section {
    margin-bottom: 24px;
}

.function-section h2 {
    margin-bottom: 16px;
    font-size: 16px;
    color: var(--text-primary);
}

.function-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}

.function-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
}

.function-card:hover {
    background-color: #404040;
    border-color: var(--primary-color);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px var(--shadow);
}

.function-card.active {
    background-color: rgba(33, 150, 243, 0.1);
    border-color: var(--primary-color);
}

.function-card h3 {
    margin: 0 0 8px 0;
    font-size: 14px;
    color: var(--primary-color);
}

.function-card p {
    margin: 0 0 16px 0;
    font-size: 11px;
    color: var(--text-secondary);
}
```

### Component Styles (css/components.css)

```css
/* Panels */
.panel {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 4px var(--shadow);
}

.panel h2 {
    margin-bottom: 20px;
    font-size: 16px;
    color: var(--primary-color);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
}

/* Form Elements */
.layer-selection,
.method-selection,
.palette-params {
    margin-bottom: 16px;
}

label {
    display: block;
    margin-bottom: 4px;
    font-size: 11px;
    font-weight: 500;
}

.layer-dropdown,
select {
    width: 100%;
    padding: 8px;
    background-color: var(--background);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 11px;
}

.slider-container {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}

input[type="range"] {
    flex: 1;
    height: 4px;
    background: var(--border);
    outline: none;
    border-radius: 2px;
}

input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    width: 16px;
    height: 16px;
    background: var(--primary-color);
    border-radius: 50%;
    cursor: pointer;
}

.slider-value {
    min-width: 35px;
    font-size: 10px;
    color: var(--text-secondary);
    text-align: right;
}

/* Buttons */
.btn-primary,
.btn-secondary,
.btn-small {
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-primary {
    background-color: var(--primary-color);
    color: white;
    width: 100%;
    margin-bottom: 8px;
}

.btn-primary:hover:not(:disabled) {
    background-color: #1976D2;
}

.btn-primary:disabled {
    background-color: #666;
    cursor: not-allowed;
}

.btn-secondary {
    background-color: var(--secondary-color);
    color: white;
    width: 100%;
}

.btn-secondary:hover:not(:disabled) {
    background-color: #D84315;
}

.btn-small {
    background-color: var(--surface);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 6px 12px;
    font-size: 10px;
    width: auto;
}

/* Palette Display */
.palette-display {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 16px;
    padding: 16px;
    background-color: var(--background);
    border-radius: 4px;
}

.palette-color {
    aspect-ratio: 1;
    border-radius: 4px;
    border: 1px solid var(--border);
    position: relative;
    cursor: pointer;
    transition: transform 0.2s ease;
}

.palette-color:hover {
    transform: scale(1.1);
    border-color: var(--primary-color);
}

.palette-color::after {
    content: attr(data-hex);
    position: absolute;
    bottom: -20px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    color: var(--text-secondary);
    white-space: nowrap;
}

/* Progress Bar */
.progress-container {
    margin: 16px 0;
}

.progress-bar {
    width: 100%;
    height: 6px;
    background-color: var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 8px;
}

.progress-fill {
    height: 100%;
    background-color: var(--primary-color);
    transition: width 0.3s ease;
    width: 0%;
}

.progress-text {
    font-size: 10px;
    color: var(--text-secondary);
    text-align: center;
}

/* Log Container */
.log-container {
    background-color: var(--background);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-top: 16px;
}

.log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
}

.log-header h3 {
    font-size: 12px;
    color: var(--text-primary);
}

.log-content {
    max-height: 150px;
    overflow-y: auto;
    padding: 8px 12px;
    font-family: 'Courier New', monospace;
    font-size: 10px;
}

.log-entry {
    margin-bottom: 4px;
    padding: 2px 0;
}

.log-entry.info {
    color: var(--text-secondary);
}

.log-entry.success {
    color: var(--success);
}

.log-entry.warning {
    color: var(--warning);
}

.log-entry.error {
    color: var(--error);
}

/* Responsive Design */
@media (max-width: 300px) {
    .function-grid {
        grid-template-columns: 1fr;
    }
    
    .palette-display {
        grid-template-columns: repeat(3, 1fr);
    }
}
```

---

## 🎯 INTERACTIVE BEHAVIOR

### Panel Switching Logic

```javascript
// js/ui.js - Panel management
class PanelManager {
    constructor() {
        this.currentPanel = null;
        this.initializePanelSwitching();
    }

    initializePanelSwitching() {
        // Function card click handlers
        document.querySelectorAll('.function-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const functionType = card.dataset.function;
                this.switchToPanel(functionType);
                this.updateActiveCard(card);
            });
        });
    }

    switchToPanel(functionType) {
        // Hide all panels
        document.querySelectorAll('.panel').forEach(panel => {
            panel.style.display = 'none';
        });

        // Show selected panel
        const panelMap = {
            'color-matching': 'colorMatchingPanel',
            'palette-analysis': 'paletteAnalysisPanel',
            'color-harmony': 'colorHarmonyPanel',
            'batch-processing': 'batchProcessingPanel'
        };

        const panelId = panelMap[functionType];
        if (panelId) {
            const panel = document.getElementById(panelId);
            if (panel) {
                panel.style.display = 'block';
                this.currentPanel = functionType;
                this.logActivity(`Switched to ${functionType} panel`);
            }
        }
    }

    updateActiveCard(activeCard) {
        // Remove active class from all cards
        document.querySelectorAll('.function-card').forEach(card => {
            card.classList.remove('active');
        });

        // Add active class to clicked card
        activeCard.classList.add('active');
    }
}
```

### Form Validation & Updates

```javascript
// js/ui.js - Form management
class FormManager {
    constructor() {
        this.initializeFormHandlers();
    }

    initializeFormHandlers() {
        // Slider value updates
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            slider.addEventListener('input', (e) => {
                this.updateSliderValue(e.target);
            });
        });

        // Layer dropdown population
        this.populateLayerDropdowns();
    }

    updateSliderValue(slider) {
        const valueSpan = slider.parentElement.querySelector('.slider-value');
        if (valueSpan) {
            const value = slider.value;
            const suffix = slider.id.includes('strength') || 
                          slider.id.includes('preserveDetails') ? '%' : '';
            valueSpan.textContent = value + suffix;
        }
    }

    async populateLayerDropdowns() {
        try {
            // Get layers from Photoshop via ExtendScript
            const layers = await window.photoshopAPI.getLayers();
            
            const dropdowns = [
                'sourceLayer',
                'targetLayer', 
                'analysisLayer'
            ];

            dropdowns.forEach(dropdownId => {
                const dropdown = document.getElementById(dropdownId);
                if (dropdown) {
                    // Clear existing options (except first)
                    while (dropdown.children.length > 1) {
                        dropdown.removeChild(dropdown.lastChild);
                    }

                    // Add layer options
                    layers.forEach(layer => {
                        const option = document.createElement('option');
                        option.value = layer.id;
                        option.textContent = layer.name;
                        dropdown.appendChild(option);
                    });
                }
            });

        } catch (error) {
            this.logActivity('Failed to populate layer dropdowns: ' + error.message, 'error');
        }
    }
}
```

---

## 📊 STATUS & PROGRESS MANAGEMENT

### Status Indicator Updates

```javascript
// js/ui.js - Status management
class StatusManager {
    constructor() {
        this.statusDot = document.querySelector('.status-dot');
        this.statusText = document.querySelector('.status-text');
        this.checkServerStatus();
        
        // Check status every 30 seconds
        setInterval(() => this.checkServerStatus(), 30000);
    }

    async checkServerStatus() {
        try {
            this.updateStatus('connecting', 'Checking...');
            
            const response = await fetch('http://127.0.0.1:5000/api/status');
            if (response.ok) {
                this.updateStatus('online', 'Server Online');
            } else {
                this.updateStatus('offline', 'Server Error');
            }
        } catch (error) {
            this.updateStatus('offline', 'Server Offline');
        }
    }

    updateStatus(status, text) {
        this.statusDot.className = `status-dot ${status}`;
        this.statusText.textContent = text;
    }
}
```

### Progress Bar Management

```javascript
// js/ui.js - Progress management
class ProgressManager {
    constructor() {
        this.container = document.getElementById('progressContainer');
        this.fill = document.getElementById('progressFill');
        this.text = document.getElementById('progressText');
    }

    show(message = 'Processing...') {
        this.container.style.display = 'block';
        this.text.textContent = message;
        this.setProgress(0);
    }

    hide() {
        this.container.style.display = 'none';
    }

    setProgress(percentage) {
        this.fill.style.width = Math.max(0, Math.min(100, percentage)) + '%';
    }

    updateText(message) {
        this.text.textContent = message;
    }
}
```

---

## 📝 ACTIVITY LOGGING

```javascript
// js/ui.js - Logging system
class ActivityLogger {
    constructor() {
        this.logContent = document.getElementById('logContent');
        this.clearButton = document.getElementById('clearLog');
        
        this.clearButton.addEventListener('click', () => this.clearLog());
    }

    log(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${timestamp}] ${message}`;
        
        this.logContent.appendChild(entry);
        this.logContent.scrollTop = this.logContent.scrollHeight;
        
        // Keep only last 100 entries
        while (this.logContent.children.length > 100) {
            this.logContent.removeChild(this.logContent.firstChild);
        }
    }

    clearLog() {
        this.logContent.innerHTML = '';
        this.log('Log cleared');
    }
}

// Global logging function
window.logActivity = function(message, type = 'info') {
    if (window.activityLogger) {
        window.activityLogger.log(message, type);
    }
};
```

---

## 🔗 NAVIGATION

**📖 Rozdziały:**
- **[⬅️ Chapter 2 - CEP Extension Setup](./gatto-WORKING-04-photoshop-chapter2.md)**
- **[➡️ Chapter 4 - ExtendScript Core Functions](./gatto-WORKING-04-photoshop-chapter4.md)**
- **[📋 Spis treści](./gatto-WORKING-04-photoshop-toc.md)**

---

*Ten rozdział opisuje kompletny interfejs użytkownika dla panelu CEP, włączając HTML, CSS, i podstawową logikę JavaScript dla zarządzania interfejsem.*
