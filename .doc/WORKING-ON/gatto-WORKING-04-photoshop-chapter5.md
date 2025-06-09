# GattoNero AI Assistant - Photoshop Integration
## Chapter 5: JavaScript Application Logic

> **Status:** 📝 AKTYWNY  
> **Poprzedni:** [Chapter 4 - ExtendScript Core Functions](./gatto-WORKING-04-photoshop-chapter4.md)  
> **Następny:** [Chapter 6 - Integration Testing](./gatto-WORKING-04-photoshop-chapter6.md)  
> **Spis treści:** [gatto-WORKING-04-photoshop-toc.md](./gatto-WORKING-04-photoshop-toc.md)

---

## 🔧 JAVASCRIPT APPLICATION LOGIC

### Main Application Controller (js/main.js)

```javascript
// js/main.js - Main application logic

(function() {
    'use strict';
    
    // Global application state
    window.GattoNeroApp = {
        initialized: false,
        serverOnline: false,
        currentFunction: null,
        processing: false
    };
    
    // Initialize application when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        initializeApplication();
    });
    
    function initializeApplication() {
        console.log('[GattoNero] Initializing application...');
        
        try {
            // Initialize managers
            window.panelManager = new PanelManager();
            window.statusManager = new StatusManager();
            window.progressManager = new ProgressManager();
            window.activityLogger = new ActivityLogger();
            window.formManager = new FormManager();
            
            // Initialize Photoshop API bridge
            initializePhotoshopAPI();
            
            // Set up event handlers
            setupEventHandlers();
            
            // Initial UI update
            updateUI();
            
            window.GattoNeroApp.initialized = true;
            window.logActivity('Application initialized successfully', 'success');
            
        } catch (error) {
            console.error('[GattoNero] Initialization error:', error);
            window.logActivity('Application initialization failed: ' + error.message, 'error');
        }
    }
    
    function initializePhotoshopAPI() {
        /**
         * Initialize bridge to ExtendScript
         */
        try {
            // CSInterface for CEP communication
            window.csInterface = new CSInterface();
            
            // Test ExtendScript connection
            window.csInterface.evalScript('GattoNeroAPI.initialize()', function(result) {
                try {
                    var response = JSON.parse(result);
                    if (response.success) {
                        window.logActivity('ExtendScript API connected', 'success');
                    } else {
                        window.logActivity('ExtendScript API error: ' + response.error, 'error');
                    }
                } catch (e) {
                    window.logActivity('ExtendScript API connection failed', 'error');
                }
            });
            
        } catch (error) {
            console.error('[GattoNero] PhotoshopAPI initialization error:', error);
            window.logActivity('Failed to initialize Photoshop API', 'error');
        }
    }
    
    function setupEventHandlers() {
        /**
         * Set up all UI event handlers
         */
        
        // Color Matching
        const colorMatchingBtn = document.getElementById('processColorMatching');
        if (colorMatchingBtn) {
            colorMatchingBtn.addEventListener('click', handleColorMatching);
        }
        
        const previewBtn = document.getElementById('previewColorMatching');
        if (previewBtn) {
            previewBtn.addEventListener('click', handleColorMatchingPreview);
        }
        
        // Palette Analysis
        const paletteAnalysisBtn = document.getElementById('analyzePalette');
        if (paletteAnalysisBtn) {
            paletteAnalysisBtn.addEventListener('click', handlePaletteAnalysis);
        }
        
        const savePaletteBtn = document.getElementById('savePalette');
        if (savePaletteBtn) {
            savePaletteBtn.addEventListener('click', handleSavePalette);
        }
        
        // Layer dropdowns refresh on focus
        document.querySelectorAll('.layer-dropdown').forEach(dropdown => {
            dropdown.addEventListener('focus', function() {
                window.formManager.populateLayerDropdowns();
            });
        });
    }
    
    function updateUI() {
        /**
         * Update UI state based on application status
         */
        const buttons = document.querySelectorAll('.btn-primary, .btn-secondary');
        buttons.forEach(btn => {
            btn.disabled = window.GattoNeroApp.processing;
        });
    }
    
    // Export main functions
    window.GattoNeroApp.updateUI = updateUI;
    
})();
```

---

## 🎨 COLOR MATCHING LOGIC

### Color Matching Handler (js/color-matching.js)

```javascript
// js/color-matching.js - Color matching functionality

function handleColorMatching() {
    /**
     * Handle color matching process
     */
    if (window.GattoNeroApp.processing) {
        window.logActivity('Processing already in progress', 'warning');
        return;
    }
    
    try {
        // Get form values
        const sourceLayerId = document.getElementById('sourceLayer').value;
        const targetLayerId = document.getElementById('targetLayer').value;
        const method = document.getElementById('matchingMethod').value;
        const strength = document.getElementById('strength').value;
        const preserveDetails = document.getElementById('preserveDetails').value;
        
        // Validate inputs
        if (!sourceLayerId || !targetLayerId) {
            window.logActivity('Please select both source and target layers', 'error');
            return;
        }
        
        if (sourceLayerId === targetLayerId) {
            window.logActivity('Source and target layers must be different', 'error');
            return;
        }
        
        // Start processing
        window.GattoNeroApp.processing = true;
        window.GattoNeroApp.updateUI();
        
        window.progressManager.show('Preparing color matching...');
        window.logActivity('Starting color matching process', 'info');
        
        // Prepare parameters
        const params = {
            sourceLayer: parseInt(sourceLayerId),
            targetLayer: parseInt(targetLayerId),
            method: method,
            strength: parseInt(strength),
            preserveDetails: parseInt(preserveDetails)
        };
        
        // Call ExtendScript function
        const scriptCall = `processFromCEP('processColorMatching', ${JSON.stringify(params)})`;
        
        window.csInterface.evalScript(scriptCall, function(result) {
            handleColorMatchingResult(result);
        });
        
    } catch (error) {
        console.error('[ColorMatching] Error:', error);
        window.logActivity('Color matching error: ' + error.message, 'error');
        finishProcessing();
    }
}

function handleColorMatchingResult(result) {
    /**
     * Handle color matching result from ExtendScript
     * @param {String} result - JSON result from ExtendScript
     */
    try {
        const response = JSON.parse(result);
        
        if (response.success) {
            window.progressManager.setProgress(100);
            window.progressManager.updateText('Color matching completed!');
            
            window.logActivity(
                `Color matching successful: ${response.layerName}`, 
                'success'
            );
            
            // Update layer dropdowns to include new layer
            setTimeout(() => {
                window.formManager.populateLayerDropdowns();
                finishProcessing();
            }, 1000);
            
        } else {
            window.logActivity(
                'Color matching failed: ' + (response.error || 'Unknown error'), 
                'error'
            );
            finishProcessing();
        }
        
    } catch (error) {
        console.error('[ColorMatching] Result parsing error:', error);
        window.logActivity('Failed to parse color matching result', 'error');
        finishProcessing();
    }
}

function handleColorMatchingPreview() {
    /**
     * Handle color matching preview (non-destructive)
     */
    window.logActivity('Preview functionality not yet implemented', 'info');
    // TODO: Implement preview functionality
}
```

---

## 🎭 PALETTE ANALYSIS LOGIC

### Palette Analysis Handler (js/palette-analysis.js)

```javascript
// js/palette-analysis.js - Palette analysis functionality

function handlePaletteAnalysis() {
    /**
     * Handle palette analysis process
     */
    if (window.GattoNeroApp.processing) {
        window.logActivity('Processing already in progress', 'warning');
        return;
    }
    
    try {
        // Get form values
        const layerId = document.getElementById('analysisLayer').value;
        const colorCount = document.getElementById('colorCount').value;
        const tolerance = document.getElementById('tolerance').value;
        
        // Validate inputs
        if (!layerId) {
            window.logActivity('Please select a layer to analyze', 'error');
            return;
        }
        
        // Start processing
        window.GattoNeroApp.processing = true;
        window.GattoNeroApp.updateUI();
        
        window.progressManager.show('Analyzing color palette...');
        window.logActivity('Starting palette analysis', 'info');
        
        // Prepare parameters
        const params = {
            layerId: parseInt(layerId),
            colorCount: parseInt(colorCount),
            tolerance: parseInt(tolerance)
        };
        
        // Call ExtendScript function
        const scriptCall = `processFromCEP('analyzePalette', ${JSON.stringify(params)})`;
        
        window.csInterface.evalScript(scriptCall, function(result) {
            handlePaletteAnalysisResult(result);
        });
        
    } catch (error) {
        console.error('[PaletteAnalysis] Error:', error);
        window.logActivity('Palette analysis error: ' + error.message, 'error');
        finishProcessing();
    }
}

function handlePaletteAnalysisResult(result) {
    /**
     * Handle palette analysis result from ExtendScript
     * @param {String} result - JSON result from ExtendScript
     */
    try {
        const response = JSON.parse(result);
        
        if (response.success) {
            window.progressManager.setProgress(100);
            window.progressManager.updateText('Palette analysis completed!');
            
            window.logActivity('Palette analysis successful', 'success');
            
            // Display palette results
            displayPaletteResults(response.palette, response.statistics);
            
            // Store current palette for saving
            window.currentPalette = response.palette;
            
            setTimeout(() => {
                finishProcessing();
            }, 1000);
            
        } else {
            window.logActivity(
                'Palette analysis failed: ' + (response.error || 'Unknown error'), 
                'error'
            );
            finishProcessing();
        }
        
    } catch (error) {
        console.error('[PaletteAnalysis] Result parsing error:', error);
        window.logActivity('Failed to parse palette analysis result', 'error');
        finishProcessing();
    }
}

function displayPaletteResults(palette, statistics) {
    /**
     * Display palette analysis results in UI
     * @param {Array} palette - Array of color objects
     * @param {Object} statistics - Palette statistics
     */
    const paletteDisplay = document.getElementById('paletteResults');
    if (!paletteDisplay) return;
    
    // Clear previous results
    paletteDisplay.innerHTML = '';
    
    // Create color swatches
    palette.forEach((color, index) => {
        const colorDiv = document.createElement('div');
        colorDiv.className = 'palette-color';
        colorDiv.style.backgroundColor = color.hex;
        colorDiv.setAttribute('data-hex', color.hex);
        colorDiv.setAttribute('data-percentage', color.percentage.toFixed(1) + '%');
        colorDiv.title = `${color.hex} (${color.percentage.toFixed(1)}%)`;
        
        // Add click handler for copying color value
        colorDiv.addEventListener('click', function() {
            copyColorToClipboard(color.hex);
            window.logActivity(`Copied color: ${color.hex}`, 'info');
        });
        
        paletteDisplay.appendChild(colorDiv);
    });
    
    // Show the palette display
    paletteDisplay.style.display = 'grid';
    
    // Log statistics
    if (statistics) {
        window.logActivity(
            `Palette contains ${palette.length} colors, ` +
            `dominant: ${statistics.dominant_color || 'unknown'}`, 
            'info'
        );
    }
}

function copyColorToClipboard(hexColor) {
    /**
     * Copy color value to clipboard
     * @param {String} hexColor - Hex color value
     */
    try {
        // Create temporary input element
        const tempInput = document.createElement('input');
        tempInput.value = hexColor;
        document.body.appendChild(tempInput);
        tempInput.select();
        document.execCommand('copy');
        document.body.removeChild(tempInput);
        
    } catch (error) {
        console.error('[PaletteAnalysis] Clipboard error:', error);
    }
}

function handleSavePalette() {
    /**
     * Handle saving current palette
     */
    if (!window.currentPalette) {
        window.logActivity('No palette to save', 'warning');
        return;
    }
    
    try {
        // Create palette data
        const paletteData = {
            name: 'GattoNero_Palette_' + Date.now(),
            colors: window.currentPalette,
            created: new Date().toISOString()
        };
        
        // Convert to JSON string
        const jsonData = JSON.stringify(paletteData, null, 2);
        
        // Create downloadable file
        const blob = new Blob([jsonData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const a = document.createElement('a');
        a.href = url;
        a.download = paletteData.name + '.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
        
        window.logActivity('Palette saved successfully', 'success');
        
    } catch (error) {
        console.error('[PaletteAnalysis] Save error:', error);
        window.logActivity('Failed to save palette: ' + error.message, 'error');
    }
}
```

---

## 🔄 LAYER MANAGEMENT

### Layer Operations (js/layer-management.js)

```javascript
// js/layer-management.js - Layer management functionality

class LayerManager {
    constructor() {
        this.layers = [];
        this.selectedLayers = new Set();
        this.refreshInterval = null;
    }
    
    async refreshLayers() {
        /**
         * Refresh layer list from Photoshop
         */
        try {
            return new Promise((resolve, reject) => {
                const scriptCall = "processFromCEP('getLayers', {})";
                
                window.csInterface.evalScript(scriptCall, (result) => {
                    try {
                        const response = JSON.parse(result);
                        
                        if (response.success) {
                            this.layers = response.data;
                            this.updateLayerUI();
                            resolve(this.layers);
                        } else {
                            reject(new Error(response.error || 'Failed to get layers'));
                        }
                    } catch (error) {
                        reject(error);
                    }
                });
            });
            
        } catch (error) {
            console.error('[LayerManager] Refresh error:', error);
            throw error;
        }
    }
    
    updateLayerUI() {
        /**
         * Update layer dropdowns in UI
         */
        const dropdowns = document.querySelectorAll('.layer-dropdown');
        
        dropdowns.forEach(dropdown => {
            // Save current selection
            const currentValue = dropdown.value;
            
            // Clear options (except first placeholder)
            while (dropdown.children.length > 1) {
                dropdown.removeChild(dropdown.lastChild);
            }
            
            // Add layer options
            this.layers.forEach(layer => {
                const option = document.createElement('option');
                option.value = layer.id;
                option.textContent = this.formatLayerName(layer);
                dropdown.appendChild(option);
            });
            
            // Restore selection if still valid
            if (currentValue && this.layerExists(currentValue)) {
                dropdown.value = currentValue;
            }
        });
    }
    
    formatLayerName(layer) {
        /**
         * Format layer name for display
         * @param {Object} layer - Layer object
         * @returns {String} Formatted name
         */
        let name = layer.name;
        
        // Add path if it's a nested layer
        if (layer.path && layer.path !== layer.name) {
            name = layer.path;
        }
        
        // Add layer type indicator
        const typeMap = {
            'LayerKind.NORMAL': '📷',
            'LayerKind.TEXT': '📝',
            'LayerKind.SOLIDFILL': '🎨',
            'LayerKind.GRADIENTFILL': '🌈',
            'LayerKind.PATTERNFILL': '🔲'
        };
        
        const icon = typeMap[layer.kind] || '📄';
        
        return `${icon} ${name}`;
    }
    
    layerExists(layerId) {
        /**
         * Check if layer exists in current list
         * @param {String} layerId - Layer ID to check
         * @returns {Boolean} Layer exists
         */
        return this.layers.some(layer => layer.id.toString() === layerId.toString());
    }
    
    getLayerById(layerId) {
        /**
         * Get layer object by ID
         * @param {String|Number} layerId - Layer ID
         * @returns {Object|null} Layer object
         */
        return this.layers.find(layer => layer.id.toString() === layerId.toString()) || null;
    }
    
    selectLayer(layerId) {
        /**
         * Select layer in Photoshop
         * @param {String|Number} layerId - Layer ID
         * @returns {Promise} Selection result
         */
        return new Promise((resolve, reject) => {
            const params = { layerId: parseInt(layerId) };
            const scriptCall = `processFromCEP('selectLayer', ${JSON.stringify(params)})`;
            
            window.csInterface.evalScript(scriptCall, (result) => {
                try {
                    const response = JSON.parse(result);
                    
                    if (response.success) {
                        window.logActivity(`Selected layer: ${layerId}`, 'info');
                        resolve(response);
                    } else {
                        reject(new Error(response.error || 'Failed to select layer'));
                    }
                } catch (error) {
                    reject(error);
                }
            });
        });
    }
    
    startAutoRefresh(intervalMs = 5000) {
        /**
         * Start automatic layer refresh
         * @param {Number} intervalMs - Refresh interval in milliseconds
         */
        this.stopAutoRefresh();
        
        this.refreshInterval = setInterval(() => {
            this.refreshLayers().catch(error => {
                console.error('[LayerManager] Auto-refresh error:', error);
            });
        }, intervalMs);
        
        // Initial refresh
        this.refreshLayers();
    }
    
    stopAutoRefresh() {
        /**
         * Stop automatic layer refresh
         */
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
}

// Initialize global layer manager
window.layerManager = new LayerManager();
```

---

## 🔄 EVENT HANDLING

### Event Management (js/event-handler.js)

```javascript
// js/event-handler.js - Global event handling

class EventHandler {
    constructor() {
        this.setupGlobalEvents();
        this.setupKeyboardShortcuts();
    }
    
    setupGlobalEvents() {
        /**
         * Set up application-wide event handlers
         */
        
        // Window focus/blur events
        window.addEventListener('focus', () => {
            this.onWindowFocus();
        });
        
        window.addEventListener('blur', () => {
            this.onWindowBlur();
        });
        
        // Before unload cleanup
        window.addEventListener('beforeunload', () => {
            this.onBeforeUnload();
        });
        
        // Error handling
        window.addEventListener('error', (event) => {
            this.onGlobalError(event);
        });
        
        window.addEventListener('unhandledrejection', (event) => {
            this.onUnhandledRejection(event);
        });
    }
    
    setupKeyboardShortcuts() {
        /**
         * Set up keyboard shortcuts
         */
        document.addEventListener('keydown', (event) => {
            // Ctrl/Cmd + R: Refresh layers
            if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
                event.preventDefault();
                this.handleRefreshLayers();
            }
            
            // Ctrl/Cmd + L: Clear log
            if ((event.ctrlKey || event.metaKey) && event.key === 'l') {
                event.preventDefault();
                this.handleClearLog();
            }
            
            // Escape: Cancel current operation
            if (event.key === 'Escape') {
                this.handleCancel();
            }
        });
    }
    
    onWindowFocus() {
        /**
         * Handle window focus event
         */
        // Refresh layer list when window gains focus
        if (window.layerManager) {
            window.layerManager.refreshLayers().catch(error => {
                console.error('[EventHandler] Focus refresh error:', error);
            });
        }
        
        // Check server status
        if (window.statusManager) {
            window.statusManager.checkServerStatus();
        }
    }
    
    onWindowBlur() {
        /**
         * Handle window blur event
         */
        // Could pause auto-refresh or other operations
    }
    
    onBeforeUnload() {
        /**
         * Handle before unload event
         */
        // Cleanup operations
        if (window.layerManager) {
            window.layerManager.stopAutoRefresh();
        }
    }
    
    onGlobalError(event) {
        /**
         * Handle global JavaScript errors
         * @param {ErrorEvent} event - Error event
         */
        console.error('[EventHandler] Global error:', event.error);
        
        if (window.logActivity) {
            window.logActivity(
                'Application error: ' + event.error.message, 
                'error'
            );
        }
    }
    
    onUnhandledRejection(event) {
        /**
         * Handle unhandled promise rejections
         * @param {PromiseRejectionEvent} event - Rejection event
         */
        console.error('[EventHandler] Unhandled rejection:', event.reason);
        
        if (window.logActivity) {
            window.logActivity(
                'Promise rejection: ' + event.reason, 
                'error'
            );
        }
    }
    
    handleRefreshLayers() {
        /**
         * Handle refresh layers shortcut
         */
        if (window.layerManager) {
            window.layerManager.refreshLayers()
                .then(() => {
                    window.logActivity('Layers refreshed', 'info');
                })
                .catch(error => {
                    window.logActivity('Failed to refresh layers: ' + error.message, 'error');
                });
        }
    }
    
    handleClearLog() {
        /**
         * Handle clear log shortcut
         */
        if (window.activityLogger) {
            window.activityLogger.clearLog();
        }
    }
    
    handleCancel() {
        /**
         * Handle cancel operation shortcut
         */
        if (window.GattoNeroApp && window.GattoNeroApp.processing) {
            // TODO: Implement operation cancellation
            window.logActivity('Operation cancellation not implemented', 'warning');
        }
    }
}

// Initialize global event handler
window.eventHandler = new EventHandler();
```

---

## 🔧 UTILITY FUNCTIONS

### Processing Helpers (js/processing-utils.js)

```javascript
// js/processing-utils.js - Processing utility functions

function finishProcessing() {
    /**
     * Clean up after processing operation
     */
    window.GattoNeroApp.processing = false;
    window.GattoNeroApp.updateUI();
    window.progressManager.hide();
}

function validateProcessingInputs(requiredFields) {
    /**
     * Validate required form inputs
     * @param {Array} requiredFields - Array of required field IDs
     * @returns {Object} Validation result
     */
    const missing = [];
    const values = {};
    
    requiredFields.forEach(fieldId => {
        const element = document.getElementById(fieldId);
        if (!element || !element.value.trim()) {
            missing.push(fieldId);
        } else {
            values[fieldId] = element.value.trim();
        }
    });
    
    return {
        valid: missing.length === 0,
        missing: missing,
        values: values
    };
}

function formatProcessingTime(startTime, endTime) {
    /**
     * Format processing time duration
     * @param {Date} startTime - Start time
     * @param {Date} endTime - End time
     * @returns {String} Formatted duration
     */
    const duration = endTime - startTime;
    const seconds = Math.floor(duration / 1000);
    const minutes = Math.floor(seconds / 60);
    
    if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

function debounce(func, wait) {
    /**
     * Debounce function calls
     * @param {Function} func - Function to debounce
     * @param {Number} wait - Wait time in milliseconds
     * @returns {Function} Debounced function
     */
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export utility functions
window.ProcessingUtils = {
    finishProcessing,
    validateProcessingInputs,
    formatProcessingTime,
    debounce
};
```

---

## 🔗 NAVIGATION

**📖 Rozdziały:**
- **[⬅️ Chapter 4 - ExtendScript Core Functions](./gatto-WORKING-04-photoshop-chapter4.md)**
- **[➡️ Chapter 6 - Integration Testing](./gatto-WORKING-04-photoshop-chapter6.md)**
- **[📋 Spis treści](./gatto-WORKING-04-photoshop-toc.md)**

---

*Ten rozdział opisuje logikę aplikacji JavaScript odpowiedzialną za zarządzanie interfejsem użytkownika, obsługę zdarzeń oraz koordynację komunikacji z ExtendScript.*
