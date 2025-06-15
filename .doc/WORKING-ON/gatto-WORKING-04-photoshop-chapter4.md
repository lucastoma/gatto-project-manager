# GattoNero AI Assistant - Photoshop Integration
## Chapter 4: ExtendScript Core Functions

> **Status:** 📝 AKTYWNY  
> **Poprzedni:** [Chapter 3 - CEP Panel Interface](./gatto-WORKING-04-photoshop-chapter3.md)  
> **Następny:** [Chapter 5 - JavaScript Application Logic](./gatto-WORKING-04-photoshop-chapter5.md)  
> **Spis treści:** [gatto-WORKING-04-photoshop-toc.md](./gatto-WORKING-04-photoshop-toc.md)

---

## 🔧 EXTENDSCRIPT CORE FUNCTIONS

> **✅ POPRAWKA:** ExtendScript pliki (.jsx) znajdują się w folderze `host/`, nie w głównym katalogu

### Main ExtendScript File (host/main.jsx)

```javascript
// host/main.jsx - Main ExtendScript entry point

#target photoshop

// Include external scripts
#include "layer-operations.jsx"
#include "file-operations.jsx"
#include "api-communication.jsx"
#include "utils.jsx"

/**
 * Main GattoNero ExtendScript API
 * Provides bridge between CEP panel and Photoshop
 */
var GattoNeroAPI = (function() {
    'use strict';
    
    var api = {};
    
    // Initialize API
    api.initialize = function() {
        try {
            $.writeln('[GattoNero] ExtendScript API initialized');
            return {
                success: true,
                message: 'API initialized successfully',
                version: '1.0.0'
            };
        } catch (error) {
            $.writeln('[GattoNero] Initialization error: ' + error.message);
            return {
                success: false,
                error: error.message
            };
        }
    };
    
    // Export main functions
    api.getLayers = LayerOperations.getLayers;
    api.selectLayer = LayerOperations.selectLayer;
    api.exportLayer = FileOperations.exportLayerToTIFF;
    api.importLayer = FileOperations.importTIFFAsLayer;
    api.callAPI = APIComm.makeAPICall;
    api.processColorMatching = processColorMatching;
    api.analyzePalette = analyzePalette;
    
    return api;
})();

// Main processing functions
function processColorMatching(params) {
    /**
     * Process color matching between layers
     * @param {Object} params - Processing parameters
     * @returns {Object} Result object
     */
    try {
        $.writeln('[GattoNero] Starting color matching process');
        
        // Validate parameters
        if (!params.sourceLayer || !params.targetLayer) {
            throw new Error('Source and target layers are required');
        }
        
        // Get document
        var doc = app.activeDocument;
        if (!doc) {
            throw new Error('No active document');
        }
        
        // Select source layer
        var sourceLayer = LayerOperations.getLayerById(params.sourceLayer);
        var targetLayer = LayerOperations.getLayerById(params.targetLayer);
        
        if (!sourceLayer || !targetLayer) {
            throw new Error('Could not find specified layers');
        }
        
        // Export layers for processing
        var tempDir = Folder.temp + '/gattonero_temp/';
        var tempFolder = new Folder(tempDir);
        if (!tempFolder.exists) {
            tempFolder.create();
        }
        
        var sourcePath = tempDir + 'source_' + Date.now() + '.tiff';
        var targetPath = tempDir + 'target_' + Date.now() + '.tiff';
        
        $.writeln('[GattoNero] Exporting source layer to: ' + sourcePath);
        FileOperations.exportLayerToTIFF(sourceLayer, sourcePath);
        
        $.writeln('[GattoNero] Exporting target layer to: ' + targetPath);
        FileOperations.exportLayerToTIFF(targetLayer, targetPath);
        
        // Prepare API request
        var apiParams = {
            method: params.method || 'histogram',
            strength: params.strength || 80,
            preserve_details: params.preserveDetails || 60
        };
        
        // Make API call
        $.writeln('[GattoNero] Calling color matching API');
        var result = APIComm.makeAPICall('color_matching', {
            source_image: sourcePath,
            target_image: targetPath,
            params: apiParams
        });
        
        if (result.success && result.output_path) {
            // Import processed image as new layer
            $.writeln('[GattoNero] Importing processed result');
            var processedLayer = FileOperations.importTIFFAsLayer(
                result.output_path, 
                targetLayer.name + '_matched'
            );
            
            // Cleanup temp files
            FileOperations.cleanupTempFiles([sourcePath, targetPath, result.output_path]);
            
            return {
                success: true,
                message: 'Color matching completed successfully',
                layerId: processedLayer.id,
                layerName: processedLayer.name
            };
        } else {
            throw new Error(result.error || 'API call failed');
        }
        
    } catch (error) {
        $.writeln('[GattoNero] Color matching error: ' + error.message);
        return {
            success: false,
            error: error.message
        };
    }
}

function analyzePalette(params) {
    /**
     * Analyze color palette from layer
     * @param {Object} params - Analysis parameters
     * @returns {Object} Result object with palette data
     */
    try {
        $.writeln('[GattoNero] Starting palette analysis');
        
        // Validate parameters
        if (!params.layerId) {
            throw new Error('Layer ID is required');
        }
        
        // Get layer
        var layer = LayerOperations.getLayerById(params.layerId);
        if (!layer) {
            throw new Error('Could not find specified layer');
        }
        
        // Export layer for analysis
        var tempDir = Folder.temp + '/gattonero_temp/';
        var tempFolder = new Folder(tempDir);
        if (!tempFolder.exists) {
            tempFolder.create();
        }
        
        var imagePath = tempDir + 'analysis_' + Date.now() + '.tiff';
        $.writeln('[GattoNero] Exporting layer for analysis: ' + imagePath);
        FileOperations.exportLayerToTIFF(layer, imagePath);
        
        // Prepare API request
        var apiParams = {
            color_count: params.colorCount || 8,
            tolerance: params.tolerance || 10
        };
        
        // Make API call
        $.writeln('[GattoNero] Calling palette analysis API');
        var result = APIComm.makeAPICall('analyze_palette', {
            image_path: imagePath,
            params: apiParams
        });
        
        // Cleanup temp file
        FileOperations.cleanupTempFiles([imagePath]);
        
        if (result.success) {
            return {
                success: true,
                message: 'Palette analysis completed',
                palette: result.palette,
                statistics: result.statistics
            };
        } else {
            throw new Error(result.error || 'Palette analysis failed');
        }
        
    } catch (error) {
        $.writeln('[GattoNero] Palette analysis error: ' + error.message);
        return {
            success: false,
            error: error.message
        };
    }
}
```

---

## 📁 LAYER OPERATIONS

### Layer Management (host/layer-operations.jsx)

```javascript
// host/layer-operations.jsx - Layer management functions

var LayerOperations = (function() {
    'use strict';
    
    var ops = {};
    
    ops.getLayers = function() {
        /**
         * Get all layers from active document
         * @returns {Array} Array of layer objects
         */
        try {
            var doc = app.activeDocument;
            if (!doc) {
                throw new Error('No active document');
            }
            
            var layers = [];
            
            function processLayer(layer, parentPath) {
                var layerPath = parentPath ? parentPath + '/' + layer.name : layer.name;
                
                layers.push({
                    id: layer.id,
                    name: layer.name,
                    path: layerPath,
                    visible: layer.visible,
                    opacity: layer.opacity,
                    blendMode: layer.blendMode.toString(),
                    kind: layer.kind.toString(),
                    bounds: {
                        left: layer.bounds[0].as('px'),
                        top: layer.bounds[1].as('px'),
                        right: layer.bounds[2].as('px'),
                        bottom: layer.bounds[3].as('px')
                    }
                });
                
                // Process sublayers if it's a layer set
                if (layer.typename === 'LayerSet') {
                    for (var i = 0; i < layer.layers.length; i++) {
                        processLayer(layer.layers[i], layerPath);
                    }
                }
            }
            
            // Process all layers
            for (var i = 0; i < doc.layers.length; i++) {
                processLayer(doc.layers[i], '');
            }
            
            $.writeln('[LayerOps] Found ' + layers.length + ' layers');
            return layers;
            
        } catch (error) {
            $.writeln('[LayerOps] Error getting layers: ' + error.message);
            throw error;
        }
    };
    
    ops.getLayerById = function(layerId) {
        /**
         * Get layer by ID
         * @param {Number} layerId - Layer ID
         * @returns {Layer} Layer object
         */
        try {
            var doc = app.activeDocument;
            if (!doc) {
                throw new Error('No active document');
            }
            
            function findLayerById(layer, targetId) {
                if (layer.id === targetId) {
                    return layer;
                }
                
                // Search in sublayers if it's a layer set
                if (layer.typename === 'LayerSet') {
                    for (var i = 0; i < layer.layers.length; i++) {
                        var found = findLayerById(layer.layers[i], targetId);
                        if (found) return found;
                    }
                }
                
                return null;
            }
            
            // Search through all layers
            for (var i = 0; i < doc.layers.length; i++) {
                var found = findLayerById(doc.layers[i], layerId);
                if (found) return found;
            }
            
            return null;
            
        } catch (error) {
            $.writeln('[LayerOps] Error finding layer ' + layerId + ': ' + error.message);
            throw error;
        }
    };
    
    ops.selectLayer = function(layerId) {
        /**
         * Select layer by ID
         * @param {Number} layerId - Layer ID
         * @returns {Boolean} Success status
         */
        try {
            var layer = ops.getLayerById(layerId);
            if (!layer) {
                throw new Error('Layer not found: ' + layerId);
            }
            
            app.activeDocument.activeLayer = layer;
            $.writeln('[LayerOps] Selected layer: ' + layer.name);
            return true;
            
        } catch (error) {
            $.writeln('[LayerOps] Error selecting layer: ' + error.message);
            throw error;
        }
    };
    
    ops.duplicateLayer = function(layer, newName) {
        /**
         * Duplicate a layer
         * @param {Layer} layer - Source layer
         * @param {String} newName - Name for duplicated layer
         * @returns {Layer} Duplicated layer
         */
        try {
            var duplicate = layer.duplicate();
            if (newName) {
                duplicate.name = newName;
            }
            
            $.writeln('[LayerOps] Duplicated layer: ' + duplicate.name);
            return duplicate;
            
        } catch (error) {
            $.writeln('[LayerOps] Error duplicating layer: ' + error.message);
            throw error;
        }
    };
    
    ops.deleteLayer = function(layer) {
        /**
         * Delete a layer
         * @param {Layer} layer - Layer to delete
         * @returns {Boolean} Success status
         */
        try {
            layer.remove();
            $.writeln('[LayerOps] Deleted layer: ' + layer.name);
            return true;
            
        } catch (error) {
            $.writeln('[LayerOps] Error deleting layer: ' + error.message);
            throw error;
        }
    };
    
    return ops;
})();
```

---

## 💾 FILE OPERATIONS

### File Import/Export (host/file-operations.jsx)

```javascript
// host/file-operations.jsx - File handling functions

var FileOperations = (function() {
    'use strict';
    
    var ops = {};
    
    ops.exportLayerToTIFF = function(layer, outputPath) {
        /**
         * Export layer to TIFF file
         * @param {Layer} layer - Layer to export
         * @param {String} outputPath - Output file path
         * @returns {String} Output path
         */
        try {
            var doc = app.activeDocument;
            var originalActiveLayer = doc.activeLayer;
            
            // Select target layer
            doc.activeLayer = layer;
            
            // Create temporary document with just this layer
            var layerBounds = layer.bounds;
            var width = layerBounds[2] - layerBounds[0];
            var height = layerBounds[3] - layerBounds[1];
            
            var tempDoc = app.documents.add(
                width > 0 ? width : doc.width, 
                height > 0 ? height : doc.height, 
                doc.resolution, 
                "temp_export_" + Date.now(), 
                NewDocumentMode.RGB
            );
            
            // Copy layer to temp document
            layer.copy();
            app.activeDocument = tempDoc;
            tempDoc.paste();
            
            // Flatten if needed
            if (tempDoc.layers.length > 1) {
                tempDoc.flatten();
            }
            
            // TIFF save options
            var tiffOptions = new TiffSaveOptions();
            tiffOptions.compression = TIFFCompression.LZW;  // Use LZW compression
            tiffOptions.imageCompression = TIFFCompression.LZW;
            tiffOptions.alphaChannels = false;
            tiffOptions.layers = false;
            tiffOptions.spotColors = false;
            tiffOptions.annotations = false;
            tiffOptions.byteOrder = ByteOrder.IBM;
            
            // Save file
            var file = new File(outputPath);
            tempDoc.saveAs(file, tiffOptions, true, Extension.LOWERCASE);
            
            // Cleanup
            tempDoc.close(SaveOptions.DONOTSAVECHANGES);
            app.activeDocument = doc;
            doc.activeLayer = originalActiveLayer;
            
            $.writeln('[FileOps] Exported layer to: ' + outputPath);
            return outputPath;
            
        } catch (error) {
            $.writeln('[FileOps] Export error: ' + error.message);
            throw new Error("Failed to export layer: " + error.message);
        }
    };
    
    ops.importTIFFAsLayer = function(filePath, layerName) {
        /**
         * Import TIFF file as new layer
         * @param {String} filePath - Path to TIFF file
         * @param {String} layerName - Name for new layer
         * @returns {Layer} Imported layer
         */
        try {
            var doc = app.activeDocument;
            var file = new File(filePath);
            
            if (!file.exists) {
                throw new Error("File not found: " + filePath);
            }
            
            // Open file as document
            var importDoc = app.open(file);
            
            // Flatten if multiple layers
            if (importDoc.layers.length > 1) {
                importDoc.flatten();
            }
            
            // Select all and copy
            importDoc.selection.selectAll();
            importDoc.selection.copy();
            
            // Switch to target document and paste
            app.activeDocument = doc;
            doc.paste();
            
            // Rename the pasted layer
            var newLayer = doc.activeLayer;
            newLayer.name = layerName || ("Imported_" + Date.now());
            
            // Close import document
            importDoc.close(SaveOptions.DONOTSAVECHANGES);
            
            $.writeln('[FileOps] Imported TIFF as layer: ' + newLayer.name);
            return newLayer;
            
        } catch (error) {
            $.writeln('[FileOps] Import error: ' + error.message);
            throw new Error("Failed to import TIFF: " + error.message);
        }
    };
    
    ops.cleanupTempFiles = function(filePaths) {
        /**
         * Delete temporary files
         * @param {Array} filePaths - Array of file paths to delete
         */
        try {
            for (var i = 0; i < filePaths.length; i++) {
                var file = new File(filePaths[i]);
                if (file.exists) {
                    file.remove();
                    $.writeln('[FileOps] Deleted temp file: ' + filePaths[i]);
                }
            }
        } catch (error) {
            $.writeln('[FileOps] Cleanup error: ' + error.message);
        }
    };
    
    ops.ensureTempDirectory = function() {
        /**
         * Ensure temp directory exists
         * @returns {String} Temp directory path
         */
        try {
            var tempDir = Folder.temp + '/gattonero_temp/';
            var tempFolder = new Folder(tempDir);
            
            if (!tempFolder.exists) {
                tempFolder.create();
                $.writeln('[FileOps] Created temp directory: ' + tempDir);
            }
            
            return tempDir;
            
        } catch (error) {
            $.writeln('[FileOps] Error creating temp directory: ' + error.message);
            throw error;
        }
    };
    
    return ops;
})();
```

---

## 🌐 API COMMUNICATION

### HTTP Requests via curl (host/api-communication.jsx)

> **✅ POPRAWKA:** Używamy curl zamiast XMLHttpRequest dla komunikacji z API

```javascript
// host/api-communication.jsx - API communication via curl

var APIComm = (function() {
    'use strict';
    
    var comm = {};
    var API_BASE_URL = 'http://127.0.0.1:5000/api';
    
    comm.makeAPICall = function(endpoint, params) {
        /**
         * Make API call using curl
         * @param {String} endpoint - API endpoint
         * @param {Object} params - Request parameters
         * @returns {Object} API response
         */
        try {
            $.writeln('[APIComm] Making API call to: ' + endpoint);
            
            var url = API_BASE_URL + '/' + endpoint;
            var tempFile = Folder.temp + '/gattonero_response_' + Date.now() + '.json';
            
            // Prepare curl command based on endpoint
            var curlCmd = comm.buildCurlCommand(url, params, tempFile);
            
            $.writeln('[APIComm] Executing curl command');
            var result = system.callSystem(curlCmd);
            
            if (result !== 0) {
                throw new Error('Curl command failed with exit code: ' + result);
            }
            
            // Read response from temp file
            var responseFile = new File(tempFile);
            if (!responseFile.exists) {
                throw new Error('Response file not found');
            }
            
            responseFile.open('r');
            var responseText = responseFile.read();
            responseFile.close();
            responseFile.remove();
            
            // Parse JSON response
            var response = JSON.parse(responseText);
            $.writeln('[APIComm] API call successful');
            
            return response;
            
        } catch (error) {
            $.writeln('[APIComm] API call error: ' + error.message);
            return {
                success: false,
                error: error.message
            };
        }
    };
    
    comm.buildCurlCommand = function(url, params, outputFile) {
        /**
         * Build curl command based on request type
         * @param {String} url - API URL
         * @param {Object} params - Request parameters
         * @param {String} outputFile - Output file path
         * @returns {String} Curl command
         */
        var cmd = 'curl -s';
        
        // Add headers
        cmd += ' -H "Content-Type: application/json"';
        cmd += ' -H "Accept: application/json"';
        
        // Handle file uploads vs. regular JSON
        if (params.source_image || params.target_image || params.image_path) {
            // File upload - use multipart/form-data
            cmd = 'curl -s';
            cmd += ' -X POST';
            
            if (params.source_image) {
                cmd += ' -F "source_image=@' + params.source_image + '"';
            }
            if (params.target_image) {
                cmd += ' -F "target_image=@' + params.target_image + '"';
            }
            if (params.image_path) {
                cmd += ' -F "image=@' + params.image_path + '"';
            }
            if (params.params) {
                cmd += ' -F "params=' + JSON.stringify(params.params) + '"';
            }
        } else {
            // Regular JSON request
            cmd += ' -X POST';
            cmd += ' -d \'' + JSON.stringify(params) + '\'';
        }
        
        // Add URL and output redirection
        cmd += ' "' + url + '"';
        cmd += ' > "' + outputFile + '"';
        
        $.writeln('[APIComm] Curl command: ' + cmd);
        return cmd;
    };
    
    comm.checkServerStatus = function() {
        /**
         * Check if API server is running
         * @returns {Boolean} Server status
         */
        try {
            var url = API_BASE_URL + '/status';
            var tempFile = Folder.temp + '/gattonero_status_' + Date.now() + '.json';
            
            var curlCmd = 'curl -s --connect-timeout 5 "' + url + '" > "' + tempFile + '"';
            var result = system.callSystem(curlCmd);
            
            if (result === 0) {
                var responseFile = new File(tempFile);
                if (responseFile.exists) {
                    responseFile.open('r');
                    var responseText = responseFile.read();
                    responseFile.close();
                    responseFile.remove();
                    
                    try {
                        var response = JSON.parse(responseText);
                        return response.status === 'online';
                    } catch (e) {
                        return false;
                    }
                }
            }
            
            return false;
            
        } catch (error) {
            $.writeln('[APIComm] Status check error: ' + error.message);
            return false;
        }
    };
    
    return comm;
})();
```

---

## 🛠️ UTILITY FUNCTIONS

### Helper Functions (host/utils.jsx)

```javascript
// host/utils.jsx - Utility functions

var Utils = (function() {
    'use strict';
    
    var utils = {};
    
    utils.logMessage = function(message, level) {
        /**
         * Log message with timestamp
         * @param {String} message - Message to log
         * @param {String} level - Log level (info, warn, error)
         */
        var timestamp = new Date().toLocaleString();
        var prefix = '[' + timestamp + '] [' + (level || 'INFO').toUpperCase() + ']';
        $.writeln(prefix + ' ' + message);
    };
    
    utils.validateDocument = function() {
        /**
         * Validate active document
         * @returns {Object} Validation result
         */
        try {
            var doc = app.activeDocument;
            if (!doc) {
                return {
                    valid: false,
                    error: 'No active document'
                };
            }
            
            if (doc.layers.length === 0) {
                return {
                    valid: false,
                    error: 'Document has no layers'
                };
            }
            
            return {
                valid: true,
                document: doc
            };
            
        } catch (error) {
            return {
                valid: false,
                error: error.message
            };
        }
    };
    
    utils.convertColorToHex = function(color) {
        /**
         * Convert Photoshop color to hex
         * @param {SolidColor} color - Photoshop color object
         * @returns {String} Hex color string
         */
        try {
            var r = Math.round(color.rgb.red);
            var g = Math.round(color.rgb.green);
            var b = Math.round(color.rgb.blue);
            
            return '#' + 
                   (r < 16 ? '0' : '') + r.toString(16) +
                   (g < 16 ? '0' : '') + g.toString(16) +
                   (b < 16 ? '0' : '') + b.toString(16);
                   
        } catch (error) {
            utils.logMessage('Color conversion error: ' + error.message, 'error');
            return '#000000';
        }
    };
    
    utils.createProgressCallback = function(total) {
        /**
         * Create progress tracking callback
         * @param {Number} total - Total number of steps
         * @returns {Function} Progress callback function
         */
        var current = 0;
        
        return function(message) {
            current++;
            var percentage = Math.round((current / total) * 100);
            utils.logMessage('[' + percentage + '%] ' + message, 'info');
        };
    };
    
    utils.sanitizeFilename = function(filename) {
        /**
         * Sanitize filename for cross-platform compatibility
         * @param {String} filename - Original filename
         * @returns {String} Sanitized filename
         */
        return filename.replace(/[<>:"/\\|?*]/g, '_')
                      .replace(/\s+/g, '_')
                      .toLowerCase();
    };
    
    utils.formatFileSize = function(bytes) {
        /**
         * Format file size in human readable format
         * @param {Number} bytes - File size in bytes
         * @returns {String} Formatted size
         */
        var sizes = ['B', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 B';
        
        var i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    };
    
    return utils;
})();
```

---

## 🔗 COMMUNICATION BRIDGE

### CEP ↔ ExtendScript Bridge

```javascript
// host/main.jsx - Communication bridge setup

// Global function accessible from CEP
function processFromCEP(functionName, params) {
    /**
     * Main entry point for CEP calls
     * @param {String} functionName - Function to execute
     * @param {Object} params - Function parameters
     * @returns {Object} Result object
     */
    try {
        Utils.logMessage('Processing CEP call: ' + functionName, 'info');
        
        switch (functionName) {
            case 'getLayers':
                return {
                    success: true,
                    data: LayerOperations.getLayers()
                };
                
            case 'selectLayer':
                LayerOperations.selectLayer(params.layerId);
                return {
                    success: true,
                    message: 'Layer selected'
                };
                
            case 'processColorMatching':
                return processColorMatching(params);
                
            case 'analyzePalette':
                return analyzePalette(params);
                
            case 'checkServerStatus':
                return {
                    success: true,
                    online: APIComm.checkServerStatus()
                };
                
            default:
                throw new Error('Unknown function: ' + functionName);
        }
        
    } catch (error) {
        Utils.logMessage('CEP call error: ' + error.message, 'error');
        return {
            success: false,
            error: error.message
        };
    }
}
```

---

## 🔗 NAVIGATION

**📖 Rozdziały:**
- **[⬅️ Chapter 3 - CEP Panel Interface](./gatto-WORKING-04-photoshop-chapter3.md)**
- **[➡️ Chapter 5 - JavaScript Application Logic](./gatto-WORKING-04-photoshop-chapter5.md)**
- **[📋 Spis treści](./gatto-WORKING-04-photoshop-toc.md)**

---

*Ten rozdział opisuje wszystkie kluczowe funkcje ExtendScript odpowiedzialne za komunikację z Photoshopem, operacje na warstwach, zarządzanie plikami oraz komunikację z API poprzez curl.*
