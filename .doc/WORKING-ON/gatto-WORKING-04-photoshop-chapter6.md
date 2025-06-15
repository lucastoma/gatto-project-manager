# GattoNero AI Assistant - Photoshop Integration
## Chapter 6: Integration Testing

> **Status:** 📝 AKTYWNY  
> **Poprzedni:** [Chapter 5 - JavaScript Application Logic](./gatto-WORKING-04-photoshop-chapter5.md)  
> **Następny:** [Chapter 7 - Deployment & Troubleshooting](./gatto-WORKING-04-photoshop-chapter7.md)  
> **Spis treści:** [gatto-WORKING-04-photoshop-toc.md](./gatto-WORKING-04-photoshop-toc.md)

---

## 🧪 INTEGRATION TESTING

### Test Framework Setup (test/test-framework.jsx)

```javascript
// test/test-framework.jsx - Simple testing framework for ExtendScript

var TestFramework = (function() {
    'use strict';
    
    var framework = {};
    var testResults = [];
    var currentSuite = null;
    
    framework.describe = function(suiteName, suiteFunc) {
        /**
         * Define a test suite
         * @param {String} suiteName - Name of the test suite
         * @param {Function} suiteFunc - Function containing tests
         */
        currentSuite = {
            name: suiteName,
            tests: [],
            passed: 0,
            failed: 0
        };
        
        $.writeln('\n=== Test Suite: ' + suiteName + ' ===');
        
        try {
            suiteFunc();
            framework.reportSuite();
        } catch (error) {
            $.writeln('Suite Error: ' + error.message);
        }
        
        testResults.push(currentSuite);
        currentSuite = null;
    };
    
    framework.it = function(testName, testFunc) {
        /**
         * Define a test case
         * @param {String} testName - Name of the test
         * @param {Function} testFunc - Test function
         */
        if (!currentSuite) {
            throw new Error('Test must be inside a describe block');
        }
        
        var test = {
            name: testName,
            passed: false,
            error: null,
            duration: 0
        };
        
        var startTime = Date.now();
        
        try {
            testFunc();
            test.passed = true;
            currentSuite.passed++;
            $.writeln('  ✓ ' + testName);
        } catch (error) {
            test.passed = false;
            test.error = error.message;
            currentSuite.failed++;
            $.writeln('  ✗ ' + testName + ' - ' + error.message);
        }
        
        test.duration = Date.now() - startTime;
        currentSuite.tests.push(test);
    };
    
    framework.expect = function(actual) {
        /**
         * Create an expectation for testing
         * @param {*} actual - Actual value
         * @returns {Object} Expectation object
         */
        return {
            toBe: function(expected) {
                if (actual !== expected) {
                    throw new Error('Expected ' + expected + ' but got ' + actual);
                }
            },
            
            toBeTrue: function() {
                if (actual !== true) {
                    throw new Error('Expected true but got ' + actual);
                }
            },
            
            toBeFalse: function() {
                if (actual !== false) {
                    throw new Error('Expected false but got ' + actual);
                }
            },
            
            toBeNull: function() {
                if (actual !== null) {
                    throw new Error('Expected null but got ' + actual);
                }
            },
            
            toBeUndefined: function() {
                if (actual !== undefined) {
                    throw new Error('Expected undefined but got ' + actual);
                }
            },
            
            toContain: function(expected) {
                if (actual.indexOf && actual.indexOf(expected) === -1) {
                    throw new Error('Expected to contain ' + expected);
                }
            },
            
            toThrow: function() {
                var threw = false;
                try {
                    actual();
                } catch (e) {
                    threw = true;
                }
                if (!threw) {
                    throw new Error('Expected function to throw an error');
                }
            }
        };
    };
    
    framework.reportSuite = function() {
        /**
         * Report test suite results
         */
        if (!currentSuite) return;
        
        var total = currentSuite.passed + currentSuite.failed;
        $.writeln('Results: ' + currentSuite.passed + '/' + total + ' passed');
        
        if (currentSuite.failed > 0) {
            $.writeln('❌ ' + currentSuite.failed + ' test(s) failed');
        } else {
            $.writeln('✅ All tests passed!');
        }
    };
    
    framework.reportAll = function() {
        /**
         * Report all test results
         */
        $.writeln('\n=== Test Summary ===');
        
        var totalPassed = 0;
        var totalFailed = 0;
        
        testResults.forEach(function(suite) {
            totalPassed += suite.passed;
            totalFailed += suite.failed;
            $.writeln(suite.name + ': ' + suite.passed + '/' + 
                     (suite.passed + suite.failed) + ' passed');
        });
        
        $.writeln('\nOverall: ' + totalPassed + '/' + 
                 (totalPassed + totalFailed) + ' tests passed');
        
        return {
            passed: totalPassed,
            failed: totalFailed,
            suites: testResults
        };
    };
    
    return framework;
})();
```

---

## 🧪 UNIT TESTS

### Layer Operations Tests (test/layer-operations.test.jsx)

```javascript
// test/layer-operations.test.jsx - Layer operations unit tests

#include "test-framework.jsx"
#include "../host/layer-operations.jsx"
#include "../host/utils.jsx"

(function() {
    'use strict';
    
    TestFramework.describe('Layer Operations', function() {
        
        TestFramework.it('should get layers from active document', function() {
            // Ensure we have an active document
            if (!app.activeDocument) {
                // Create a test document
                var testDoc = app.documents.add(800, 600, 72, "Test Document");
            }
            
            var layers = LayerOperations.getLayers();
            
            TestFramework.expect(layers).toBeObject();
            TestFramework.expect(layers.length).toBeGreaterThan(0);
            
            // Check layer properties
            var firstLayer = layers[0];
            TestFramework.expect(firstLayer.id).toBeDefined();
            TestFramework.expect(firstLayer.name).toBeDefined();
            TestFramework.expect(firstLayer.visible).toBeDefined();
        });
        
        TestFramework.it('should find layer by ID', function() {
            var layers = LayerOperations.getLayers();
            if (layers.length === 0) return;
            
            var firstLayer = layers[0];
            var foundLayer = LayerOperations.getLayerById(firstLayer.id);
            
            TestFramework.expect(foundLayer).toBeDefined();
            TestFramework.expect(foundLayer.id).toBe(firstLayer.id);
            TestFramework.expect(foundLayer.name).toBe(firstLayer.name);
        });
        
        TestFramework.it('should return null for non-existent layer ID', function() {
            var nonExistentLayer = LayerOperations.getLayerById(99999);
            TestFramework.expect(nonExistentLayer).toBeNull();
        });
        
        TestFramework.it('should select layer by ID', function() {
            var layers = LayerOperations.getLayers();
            if (layers.length === 0) return;
            
            var targetLayer = layers[0];
            var result = LayerOperations.selectLayer(targetLayer.id);
            
            TestFramework.expect(result).toBeTrue();
            TestFramework.expect(app.activeDocument.activeLayer.id).toBe(targetLayer.id);
        });
        
        TestFramework.it('should duplicate layer', function() {
            var doc = app.activeDocument;
            var originalLayer = doc.activeLayer;
            var originalLayerCount = doc.layers.length;
            
            var duplicatedLayer = LayerOperations.duplicateLayer(originalLayer, 'Test Duplicate');
            
            TestFramework.expect(duplicatedLayer).toBeDefined();
            TestFramework.expect(duplicatedLayer.name).toBe('Test Duplicate');
            TestFramework.expect(doc.layers.length).toBe(originalLayerCount + 1);
            
            // Cleanup
            duplicatedLayer.remove();
        });
    });
    
    // Add helper methods to TestFramework for these tests
    TestFramework.expect.prototype.toBeObject = function() {
        if (typeof this.actual !== 'object' || this.actual === null) {
            throw new Error('Expected object but got ' + typeof this.actual);
        }
    };
    
    TestFramework.expect.prototype.toBeDefined = function() {
        if (this.actual === undefined) {
            throw new Error('Expected value to be defined');
        }
    };
    
    TestFramework.expect.prototype.toBeGreaterThan = function(expected) {
        if (this.actual <= expected) {
            throw new Error('Expected ' + this.actual + ' to be greater than ' + expected);
        }
    };
    
})();
```

---

## 🔧 FILE OPERATIONS TESTS

### File Operations Tests (test/file-operations.test.jsx)

```javascript
// test/file-operations.test.jsx - File operations unit tests

#include "test-framework.jsx"
#include "../host/file-operations.jsx"
#include "../host/layer-operations.jsx"

(function() {
    'use strict';
    
    TestFramework.describe('File Operations', function() {
        
        var testTempDir;
        var testDoc;
        
        // Setup
        function setup() {
            // Create temp directory
            testTempDir = Folder.temp + '/gattonero_test_' + Date.now() + '/';
            var tempFolder = new Folder(testTempDir);
            tempFolder.create();
            
            // Create test document with content
            testDoc = app.documents.add(400, 300, 72, "Test Export Doc");
            
            // Add some content to the layer
            var layer = testDoc.activeLayer;
            layer.name = "Test Layer";
            
            // Fill with color for testing
            app.foregroundColor.rgb.red = 255;
            app.foregroundColor.rgb.green = 100;
            app.foregroundColor.rgb.blue = 50;
            
            testDoc.selection.selectAll();
            testDoc.selection.fill(app.foregroundColor);
            testDoc.selection.deselect();
        }
        
        // Cleanup
        function cleanup() {
            if (testDoc) {
                testDoc.close(SaveOptions.DONOTSAVECHANGES);
            }
            
            if (testTempDir) {
                var tempFolder = new Folder(testTempDir);
                if (tempFolder.exists) {
                    // Remove all files in temp directory
                    var files = tempFolder.getFiles();
                    for (var i = 0; i < files.length; i++) {
                        files[i].remove();
                    }
                    tempFolder.remove();
                }
            }
        }
        
        TestFramework.it('should export layer to TIFF', function() {
            setup();
            
            try {
                var layer = testDoc.activeLayer;
                var outputPath = testTempDir + 'test_export.tiff';
                
                var result = FileOperations.exportLayerToTIFF(layer, outputPath);
                
                TestFramework.expect(result).toBe(outputPath);
                
                // Check if file was created
                var outputFile = new File(outputPath);
                TestFramework.expect(outputFile.exists).toBeTrue();
                
                // Check file size (should be > 0)
                TestFramework.expect(outputFile.length).toBeGreaterThan(0);
                
            } finally {
                cleanup();
            }
        });
        
        TestFramework.it('should import TIFF as layer', function() {
            setup();
            
            try {
                // First export a layer
                var originalLayer = testDoc.activeLayer;
                var exportPath = testTempDir + 'test_for_import.tiff';
                
                FileOperations.exportLayerToTIFF(originalLayer, exportPath);
                
                // Now import it back
                var originalLayerCount = testDoc.layers.length;
                var importedLayer = FileOperations.importTIFFAsLayer(exportPath, 'Imported Test Layer');
                
                TestFramework.expect(importedLayer).toBeDefined();
                TestFramework.expect(importedLayer.name).toBe('Imported Test Layer');
                TestFramework.expect(testDoc.layers.length).toBe(originalLayerCount + 1);
                
            } finally {
                cleanup();
            }
        });
        
        TestFramework.it('should handle non-existent file import', function() {
            setup();
            
            try {
                var nonExistentPath = testTempDir + 'does_not_exist.tiff';
                
                TestFramework.expect(function() {
                    FileOperations.importTIFFAsLayer(nonExistentPath, 'Should Fail');
                }).toThrow();
                
            } finally {
                cleanup();
            }
        });
        
        TestFramework.it('should cleanup temp files', function() {
            setup();
            
            try {
                // Create some temp files
                var tempFile1 = new File(testTempDir + 'temp1.txt');
                var tempFile2 = new File(testTempDir + 'temp2.txt');
                
                tempFile1.open('w');
                tempFile1.write('test content');
                tempFile1.close();
                
                tempFile2.open('w');
                tempFile2.write('test content');
                tempFile2.close();
                
                TestFramework.expect(tempFile1.exists).toBeTrue();
                TestFramework.expect(tempFile2.exists).toBeTrue();
                
                // Cleanup files
                FileOperations.cleanupTempFiles([tempFile1.fsName, tempFile2.fsName]);
                
                TestFramework.expect(tempFile1.exists).toBeFalse();
                TestFramework.expect(tempFile2.exists).toBeFalse();
                
            } finally {
                cleanup();
            }
        });
        
        TestFramework.it('should ensure temp directory creation', function() {
            var tempDir = FileOperations.ensureTempDirectory();
            
            TestFramework.expect(tempDir).toBeDefined();
            TestFramework.expect(typeof tempDir).toBe('string');
            
            var tempFolder = new Folder(tempDir);
            TestFramework.expect(tempFolder.exists).toBeTrue();
        });
    });
    
})();
```

---

## 🌐 API COMMUNICATION TESTS

### API Communication Tests (test/api-communication.test.jsx)

```javascript
// test/api-communication.test.jsx - API communication tests

#include "test-framework.jsx"
#include "../host/api-communication.jsx"

(function() {
    'use strict';
    
    TestFramework.describe('API Communication', function() {
        
        TestFramework.it('should build curl command for JSON request', function() {
            var url = 'http://127.0.0.1:5000/api/test';
            var params = {
                method: 'test',
                param1: 'value1',
                param2: 42
            };
            var outputFile = '/tmp/response.json';
            
            var command = APIComm.buildCurlCommand(url, params, outputFile);
            
            TestFramework.expect(command).toContain('curl -s');
            TestFramework.expect(command).toContain('-X POST');
            TestFramework.expect(command).toContain('Content-Type: application/json');
            TestFramework.expect(command).toContain(url);
            TestFramework.expect(command).toContain(outputFile);
        });
        
        TestFramework.it('should build curl command for file upload', function() {
            var url = 'http://127.0.0.1:5000/api/upload';
            var params = {
                source_image: '/path/to/source.tiff',
                target_image: '/path/to/target.tiff',
                params: {
                    method: 'histogram',
                    strength: 80
                }
            };
            var outputFile = '/tmp/response.json';
            
            var command = APIComm.buildCurlCommand(url, params, outputFile);
            
            TestFramework.expect(command).toContain('curl -s');
            TestFramework.expect(command).toContain('-X POST');
            TestFramework.expect(command).toContain('-F "source_image=@/path/to/source.tiff"');
            TestFramework.expect(command).toContain('-F "target_image=@/path/to/target.tiff"');
            TestFramework.expect(command).toContain(url);
        });
        
        TestFramework.it('should check server status', function() {
            // Note: This test requires the server to be running
            // In a real test environment, you might mock this
            
            var status = APIComm.checkServerStatus();
            
            // Status should be either true or false, not undefined
            TestFramework.expect(typeof status).toBe('boolean');
        });
        
        // Mock test for API call (since we can't rely on server being up)
        TestFramework.it('should handle API call response parsing', function() {
            // Create a mock response file
            var tempFile = Folder.temp + '/test_response.json';
            var mockResponse = {
                success: true,
                message: 'Test successful',
                data: { test: 'value' }
            };
            
            var file = new File(tempFile);
            file.open('w');
            file.write(JSON.stringify(mockResponse));
            file.close();
            
            // Test JSON parsing (simulate what happens in makeAPICall)
            file.open('r');
            var responseText = file.read();
            file.close();
            file.remove();
            
            var parsedResponse = JSON.parse(responseText);
            
            TestFramework.expect(parsedResponse.success).toBeTrue();
            TestFramework.expect(parsedResponse.message).toBe('Test successful');
            TestFramework.expect(parsedResponse.data.test).toBe('value');
        });
    });
    
})();
```

---

## 🧪 INTEGRATION TESTS

### Full Integration Tests (test/integration.test.jsx)

```javascript
// test/integration.test.jsx - Full integration tests

#include "test-framework.jsx"
#include "../host/main.jsx"

(function() {
    'use strict';
    
    TestFramework.describe('Integration Tests', function() {
        
        var testDoc;
        var testTempDir;
        
        function setupIntegrationTest() {
            // Create test document with multiple layers
            testDoc = app.documents.add(800, 600, 72, "Integration Test Doc");
            
            // Create test layers with different content
            createTestLayer('Red Layer', [255, 0, 0]);
            createTestLayer('Blue Layer', [0, 0, 255]);
            createTestLayer('Green Layer', [0, 255, 0]);
            
            // Setup temp directory
            testTempDir = Folder.temp + '/gattonero_integration_' + Date.now() + '/';
            var tempFolder = new Folder(testTempDir);
            tempFolder.create();
        }
        
        function createTestLayer(name, rgbColor) {
            var layer = testDoc.artLayers.add();
            layer.name = name;
            
            // Fill with color
            var color = new SolidColor();
            color.rgb.red = rgbColor[0];
            color.rgb.green = rgbColor[1];
            color.rgb.blue = rgbColor[2];
            
            testDoc.activeLayer = layer;
            testDoc.selection.selectAll();
            testDoc.selection.fill(color);
            testDoc.selection.deselect();
            
            return layer;
        }
        
        function cleanupIntegrationTest() {
            if (testDoc) {
                testDoc.close(SaveOptions.DONOTSAVECHANGES);
            }
            
            if (testTempDir) {
                var tempFolder = new Folder(testTempDir);
                if (tempFolder.exists) {
                    var files = tempFolder.getFiles();
                    for (var i = 0; i < files.length; i++) {
                        files[i].remove();
                    }
                    tempFolder.remove();
                }
            }
        }
        
        TestFramework.it('should initialize GattoNero API', function() {
            var result = GattoNeroAPI.initialize();
            
            TestFramework.expect(result.success).toBeTrue();
            TestFramework.expect(result.version).toBeDefined();
        });
        
        TestFramework.it('should get layers through API', function() {
            setupIntegrationTest();
            
            try {
                var layers = GattoNeroAPI.getLayers();
                
                TestFramework.expect(layers).toBeDefined();
                TestFramework.expect(layers.length).toBeGreaterThan(0);
                
                // Should have our test layers
                var layerNames = layers.map(function(layer) { return layer.name; });
                TestFramework.expect(layerNames).toContain('Red Layer');
                TestFramework.expect(layerNames).toContain('Blue Layer');
                TestFramework.expect(layerNames).toContain('Green Layer');
                
            } finally {
                cleanupIntegrationTest();
            }
        });
        
        TestFramework.it('should select layer through API', function() {
            setupIntegrationTest();
            
            try {
                var layers = GattoNeroAPI.getLayers();
                var targetLayer = layers.find(function(layer) {
                    return layer.name === 'Blue Layer';
                });
                
                TestFramework.expect(targetLayer).toBeDefined();
                
                GattoNeroAPI.selectLayer(targetLayer.id);
                
                TestFramework.expect(testDoc.activeLayer.name).toBe('Blue Layer');
                
            } finally {
                cleanupIntegrationTest();
            }
        });
        
        TestFramework.it('should export and import layer through API', function() {
            setupIntegrationTest();
            
            try {
                var layers = GattoNeroAPI.getLayers();
                var sourceLayer = layers.find(function(layer) {
                    return layer.name === 'Red Layer';
                });
                
                TestFramework.expect(sourceLayer).toBeDefined();
                
                // Export layer
                var exportPath = testTempDir + 'exported_red_layer.tiff';
                var result = GattoNeroAPI.exportLayer(sourceLayer, exportPath);
                
                TestFramework.expect(result).toBe(exportPath);
                
                var exportedFile = new File(exportPath);
                TestFramework.expect(exportedFile.exists).toBeTrue();
                
                // Import layer back
                var originalLayerCount = testDoc.layers.length;
                var importedLayer = GattoNeroAPI.importLayer(exportPath, 'Imported Red Layer');
                
                TestFramework.expect(importedLayer).toBeDefined();
                TestFramework.expect(importedLayer.name).toBe('Imported Red Layer');
                TestFramework.expect(testDoc.layers.length).toBe(originalLayerCount + 1);
                
            } finally {
                cleanupIntegrationTest();
            }
        });
        
        TestFramework.it('should handle palette analysis workflow', function() {
            setupIntegrationTest();
            
            try {
                var layers = GattoNeroAPI.getLayers();
                var testLayer = layers.find(function(layer) {
                    return layer.name === 'Red Layer';
                });
                
                TestFramework.expect(testLayer).toBeDefined();
                
                var params = {
                    layerId: testLayer.id,
                    colorCount: 5,
                    tolerance: 10
                };
                
                // Note: This test will fail if the API server is not running
                // In a complete test environment, you would mock the API response
                var result = analyzePalette(params);
                
                // Should handle the case where server is offline gracefully
                TestFramework.expect(result).toBeDefined();
                TestFramework.expect(typeof result.success).toBe('boolean');
                
                if (result.success) {
                    TestFramework.expect(result.palette).toBeDefined();
                    TestFramework.expect(result.palette.length).toBeGreaterThan(0);
                }
                
            } finally {
                cleanupIntegrationTest();
            }
        });
    });
    
})();
```

---

## 🎯 TEST RUNNER

### Main Test Runner (test/run-tests.jsx)

```javascript
// test/run-tests.jsx - Main test runner

#target photoshop

(function() {
    'use strict';
    
    $.writeln('='.repeat(60));
    $.writeln('GattoNero AI Assistant - Test Suite');
    $.writeln('='.repeat(60));
    
    // Include all test files
    try {
        $.writeln('Loading test framework...');
        #include "test-framework.jsx"
        
        $.writeln('Running Layer Operations Tests...');
        #include "layer-operations.test.jsx"
        
        $.writeln('Running File Operations Tests...');
        #include "file-operations.test.jsx"
        
        $.writeln('Running API Communication Tests...');
        #include "api-communication.test.jsx"
        
        $.writeln('Running Integration Tests...');
        #include "integration.test.jsx"
        
        // Generate final report
        var finalReport = TestFramework.reportAll();
        
        $.writeln('\n' + '='.repeat(60));
        $.writeln('FINAL TEST REPORT');
        $.writeln('='.repeat(60));
        
        if (finalReport.failed === 0) {
            $.writeln('🎉 ALL TESTS PASSED! 🎉');
            $.writeln('Total: ' + finalReport.passed + ' tests passed');
        } else {
            $.writeln('❌ SOME TESTS FAILED');
            $.writeln('Passed: ' + finalReport.passed);
            $.writeln('Failed: ' + finalReport.failed);
            $.writeln('Success Rate: ' + 
                     Math.round((finalReport.passed / (finalReport.passed + finalReport.failed)) * 100) + '%');
        }
        
        $.writeln('='.repeat(60));
        
        // Return summary for potential automation
        return {
            success: finalReport.failed === 0,
            passed: finalReport.passed,
            failed: finalReport.failed,
            suites: finalReport.suites
        };
        
    } catch (error) {
        $.writeln('❌ TEST RUNNER ERROR: ' + error.message);
        return {
            success: false,
            error: error.message
        };
    }
})();
```

---

## 🔧 MANUAL TESTING PROCEDURES

### Manual Test Checklist

```markdown
# GattoNero Manual Testing Checklist

## Pre-Test Setup
- [ ] Python API server is running on port 5000
- [ ] Photoshop is running with test document open
- [ ] GattoNero extension is installed and visible
- [ ] Test images are available in various formats

## UI Testing
- [ ] Extension panel loads without errors
- [ ] All function cards are clickable and responsive
- [ ] Status indicator shows correct server connection status
- [ ] Layer dropdowns populate with current document layers
- [ ] Form sliders update values correctly
- [ ] Progress bars display during operations
- [ ] Activity log shows messages correctly

## Layer Operations Testing
- [ ] Layer dropdown refreshes when document changes
- [ ] Layer selection works correctly
- [ ] Multi-layer documents are handled properly
- [ ] Nested layer groups are supported
- [ ] Layer visibility states are respected

## Color Matching Testing
- [ ] Color matching with different algorithms works
- [ ] Result layers are created with appropriate names
- [ ] Original layers remain unchanged
- [ ] Error handling for invalid layer selections
- [ ] Progress feedback during processing

## Palette Analysis Testing
- [ ] Palette extraction completes successfully
- [ ] Color swatches display correctly in grid
- [ ] Color values can be copied to clipboard
- [ ] Palette can be saved as JSON file
- [ ] Different color count settings work

## File Operations Testing
- [ ] TIFF export creates valid files
- [ ] Exported files can be opened in other applications
- [ ] Import creates new layers correctly
- [ ] Temporary files are cleaned up properly
- [ ] File size limits are respected

## API Communication Testing
- [ ] Server status checking works
- [ ] File uploads complete successfully
- [ ] API responses are parsed correctly
- [ ] Error handling for server offline scenarios
- [ ] Timeout handling for slow responses

## Error Handling Testing
- [ ] Invalid layer selections show appropriate errors
- [ ] Network errors are handled gracefully
- [ ] File system errors are reported clearly
- [ ] Memory limitations are managed
- [ ] User can recover from error states

## Performance Testing
- [ ] Large images process within reasonable time
- [ ] Memory usage remains stable during operations
- [ ] UI remains responsive during processing
- [ ] Multiple operations can be performed sequentially
- [ ] Extension can be used for extended periods

## Integration Testing
- [ ] Extension works with different Photoshop versions
- [ ] Compatible with various document color modes
- [ ] Works with different layer types (smart objects, text, etc.)
- [ ] Integration with Photoshop's undo system
- [ ] Compatibility with other extensions
```

---

## 🔍 DEBUGGING TOOLS

### Debug Console Setup

```javascript
// debug/debug-console.js - Debug console for development

class DebugConsole {
    constructor() {
        this.enabled = false;
        this.logLevel = 'info'; // 'debug', 'info', 'warn', 'error'
        this.maxLogEntries = 500;
        this.logs = [];
        
        this.setupConsole();
    }
    
    setupConsole() {
        // Create debug overlay
        this.createDebugOverlay();
        
        // Override console methods
        this.interceptConsoleLogs();
        
        // Add keyboard shortcut to toggle (Ctrl+Shift+D)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                this.toggle();
            }
        });
    }
    
    createDebugOverlay() {
        this.overlay = document.createElement('div');
        this.overlay.id = 'debug-console';
        this.overlay.style.cssText = `
            position: fixed;
            top: 0;
            right: 0;
            width: 400px;
            height: 300px;
            background: rgba(0, 0, 0, 0.9);
            color: #00ff00;
            font-family: 'Courier New', monospace;
            font-size: 10px;
            z-index: 10000;
            overflow-y: auto;
            padding: 10px;
            border-left: 2px solid #00ff00;
            display: none;
        `;
        
        document.body.appendChild(this.overlay);
    }
    
    interceptConsoleLogs() {
        const originalLog = console.log;
        const originalError = console.error;
        const originalWarn = console.warn;
        
        console.log = (...args) => {
            this.addLog('info', args.join(' '));
            originalLog.apply(console, args);
        };
        
        console.error = (...args) => {
            this.addLog('error', args.join(' '));
            originalError.apply(console, args);
        };
        
        console.warn = (...args) => {
            this.addLog('warn', args.join(' '));
            originalWarn.apply(console, args);
        };
    }
    
    addLog(level, message) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = {
            level: level,
            message: message,
            timestamp: timestamp
        };
        
        this.logs.push(logEntry);
        
        // Limit log entries
        if (this.logs.length > this.maxLogEntries) {
            this.logs.shift();
        }
        
        this.updateDisplay();
    }
    
    updateDisplay() {
        if (!this.enabled || !this.overlay) return;
        
        const filteredLogs = this.logs.filter(log => 
            this.shouldShowLog(log.level)
        );
        
        this.overlay.innerHTML = filteredLogs
            .slice(-50) // Show last 50 entries
            .map(log => this.formatLogEntry(log))
            .join('\n');
        
        // Auto-scroll to bottom
        this.overlay.scrollTop = this.overlay.scrollHeight;
    }
    
    formatLogEntry(log) {
        const colors = {
            debug: '#888',
            info: '#00ff00',
            warn: '#ffaa00',
            error: '#ff0000'
        };
        
        const color = colors[log.level] || '#00ff00';
        
        return `<div style="color: ${color}">` +
               `[${log.timestamp}] ${log.level.toUpperCase()}: ${log.message}` +
               `</div>`;
    }
    
    shouldShowLog(level) {
        const levels = ['debug', 'info', 'warn', 'error'];
        const currentIndex = levels.indexOf(this.logLevel);
        const logIndex = levels.indexOf(level);
        
        return logIndex >= currentIndex;
    }
    
    toggle() {
        this.enabled = !this.enabled;
        this.overlay.style.display = this.enabled ? 'block' : 'none';
        
        if (this.enabled) {
            this.updateDisplay();
        }
    }
    
    setLogLevel(level) {
        this.logLevel = level;
        this.updateDisplay();
    }
    
    clear() {
        this.logs = [];
        this.updateDisplay();
    }
}

// Initialize debug console in development
if (window.location.hostname === 'localhost' || 
    window.location.protocol === 'file:') {
    window.debugConsole = new DebugConsole();
}
```

---

## 🔗 NAVIGATION

**📖 Rozdziały:**
- **[⬅️ Chapter 5 - JavaScript Application Logic](./gatto-WORKING-04-photoshop-chapter5.md)**
- **[➡️ Chapter 7 - Deployment & Troubleshooting](./gatto-WORKING-04-photoshop-chapter7.md)**
- **[📋 Spis treści](./gatto-WORKING-04-photoshop-toc.md)**

---

*Ten rozdział opisuje kompletne testowanie integracji, włączając testy jednostkowe, testy integracyjne, procedury testowania manualnego oraz narzędzia debugowania.*
