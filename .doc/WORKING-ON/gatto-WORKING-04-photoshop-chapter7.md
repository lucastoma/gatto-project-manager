# Chapter 7: Deployment & Troubleshooting

**Previous:** [Chapter 6: Integration Testing](gatto-WORKING-04-photoshop-chapter6.md) | **TOC:** [Table of Contents](gatto-WORKING-04-photoshop-toc.md)

---

## 7.1 Deployment Checklist

### Pre-Deployment Validation
Before deploying the CEP extension, ensure all components are properly configured:

```bash
# Verify file structure
├── manifest.xml           # CEP extension configuration
├── index.html             # Panel UI (MainPath)
├── css/
│   └── styles.css         # Panel styling
├── js/
│   └── main.js           # Panel JavaScript logic
└── host/
    └── main.jsx          # ExtendScript functions (ScriptPath)
```

### Configuration Verification
1. **Manifest.xml Critical Settings:**
   ```xml
   <MainPath>./index.html</MainPath>        <!-- NOT ./js/main.js -->
   <ScriptPath>./host/main.jsx</ScriptPath> <!-- ExtendScript file -->
   <CEFCommandLine>
       <Parameter>--enable-nodejs</Parameter>
       <Parameter>--mixed-context</Parameter>
   </CEFCommandLine>
   ```

2. **API Endpoint Configuration:**
   ```javascript
   // In main.jsx - verify API endpoint
   const API_BASE_URL = "http://localhost:5000";
   const CURL_PATH = "curl"; // Ensure curl is in system PATH
   ```

---

## 7.2 Installation Procedures

### Development Installation
1. **Enable Debug Mode:**
   ```registry
   # Windows Registry (run as Administrator)
   [HKEY_CURRENT_USER\Software\Adobe\CSXS.11]
   "PlayerDebugMode"="1"
   
   [HKEY_CURRENT_USER\Software\Adobe\CSXS.10]
   "PlayerDebugMode"="1"
   ```

2. **Extension Placement:**
   ```bash
   # Copy extension to CEP extensions folder
   %APPDATA%\Adobe\CEP\extensions\GattoNeroPhotoshop\
   ```

3. **Restart Photoshop** and verify extension appears in Window > Extensions menu

### Production Deployment
1. **Create ZXP Package:**
   ```bash
   # Use Adobe ZXPSignCmd tool
   ZXPSignCmd -sign input_folder output.zxp certificate.p12 password -tsa timestamp_url
   ```

2. **Install via Adobe Extension Manager or CC Desktop App**

---

## 7.3 Common Issues & Solutions

### 7.3.1 Extension Not Loading

**Issue:** Extension doesn't appear in Photoshop Extensions menu

**Solutions:**
1. **Check Debug Mode:** Verify registry keys are set correctly
2. **Validate Manifest:** Ensure manifest.xml syntax is correct
3. **File Permissions:** Check extension folder permissions
4. **Version Compatibility:** Verify HostList matches Photoshop version

```xml
<!-- Correct HostList for multiple Photoshop versions -->
<HostList>
    <Host Name="PHXS" Version="[24.0,99.9]"/>
    <Host Name="PHSP" Version="[24.0,99.9]"/>
</HostList>
```

### 7.3.2 Panel Interface Issues

**Issue:** Panel loads but UI elements don't respond

**Solutions:**
1. **JavaScript Console:** Open Chrome DevTools (F12) in panel
2. **CSInterface Loading:** Verify CSInterface.js is loaded first
3. **Event Listeners:** Check if event listeners are properly attached

```javascript
// Debug UI initialization
window.addEventListener('load', function() {
    console.log('Panel loaded');
    if (typeof CSInterface !== 'undefined') {
        console.log('CSInterface available');
        initializeUI();
    } else {
        console.error('CSInterface not available');
    }
});
```

### 7.3.3 ExtendScript Communication Failures

**Issue:** Panel can't communicate with ExtendScript

**Solutions:**
1. **ScriptPath Verification:** Ensure manifest.xml points to correct .jsx file
2. **Function Existence:** Verify ExtendScript functions are properly defined
3. **Error Handling:** Add try-catch blocks in ExtendScript

```javascript
// Panel side - proper error handling
csInterface.evalScript('processColorMatching()', function(result) {
    if (result === 'EvalScript error.') {
        console.error('ExtendScript evaluation failed');
        updateStatus('ExtendScript communication error', 'error');
    } else {
        console.log('ExtendScript result:', result);
    }
});
```

### 7.3.4 API Connection Problems

**Issue:** Color matching API calls fail

**Solutions:**
1. **Server Status:** Verify Python API server is running
2. **Curl Availability:** Ensure curl is in system PATH
3. **Network Connectivity:** Test API endpoint manually

```jsx
// ExtendScript - test API connectivity
function testAPIConnection() {
    try {
        var curlCmd = 'curl -X GET "http://localhost:5000/health" -H "Content-Type: application/json"';
        var result = system.callSystem(curlCmd);
        $.writeln('API test result: ' + result);
        return result.indexOf('healthy') !== -1;
    } catch (e) {
        $.writeln('API connection error: ' + e.toString());
        return false;
    }
}
```

---

## 7.4 Debugging Tools & Techniques

### 7.4.1 Chrome DevTools for CEP Panel
```javascript
// Enable debugging in panel
if (typeof __adobe_cep__ !== 'undefined') {
    // Production mode
    console.log = function() {}; // Disable console in production
} else {
    // Development mode - enable full debugging
    window.addEventListener('keydown', function(e) {
        if (e.keyCode === 123) { // F12
            csInterface.openURLInDefaultBrowser('chrome://inspect');
        }
    });
}
```

### 7.4.2 ExtendScript Debugging
```jsx
// Enable ExtendScript debugging
$.level = 1; // Enable debugging
$.writeln('Debug message'); // Output to console

// File-based logging for complex debugging
function logToFile(message) {
    try {
        var logFile = new File(Folder.temp + '/gatto_debug.log');
        logFile.open('a');
        logFile.writeln(new Date().toISOString() + ': ' + message);
        logFile.close();
    } catch (e) {
        $.writeln('Logging error: ' + e.toString());
    }
}
```

### 7.4.3 API Server Debugging
```python
# Enable detailed logging in Flask app
import logging
logging.basicConfig(level=logging.DEBUG)

# Add request/response logging
@app.before_request
def log_request_info():
    app.logger.debug('Request: %s %s', request.method, request.url)
    app.logger.debug('Headers: %s', request.headers)

@app.after_request
def log_response_info(response):
    app.logger.debug('Response: %s', response.status_code)
    return response
```

---

## 7.5 Performance Optimization

### 7.5.1 Image Processing Optimization
```jsx
// Batch operations for better performance
function optimizedColorMatching(sourceLayer, targetLayer) {
    app.activeDocument.suspendHistory('Color Matching', 'performColorMatching');
    
    function performColorMatching() {
        // Disable screen updates during processing
        app.displayDialogs = DialogModes.NO;
        app.playbackDisplayDialogs = DialogModes.NO;
        
        try {
            // Perform color matching operations
            var result = processColorMatchingAPI(sourceLayer, targetLayer);
            return result;
        } finally {
            // Re-enable dialogs
            app.displayDialogs = DialogModes.ALL;
            app.playbackDisplayDialogs = DialogModes.ALL;
        }
    }
}
```

### 7.5.2 Memory Management
```jsx
// Proper cleanup after operations
function cleanupResources() {
    try {
        // Clear clipboard
        if (app.activeDocument.activeLayer) {
            app.activeDocument.activeLayer.copy();
        }
        
        // Force garbage collection
        app.purge(PurgeTarget.ALLCACHES);
        
        // Clear temporary files
        clearTempFiles();
        
    } catch (e) {
        $.writeln('Cleanup error: ' + e.toString());
    }
}
```

---

## 7.6 Error Recovery & Fallbacks

### 7.6.1 Automatic Recovery System
```javascript
// Panel side - implement retry logic
class ColorMatchingService {
    constructor() {
        this.maxRetries = 3;
        this.retryDelay = 1000; // 1 second
    }
    
    async processWithRetry(operation, params) {
        for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
            try {
                const result = await this.executeOperation(operation, params);
                return result;
            } catch (error) {
                console.warn(`Attempt ${attempt} failed:`, error);
                
                if (attempt === this.maxRetries) {
                    throw new Error(`Operation failed after ${this.maxRetries} attempts`);
                }
                
                await this.delay(this.retryDelay * attempt);
            }
        }
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
```

### 7.6.2 Graceful Degradation
```jsx
// ExtendScript - fallback for API failures
function colorMatchingWithFallback(sourceLayer, targetLayer) {
    var apiResult = null;
    
    try {
        // Attempt API-based color matching
        apiResult = callColorMatchingAPI(sourceLayer, targetLayer);
    } catch (apiError) {
        $.writeln('API failed, using fallback method: ' + apiError.toString());
        
        // Fallback to basic Photoshop color matching
        apiResult = basicColorMatching(sourceLayer, targetLayer);
    }
    
    return apiResult;
}

function basicColorMatching(sourceLayer, targetLayer) {
    // Simple fallback using Photoshop's built-in functions
    var originalLayer = app.activeDocument.activeLayer;
    
    try {
        app.activeDocument.activeLayer = targetLayer;
        
        // Apply basic color correction
        var colorBalance = app.activeDocument.activeLayer.adjustments.add();
        colorBalance.kind = AdjustmentReference.COLORBALANCE;
        
        return { success: true, method: 'fallback' };
        
    } catch (e) {
        return { success: false, error: e.toString() };
    } finally {
        app.activeDocument.activeLayer = originalLayer;
    }
}
```

---

## 7.7 Monitoring & Maintenance

### 7.7.1 Usage Analytics
```javascript
// Track extension usage for optimization
class AnalyticsTracker {
    constructor() {
        this.sessionStart = Date.now();
        this.operations = [];
    }
    
    trackOperation(operation, duration, success) {
        this.operations.push({
            operation: operation,
            duration: duration,
            success: success,
            timestamp: Date.now()
        });
        
        // Store locally for analysis
        localStorage.setItem('gatto_analytics', JSON.stringify(this.operations));
    }
    
    generateReport() {
        const report = {
            sessionDuration: Date.now() - this.sessionStart,
            totalOperations: this.operations.length,
            successRate: this.operations.filter(op => op.success).length / this.operations.length,
            averageDuration: this.operations.reduce((sum, op) => sum + op.duration, 0) / this.operations.length
        };
        
        console.log('Session Report:', report);
        return report;
    }
}
```

### 7.7.2 Health Checks
```javascript
// Regular health monitoring
class HealthMonitor {
    constructor() {
        this.checks = [];
        this.startMonitoring();
    }
    
    startMonitoring() {
        setInterval(() => {
            this.performHealthCheck();
        }, 30000); // Every 30 seconds
    }
    
    async performHealthCheck() {
        const checks = {
            panelResponsive: this.checkPanelResponsive(),
            apiConnection: await this.checkAPIConnection(),
            photoshopConnection: this.checkPhotoshopConnection(),
            memoryUsage: this.checkMemoryUsage()
        };
        
        this.checks.push({
            timestamp: Date.now(),
            checks: checks
        });
        
        // Alert if any critical issues
        if (!checks.panelResponsive || !checks.photoshopConnection) {
            this.alertCriticalIssue(checks);
        }
    }
    
    checkPanelResponsive() {
        try {
            document.getElementById('status').textContent = 'Health check';
            return true;
        } catch (e) {
            return false;
        }
    }
    
    alertCriticalIssue(checks) {
        updateStatus('Critical system issue detected', 'error');
        console.error('Health check failed:', checks);
    }
}
```

---

## 7.8 Version Management & Updates

### 7.8.1 Version Checking
```javascript
// Check for extension updates
class UpdateManager {
    constructor() {
        this.currentVersion = '1.0.0'; // From manifest.xml
        this.updateCheckInterval = 24 * 60 * 60 * 1000; // 24 hours
    }
    
    async checkForUpdates() {
        try {
            const response = await fetch('https://api.gattonero.com/version');
            const latestVersion = await response.json();
            
            if (this.isNewerVersion(latestVersion.version, this.currentVersion)) {
                this.notifyUpdate(latestVersion);
            }
        } catch (e) {
            console.warn('Update check failed:', e);
        }
    }
    
    isNewerVersion(latest, current) {
        const latestParts = latest.split('.').map(Number);
        const currentParts = current.split('.').map(Number);
        
        for (let i = 0; i < Math.max(latestParts.length, currentParts.length); i++) {
            const latestPart = latestParts[i] || 0;
            const currentPart = currentParts[i] || 0;
            
            if (latestPart > currentPart) return true;
            if (latestPart < currentPart) return false;
        }
        
        return false;
    }
    
    notifyUpdate(updateInfo) {
        updateStatus(`Update available: v${updateInfo.version}`, 'info');
        // Show update notification in panel
    }
}
```

---

## 7.9 Documentation & Support

### 7.9.1 User Documentation
Create comprehensive user guides covering:
- Installation instructions
- Basic usage tutorials
- Troubleshooting common issues
- Performance tips
- Feature explanations

### 7.9.2 Developer Documentation
Maintain technical documentation including:
- API documentation
- Code architecture overview
- Extension points for customization
- Testing procedures
- Deployment guidelines

### 7.9.3 Support Channels
Establish support mechanisms:
- GitHub Issues for bug reports
- Documentation wiki for FAQs
- Community forums for user discussions
- Direct support for enterprise users

---

## Summary

This chapter covered essential deployment and troubleshooting aspects:

1. **Deployment Checklist** - Pre-deployment validation and configuration verification
2. **Installation Procedures** - Development and production installation methods
3. **Common Issues & Solutions** - Troubleshooting extension loading, UI, communication, and API problems
4. **Debugging Tools** - Chrome DevTools, ExtendScript debugging, and API server debugging
5. **Performance Optimization** - Image processing optimization and memory management
6. **Error Recovery** - Automatic recovery systems and graceful degradation
7. **Monitoring & Maintenance** - Usage analytics and health checks
8. **Version Management** - Update checking and version control
9. **Documentation & Support** - User guides and support channels

**Key Takeaways:**
- ✅ Always verify manifest.xml configuration before deployment
- ✅ Implement comprehensive error handling and recovery mechanisms
- ✅ Use proper debugging tools for each component (Panel, ExtendScript, API)
- ✅ Monitor extension health and performance regularly
- ✅ Maintain up-to-date documentation for users and developers

**Next Steps:**
- Deploy extension to development environment
- Conduct thorough testing using procedures from Chapter 6
- Gather user feedback and iterate on improvements
- Plan production deployment and support infrastructure

---

**Navigation:** [Chapter 6: Integration Testing](gatto-WORKING-04-photoshop-chapter6.md) | **TOC:** [Table of Contents](gatto-WORKING-04-photoshop-toc.md)
