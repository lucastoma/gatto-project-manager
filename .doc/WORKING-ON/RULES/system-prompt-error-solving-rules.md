# GattoNero AI Assistant - System Prompt Rules (Error-Solving)
## Lżejszy Zestaw Rules dla Szybkich Poprawek i Bugfixów

> **Status:** ✅ ERROR-SOLVING RULES  
> **Ostatnia aktualizacja:** 2024-12-19  
> **Przeznaczenie:** Naprawki błędów, optymalizacje, małe zmiany  
> **Bazuje na:** Essential patterns z Building Rules

---

## 🎯 FILOZOFIA ERROR-SOLVING RULES

### Kiedy Używać
- Naprawianie błędów w istniejącym kodzie
- Debugowanie problemów funkcjonalności
- Małe optymalizacje performance
- Poprawki dokumentacji
- Quick fixes i hotfixes
- Troubleshooting JSX scripts

### Zasady Podstawowe
- **Szybkość:** Minimal viable fix
- **Bezpieczeństwo:** Don't break working code
- **Izolacja:** Fix specific issue without major changes
- **Weryfikacja:** Test fix before completion
- **Dokumentacja:** Update only affected docs

---

## 🏗️ ESSENTIAL ARCHITECTURE

### Current File Structure (Reference)
```
app/
├── core/                    # Core functionality
├── api/routes.py           # API endpoints
├── algorithms/             # 🆕 NEW: Modular algorithms
│   ├── algorithm_01_palette/
│   ├── algorithm_02_color_matching/
│   └── algorithm_XX_name/
├── scripts/                # JSX scripts
│   ├── palette_analyzer.jsx ✅
│   ├── color_matcher.jsx   ✅  
│   └── test_simple.jsx     ✅
├── processing/             # 🔄 LEGACY: Being migrated
└── server.py              # Flask server
```

### Algorithm Folder (Quick Reference)
```
algorithm_XX_name/
├── .implementation-todo       # TODO list
├── .implementation-knowledge  # Tech notes
├── algorithm_main.py         # Main code
├── README.md                # Description
└── tests/                   # Tests
```

---

## 🔧 QUICK FIX PATTERNS

### Error Handling Fix
```python
# Before: Risky code
def process_image(image_path):
    image = cv2.imread(image_path)
    result = some_algorithm(image)
    return result

# After: Safe error handling
def process_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        result = some_algorithm(image)
        if result is None:
            raise RuntimeError("Algorithm processing failed")
        
        return result
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise
```

### JSX Error Recovery
```jsx
// Quick fix pattern for JSX scripts
function safeExecute(operation, errorMessage) {
    try {
        return operation();
    } catch (e) {
        alert(errorMessage + ": " + e.message);
        return null;
    }
}

// Usage in existing scripts:
var result = safeExecute(
    function() { return processDocument(); },
    "Document processing failed"
);
if (result === null) return; // Exit gracefully
```

### API Error Response Fix
```python
# Quick fix for API endpoints
@app.route('/api/endpoint', methods=['POST'])
def api_endpoint():
    try:
        # Existing logic here
        result = process_request()
        return f"success,{result}", 200
        
    except ValueError as e:
        # Input validation error
        return f"error,Invalid input: {str(e)}", 400
    except FileNotFoundError as e:
        # File operation error
        return f"error,File not found: {str(e)}", 404
    except Exception as e:
        # General error
        logger.error(f"API error: {e}")
        return f"error,Internal server error", 500
```

---

## 🔍 COMMON ISSUES & FIXES

### Issue: JSX Script Hangs
**Symptoms:** Script doesn't respond, Photoshop freezes
**Quick Fix:**
```jsx
// Add timeout to HTTP requests
function waitForResponse(outputFile, timeout) {
    var startTime = Date.now();
    var maxWait = timeout || 15000; // 15 seconds default
    
    while (!outputFile.exists) {
        $.sleep(100); // 100ms polling
        if (Date.now() - startTime > maxWait) {
            throw new Error("Request timeout after " + (maxWait/1000) + " seconds");
        }
    }
    return true;
}
```

### Issue: CSV Parsing Errors
**Symptoms:** "error,..." responses not properly handled
**Quick Fix:**
```jsx
function parseCSVResponse(response) {
    if (!response || response.length === 0) {
        throw new Error("Empty response from server");
    }
    
    var parts = response.split(',');
    if (parts.length < 2) {
        throw new Error("Invalid response format: " + response);
    }
    
    var status = parts[0].trim();
    if (status === "error") {
        throw new Error(parts.slice(1).join(',').trim());
    }
    
    return {
        status: status,
        data: parts.slice(1)
    };
}
```

### Issue: Memory Errors with Large Images
**Symptoms:** Out of memory, slow processing
**Quick Fix:**
```python
def process_large_image(image, chunk_size=1024):
    """Process large images in chunks to avoid memory issues."""
    height, width = image.shape[:2]
    
    if height <= chunk_size and width <= chunk_size:
        # Small image, process normally
        return process_image_chunk(image)
    
    # Large image, process in chunks
    result = np.zeros_like(image)
    
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            y_end = min(y + chunk_size, height)
            x_end = min(x + chunk_size, width)
            
            chunk = image[y:y_end, x:x_end]
            result[y:y_end, x:x_end] = process_image_chunk(chunk)
    
    return result
```

### Issue: File Path Problems (Windows/macOS)
**Symptoms:** File not found, path errors
**Quick Fix:**
```jsx
// Cross-platform path handling
function normalizePath(path) {
    // Convert to platform-specific path
    if ($.os.indexOf("Windows") !== -1) {
        return path.replace(/\//g, "\\");
    } else {
        return path.replace(/\\/g, "/");
    }
}

function getProjectRoot() {
    var scriptFile = new File($.fileName);
    return scriptFile.parent.parent; // Go up 2 levels
}
```

---

## 🧪 QUICK TESTING

### Test Single Function
```python
# Quick test for specific function
def quick_test_function():
    try:
        # Test with sample data
        sample_input = create_test_data()
        result = your_function(sample_input)
        
        # Basic validation
        assert result is not None, "Result is None"
        assert len(result) > 0, "Empty result"
        
        print("✅ Function test passed")
        return True
        
    except Exception as e:
        print(f"❌ Function test failed: {e}")
        return False

# Run quick test
if __name__ == "__main__":
    quick_test_function()
```

### Test JSX Script
```jsx
// Add to beginning of JSX for quick testing
var QUICK_TEST = true; // Set to false for production

if (QUICK_TEST) {
    alert("Quick test mode enabled");
    // Test basic functionality without full workflow
    try {
        var testResult = testBasicFunctionality();
        alert("Quick test passed: " + testResult);
    } catch (e) {
        alert("Quick test failed: " + e.message);
    }
    return; // Exit before main workflow
}

// Normal script continues here...
```

### Test API Endpoint
```bash
# Quick curl test for API
curl -X POST http://localhost:5000/api/endpoint \
  -F "image=@test_image.tif" \
  -F "param=value" \
  -v

# Expected response: success,data
# Error response: error,message
```

---

## ⚡ PERFORMANCE QUICK FIXES

### Slow Image Processing
```python
# Quick optimization: Use appropriate data types
def optimize_image_processing(image):
    # Convert to optimal data type
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Use in-place operations when possible
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
    
    return image
```

### Slow File Operations
```python
# Quick fix: Batch file operations
import shutil

def cleanup_temp_files(temp_folder):
    """Quick cleanup of temp files."""
    try:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            os.makedirs(temp_folder)
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")
        # Continue anyway
```

### JSX Performance Issues
```jsx
// Quick fix: Reduce file operations
function optimizedExport(doc, folder) {
    // Check if export is really needed
    var timestamp = Date.now();
    var expectedFile = folder + "/temp_" + timestamp + ".tif";
    
    // Skip if recent file exists
    var existingFile = new File(expectedFile);
    if (existingFile.exists) {
        var fileAge = timestamp - existingFile.modified.getTime();
        if (fileAge < 5000) { // Less than 5 seconds old
            return existingFile; // Reuse existing file
        }
    }
    
    // Normal export
    return exportDocument(doc, folder);
}
```

---

## 🔄 LEGACY CODE COMPATIBILITY

### Working with Old Processing Module
```python
# Quick adapter for legacy code
from app.processing import color_matching as legacy_color_matching

def legacy_compatible_process(image_path, method='method1'):
    """Wrapper for legacy color matching while maintaining compatibility."""
    try:
        # Try new algorithm first
        if method in ['method1', 'method2', 'method3']:
            from app.algorithms.algorithm_02_color_matching import AlgorithmColorMatching
            algo = AlgorithmColorMatching()
            return algo.process_legacy(image_path, method)
    except ImportError:
        # Fall back to legacy
        return legacy_color_matching.process(image_path, method)
```

### Update Existing JSX Scripts
```jsx
// Quick fix: Add new error handling to existing scripts
// Add this block to existing color_matcher.jsx or palette_analyzer.jsx

// Enhanced error handling wrapper
var originalFunction = main; // Store original main function
function main() {
    try {
        return originalFunction.apply(this, arguments);
    } catch (e) {
        // Enhanced error reporting
        var errorDetails = {
            message: e.message,
            line: e.line || "unknown",
            source: e.source || "unknown",
            timestamp: new Date().toString()
        };
        
        // Log to file for debugging
        try {
            var logFile = new File(PROJECT_ROOT + "/temp_jsx/error_log.txt");
            logFile.open("a");
            logFile.writeln(JSON.stringify(errorDetails));
            logFile.close();
        } catch (logError) {
            // Silent fail on logging
        }
        
        // Show user-friendly error
        alert("Operation failed: " + e.message + "\nCheck error log for details.");
        throw e;
    }
}
```

---

## 📝 DOCUMENTATION QUICK UPDATES

### Update Algorithm README
```markdown
<!-- Quick addition to existing README.md -->

## Recent Changes
- 🐛 Fixed: {Brief description of fix}
- ⚡ Optimized: {Performance improvement}
- 📝 Updated: {Documentation update}

## Known Issues
- {Issue description}: {Status/Workaround}

## Quick Test
```bash
# Test this algorithm quickly:
python -c "from algorithm_main import AlgorithmXX; print('✅ Import OK')"
```
```

### Update .implementation-todo
```markdown
<!-- Add to existing .implementation-todo -->

## Recent Fixes - {Date}
- ✅ Fixed: {Issue description}
- ✅ Optimized: {Performance improvement}
- ⏳ In Progress: {Current work}

## Next Priority Fixes
- [ ] {High priority issue}
- [ ] {Performance bottleneck}
- [ ] {User-reported bug}
```

---

## 🎯 DEBUGGING SHORTCUTS

### Quick Debug JSX
```jsx
// Add to any JSX script for debugging
function debugLog(message, data) {
    var debugMode = true; // Set to false to disable
    if (!debugMode) return;
    
    var logMsg = "[DEBUG] " + message;
    if (data) logMsg += ": " + data.toString();
    
    alert(logMsg);
    
    // Also log to file
    try {
        var debugFile = new File(PROJECT_ROOT + "/temp_jsx/debug.log");
        debugFile.open("a");
        debugFile.writeln(new Date().toString() + " - " + logMsg);
        debugFile.close();
    } catch (e) {
        // Silent fail
    }
}

// Usage:
debugLog("Starting process", "Document count: " + app.documents.length);
```

### Quick Debug Python
```python
# Quick debugging decorator
def debug_function(func):
    def wrapper(*args, **kwargs):
        print(f"🔍 Calling {func.__name__} with args: {args[:2]}...")  # Limit output
        try:
            result = func(*args, **kwargs)
            print(f"✅ {func.__name__} completed successfully")
            return result
        except Exception as e:
            print(f"❌ {func.__name__} failed: {e}")
            raise
    return wrapper

# Usage: Add @debug_function above problematic functions
@debug_function
def problematic_function(image):
    # Your code here
    pass
```

### Quick Server Debug
```python
# Add to server.py for request debugging
@app.before_request
def log_request_info():
    if app.debug:  # Only in debug mode
        print(f"🌐 {request.method} {request.endpoint}")
        if request.files:
            print(f"📁 Files: {list(request.files.keys())}")
        if request.form:
            print(f"📝 Form: {dict(request.form)}")
```

---

## ⚠️ SAFETY CHECKLIST

### Before Making Changes
- [ ] Backup current working files
- [ ] Identify exact scope of change
- [ ] Test change in isolation if possible
- [ ] Have rollback plan ready

### After Making Changes  
- [ ] Test the specific fix works
- [ ] Verify no regression in working features
- [ ] Update relevant documentation
- [ ] Commit changes with clear message

### Emergency Rollback
```bash
# Quick rollback commands
git stash                    # Stash current changes
git checkout HEAD~1 -- file  # Restore single file
git reset --hard HEAD~1      # Full rollback (careful!)
```

---

*Lżejszy zestaw Error-Solving Rules dla szybkich poprawek GattoNero AI Assistant z zachowaniem bezpieczeństwa i efektywności.*
