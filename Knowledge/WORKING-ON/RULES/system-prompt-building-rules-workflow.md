# GattoNero AI Assistant - System Prompt Rules (Building)
## Kompletny Zestaw Rules dla Pełnego Rozwoju Projektu

> **Przeznaczenie:** Nowe funkcjonalności, algorytmy, architektura  

---

## 🎯 FILOZOFIA BUILDING RULES

### Kiedy Używać
- Tworzenie nowego algorytmu od zera
- Implementacja nowej funkcjonalności
- Refaktoryzacja architektury
- Rozszerzanie API
- Dodawanie nowych JSX scripts
- Kompletny development workflow

### Zasady Podstawowe
- **Modularność:** CORE + separate algorithm modules
- **Kolokacja:** Dokumentacja przy kodzie
- **Standardy:** Naming conventions + file structure
- **Jakość:** Testing + verification wymagane

---

## 🏗️ ARCHITEKTURA PROJEKTU

### Struktura Główna
```
GattoNeroPhotoshop/
├── app/
│   ├── core/                    # Universal core functionality
│   │   ├── file_handler.py      # File operations
│   │   └── __init__.py
│   ├── api/                     # API endpoints and routes
│   │   ├── routes.py            # Flask routes
│   │   └── __init__.py
│   ├── algorithms/              # 🆕 NEW: Modular algorithms
│   │   ├── algorithm_01_palette/         # Palette matching
│   │   ├── algorithm_02_[name]color_matching/  # next algorithm
│   │   ├── (...))
│   │   ├── algorithm_XX_[another_name]
│   │   └── __init__.py
│   ├── scripts/                 # JSX scripts for Photoshop
│   │   ├── palette_analyzer.jsx # ✅ Working
│   │   ├── color_matcher.jsx    # ✅ Working  
│   │   ├── test_simple.jsx      # ✅ Working
│   │   └── template.jsx         # Template for new scripts
│   ├── temp_jsx/               # Temporary files for JSX
│   ├── processing.py           # 🔄 LEGACY: Migrate to algorithms/
│   ├── server.py              # Flask server main
│   └── utils.py               # Utility functions
├── doc/
│   └── WORKING-ON/
│       ├── RULES/             # System prompt rules (tu jesteśmy)
│       ├── gatto-WORKING-01-core.md ✅
│       └── gatto-WORKING-01-basic-photoshop-integration.md ✅
├── results/                   # Algorithm output files
├── source/                    # Test input files
└── uploads/                   # API upload staging
```

### Algorithm Module Structure
```
app/algorithms/algorithm_XX_name/
├── .implementation-todo       # 🔒 Hidden: TODO list for algorithm
├── .implementation-knowledge  # 🔒 Hidden: Technical knowledge base  
├── algorithm_main.py         # Main algorithm implementation
├── algorithm_functionname.py        # Utility functions for the algorithm
├── algorithm_utils.py        # Utility functions for the algorithm
├── algorithm_utils_what1.py  # Utility functions for the algorithm
├── algorithm_utils_what2.py  # Utility functions for the algorithm
├── __init__.py              # Module initialization
├── README.md                # Brief description and usage
├── tests/                   # Algorithm-specific tests
│   ├── test_algorithm.py    # Unit tests
│   └── test_data/           # Test input files
└── docs/                    # Extended documentation (optional)
    └── technical_details.md
```

---

## 📋 NAMING CONVENTIONS

### Algorithm Folders
**Pattern:** `algorithm_XX_name`
- `XX` = zero-padded number (01, 02, 03...)
- `name` = descriptive lowercase with underscores
- `name_function` = another file with code connected to name of function descriptive lowercase with underscores
- **Examples:**
  - `algorithm_01_palette` (palette color matching)
  - `algorithm_02_color_matching` (statistical/histogram methods)
  - `algorithm_03_lab_transfer` (LAB color space transfer)
  - `algorithm_04_delta_e` (Delta-E perceptual matching)

### Files Within Algorithm
**Python Files:**
- `algorithm_main.py` - główna implementacja
- `__init__.py` - module setup, exports
- `utils.py` - algorithm-specific utilities

**Documentation Files:**
- `.implementation-todo` - hidden TODO list
- `.implementation-knowledge` - hidden technical notes
- `README.md` - user-facing description

**Test Files:**
- `test_algorithm.py` - unit tests
- `test_integration.py` - integration tests

### JSX Scripts
**Pattern:** `{function}_{type}.jsx`
- **Examples:**
  - `palette_analyzer.jsx` (analiza palety)
  - `color_matcher.jsx` (color matching)
  - `batch_processor.jsx` (batch operations)

---

## 🎨 JSX DEVELOPMENT PATTERNS

### Standard JSX Structure
```jsx
#target photoshop

// Configuration
var SERVER_URL = "http://127.0.0.1:5000/api/{endpoint}";
var PROJECT_ROOT = new File($.fileName).parent.parent;
var TEMP_FOLDER = PROJECT_ROOT + "/temp_jsx/";

function main() {
    try {
        // 1. Validate Environment
        validatePhotoshopEnvironment();
        
        // 2. Setup Paths and Folders
        setupTempFolder();
        
        // 3. User Input/Configuration
        var config = showConfigurationDialog();
        
        // 4. Export Files
        var tempFiles = exportDocuments(config);
        
        // 5. HTTP Request to API
        var response = executeAPIRequest(tempFiles, config);
        
        // 6. Parse CSV Response
        var result = parseCSVResponse(response);
        
        // 7. Process Result in Photoshop
        processResult(result, config);
        
        // 8. User Feedback
        alert("SUCCESS: Operation completed");
        
    } catch (e) {
        alert("ERROR: " + e.message);
        // Log error for debugging
        logError(e);
    } finally {
        // 9. Cleanup
        cleanupTempFiles(tempFiles);
    }
}

main();
```

### CSV Protocol Standards
**API Response Format:**
```csv
# Success responses:
success,{data_specific_to_endpoint}

# Error responses:
error,{error_message}
```

**Specific Endpoints:**
```csv
# /api/analyze_palette
success,{color_count},{r,g,b,r,g,b,...}
# Example: success,3,255,128,64,100,200,50,75,175,225

# /api/colormatch  
success,method{X},{output_filename}
# Example: success,method1,test_result_123456789_matched.tif

# All endpoints errors:
error,{descriptive_error_message}
# Example: error,Invalid image format. Please use TIFF or PNG.
```

### File Management Pattern
```jsx
// Setup temp folder
function setupTempFolder() {
    var tempFolder = new Folder(TEMP_FOLDER);
    if (!tempFolder.exists) {
        tempFolder.create();
    }
    return tempFolder;
}

// Export with timestamp naming
function exportDocument(doc, folder, prefix) {
    var timestamp = Date.now();
    var fileName = prefix + "_" + timestamp + ".tif";
    var filePath = new File(folder + "/" + fileName);
    
    var tiffOptions = new TiffSaveOptions();
    tiffOptions.imageCompression = TIFFEncoding.NONE;
    tiffOptions.layers = false;
    
    doc.saveAs(filePath, tiffOptions, true, Extension.LOWERCASE);
    return filePath;
}

// Cleanup with error handling
function cleanupFile(file) {
    if (file && file.exists) {
        try {
            file.remove();
        } catch (e) {
            // Silent fail - cleanup is best effort
        }
    }
}
```

### HTTP Request Pattern (Windows)
```jsx
function executeAPIRequest(files, config) {
    var tempFolder = new Folder(TEMP_FOLDER);
    var cmdFile = new File(tempFolder + "/command.cmd");
    var stdoutFile = new File(tempFolder + "/output.txt");
    
    // Build curl command
    var curlCommand = buildCurlCommand(files, config);
    
    // Write batch file
    cmdFile.open("w");
    cmdFile.writeln("@echo off");
    cmdFile.writeln("chcp 65001 > nul");  // UTF-8 encoding
    cmdFile.writeln(curlCommand);
    cmdFile.close();
    
    // Execute with output capture
    app.system('cmd /c ""' + cmdFile.fsName + '" > "' + stdoutFile.fsName + '""');
    
    // Wait for response with timeout
    var response = waitForResponse(stdoutFile, 15000); // 15 second timeout
    
    // Cleanup command files
    cleanupFile(cmdFile);
    cleanupFile(stdoutFile);
    
    return response;
}
```

### Error Handling Best Practices
```jsx
// Comprehensive error handling
try {
    // Main operation
    var result = processImage();
    
    // Validate result
    if (!isValidResult(result)) {
        throw new Error("Invalid result from processing");
    }
    
    return result;
    
} catch (e) {
    // Enhanced error context
    var errorMsg = "Operation failed: " + e.message;
    if (e.line) errorMsg += " (Line: " + e.line + ")";
    
    // Log for debugging
    logError({
        message: e.message,
        line: e.line,
        source: e.source,
        timestamp: new Date().toString()
    });
    
    // User-friendly message
    alert(errorMsg);
    throw e; // Re-throw for caller handling
    
} finally {
    // Always cleanup, even on error
    cleanupTempFiles();
    restorePhotoshopState();
}
```

---

## 🔧 ALGORITHM DEVELOPMENT WORKFLOW

### 1. Planning Phase
**Stwórz folder algorytmu:**
```bash
mkdir app/algorithms/algorithm_XX_name
cd app/algorithms/algorithm_XX_name
```

**Stwórz podstawowe pliki:**
```bash
touch algorithm_main.py
touch __init__.py  
touch README.md
touch .implementation-todo
touch .implementation-knowledge
mkdir tests
mkdir docs
```

### 2. Documentation First
**`.implementation-todo` Template:**
```markdown
# Algorithm XX - TODO List

## Phase 1: Core Implementation
- [ ] Research and algorithm design
- [ ] Core function implementation
- [ ] Basic input/output handling
- [ ] Error handling

## Phase 2: Integration
- [ ] API endpoint integration
- [ ] JSX script development  
- [ ] CSV protocol implementation
- [ ] File management

## Phase 3: Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Edge case handling

## Phase 4: Documentation
- [ ] README.md completion
- [ ] Technical documentation
- [ ] Usage examples
- [ ] API documentation updates

## Notes
- Performance target: < 1 second for typical operations
- Memory limit: < 100MB for large images
- Supported formats: TIFF, PNG, JPG
```

**`.implementation-knowledge` Template:**
```markdown
# Algorithm XX - Technical Knowledge

## Algorithm Overview
- **Purpose:** {Brief description}
- **Input:** {Input requirements}
- **Output:** {Output format}
- **Complexity:** O({time/space complexity})

## Technical Details
### Core Algorithm
{Detailed technical explanation}

### Key Functions
- `main_process(input_data)` - {description}
- `validate_input(data)` - {description}
- `format_output(result)` - {description}

### Dependencies
- numpy: {specific usage}
- opencv: {specific usage}
- scikit-image: {specific usage}

### Performance Notes
- Typical processing time: {X} seconds
- Memory usage: {X} MB for {typical image size}
- Bottlenecks: {identified performance issues}

### Known Issues
- {Issue 1}: {description and workaround}
- {Issue 2}: {description and status}

### References
- {Academic papers}
- {Technical resources}
- {Implementation references}
```

### 3. Implementation Phase
**`algorithm_main.py` Structure:**
```python
"""
Algorithm XX: {Name}
{Brief description of algorithm purpose and functionality}
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

from ..core.file_handler import load_image, save_image
from ..utils import validate_input_image

logger = logging.getLogger(__name__)

class AlgorithmXX:
    """
    {Algorithm Name} implementation.
    
    {Detailed class description}
    """
    
    def __init__(self, **kwargs):
        """Initialize algorithm with configuration."""
        self.config = self._validate_config(kwargs)
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    def process(self, input_image: np.ndarray, **params) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main processing function.
        
        Args:
            input_image: Input image as numpy array
            **params: Algorithm-specific parameters
            
        Returns:
            Tuple of (processed_image, metadata)
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If processing fails
        """
        # 1. Validate input
        self._validate_input(input_image, params)
        
        # 2. Core processing
        try:
            result_image, metadata = self._core_algorithm(input_image, params)
        except Exception as e:
            logger.error(f"Core algorithm failed: {e}")
            raise RuntimeError(f"Processing failed: {e}")
        
        # 3. Validate output
        self._validate_output(result_image, metadata)
        
        # 4. Log performance metrics
        self._log_metrics(metadata)
        
        return result_image, metadata
    
    def _core_algorithm(self, image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
        """Core algorithm implementation."""
        # TODO: Implement specific algorithm logic
        pass
    
    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set default configuration."""
        defaults = {
            # Define default parameters
        }
        return {**defaults, **config}
    
    def _validate_input(self, image: np.ndarray, params: Dict) -> None:
        """Validate input parameters."""
        if not validate_input_image(image):
            raise ValueError("Invalid input image")
        # Additional validation logic
    
    def _validate_output(self, image: np.ndarray, metadata: Dict) -> None:
        """Validate output quality."""
        if image is None or image.size == 0:
            raise RuntimeError("Generated empty result")
        # Additional output validation

# API Integration Function
def process_algorithm_xx(input_path: str, output_path: str, **params) -> Dict[str, Any]:
    """
    API endpoint function for Algorithm XX.
    
    Args:
        input_path: Path to input image
        output_path: Path for output image  
        **params: Algorithm parameters
        
    Returns:
        Processing metadata and status
    """
    try:
        # Load image
        image = load_image(input_path)
        
        # Process
        algorithm = AlgorithmXX(**params)
        result_image, metadata = algorithm.process(image, **params)
        
        # Save result
        save_image(result_image, output_path)
        
        return {
            "status": "success",
            "output_path": output_path,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Algorithm XX processing failed: {e}")
        return {
            "status": "error", 
            "error": str(e)
        }
```

### 4. API Integration
**Update `app/api/routes.py`:**
```python
from ..algorithms.algorithm_XX_name.algorithm_main import process_algorithm_xx

@app.route('/api/algorithm_xx', methods=['POST'])
def api_algorithm_xx():
    """API endpoint for Algorithm XX."""
    try:
        # Handle file upload
        if 'image' not in request.files:
            return "error,No image file provided", 400
        
        file = request.files['image']
        if file.filename == '':
            return "error,Empty filename", 400
        
        # Save uploaded file
        input_path = save_uploaded_file(file)
        
        # Get parameters
        params = extract_request_params(request)
        
        # Process
        output_path = generate_output_path(input_path, "algorithm_xx")
        result = process_algorithm_xx(input_path, output_path, **params)
        
        if result["status"] == "success":
            filename = os.path.basename(result["output_path"])
            return f"success,algorithm_xx,{filename}", 200
        else:
            return f"error,{result['error']}", 500
            
    except Exception as e:
        logger.error(f"API algorithm_xx error: {e}")
        return f"error,Internal server error", 500
```

### 5. JSX Script Development
**Create `app/scripts/algorithm_xx.jsx`:**
```jsx
#target photoshop

var SERVER_URL = "http://127.0.0.1:5000/api/algorithm_xx";

function main() {
    try {
        // Standard JSX workflow using established patterns
        var config = showConfigurationDialog();
        var tempFiles = exportDocuments(config);
        var response = executeAPIRequest(tempFiles, config);
        var result = parseCSVResponse(response);
        processResult(result, config);
        
        alert("Algorithm XX completed successfully!");
        
    } catch (e) {
        alert("ERROR: " + e.message);
    } finally {
        cleanupTempFiles(tempFiles);
    }
}

function showConfigurationDialog() {
    // Algorithm-specific configuration UI
    // Return configuration object
}

main();
```

---

## 🧪 TESTING REQUIREMENTS

### Unit Tests Structure
```python
# tests/test_algorithm.py
import pytest
import numpy as np
from ..algorithm_main import AlgorithmXX, process_algorithm_xx

class TestAlgorithmXX:
    
    @pytest.fixture
    def sample_image(self):
        """Create sample test image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture  
    def algorithm(self):
        """Create algorithm instance."""
        return AlgorithmXX()
    
    def test_initialization(self):
        """Test algorithm initialization."""
        algo = AlgorithmXX()
        assert algo.config is not None
    
    def test_process_valid_input(self, algorithm, sample_image):
        """Test processing with valid input."""
        result_image, metadata = algorithm.process(sample_image)
        
        assert result_image is not None
        assert result_image.shape == sample_image.shape
        assert metadata is not None
        assert "processing_time" in metadata
    
    def test_process_invalid_input(self, algorithm):
        """Test processing with invalid input."""
        with pytest.raises(ValueError):
            algorithm.process(None)
    
    def test_api_integration(self, tmp_path):
        """Test API integration function."""
        # Create test files
        input_path = tmp_path / "input.tif"
        output_path = tmp_path / "output.tif"
        
        # Create sample image file
        sample_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        # Save sample_image to input_path
        
        # Test API function
        result = process_algorithm_xx(str(input_path), str(output_path))
        
        assert result["status"] == "success"
        assert output_path.exists()
```

### Performance Benchmarks
```python
# tests/test_performance.py
import time
import pytest
from ..algorithm_main import AlgorithmXX

class TestPerformance:
    
    @pytest.mark.performance
    def test_processing_speed(self):
        """Test algorithm processing speed."""
        algorithm = AlgorithmXX()
        
        # Test with different image sizes
        sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        for width, height in sizes:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            start_time = time.time()
            result_image, metadata = algorithm.process(image)
            processing_time = time.time() - start_time
            
            # Performance assertions
            assert processing_time < 5.0  # Max 5 seconds
            assert "processing_time" in metadata
            assert metadata["processing_time"] < processing_time * 1.1  # Within 10%
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage during processing."""
        # Memory profiling test
        # Use memory_profiler or similar
        pass
```

---

## 📚 DOCUMENTATION REQUIREMENTS

### README.md Template
```markdown
# Algorithm XX: {Name}

{Brief description of what this algorithm does}

## Overview
- **Purpose:** {Main goal of algorithm}
- **Input:** {Input requirements}
- **Output:** {Output description}
- **Performance:** ~{X} seconds for typical images

## Usage

### Python API
```python
from app.algorithms.algorithm_XX_name import AlgorithmXX

# Initialize
algorithm = AlgorithmXX(param1=value1, param2=value2)

# Process image
result_image, metadata = algorithm.process(input_image)
```

### HTTP API
```bash
curl -X POST http://localhost:5000/api/algorithm_xx \
  -F "image=@input.tif" \
  -F "param1=value1" \
  -F "param2=value2"
```

### Photoshop JSX
```jsx
// Run algorithm_xx.jsx script in Photoshop
// Configure parameters in dialog
// Process active document
```

## Parameters
- `param1` (type): Description. Default: value. Range: min-max.
- `param2` (type): Description. Default: value. Options: [option1, option2].

## Technical Details
{Brief technical explanation}

## Performance
- Typical processing time: {X} seconds
- Memory usage: {X} MB for 1024x1024 image
- Supported formats: TIFF, PNG, JPG

## Testing
```bash
pytest tests/test_algorithm.py -v
pytest tests/test_performance.py -v -m performance
```


