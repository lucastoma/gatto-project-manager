# Template: Algorithm Folder Structure
<!-- This is a template showing the complete structure for a new algorithm -->

## Algorithm {XX} - {Name}

> **Template Purpose:** Complete folder structure for new algorithms  
> **Usage:** Copy this structure when creating new algorithm  
> **Pattern:** `algorithm_XX_name/`

---

## 📁 COMPLETE FOLDER STRUCTURE

```
algorithm_XX_name/
├── .implementation-todo           # Hidden TODO list (copy from template)
├── .implementation-knowledge      # Hidden knowledge base (copy from template)
├── algorithm_main.py             # Main algorithm implementation
├── __init__.py                   # Module initialization and exports
├── README.md                     # User-facing documentation
├── tests/                        # Algorithm-specific tests
│   ├── __init__.py              
│   ├── test_algorithm.py         # Unit tests
│   ├── test_integration.py       # Integration tests
│   ├── test_performance.py       # Performance benchmarks
│   └── test_data/               # Test input files
│       ├── sample_input.tif
│       ├── sample_output.tif
│       └── edge_cases/
├── docs/                         # Extended documentation (optional)
│   ├── technical_details.md     # Deep technical documentation
│   ├── usage_examples.md        # Detailed usage examples
│   └── images/                  # Documentation images
└── utils/                       # Algorithm-specific utilities (optional)
    ├── __init__.py
    └── helpers.py               # Helper functions
```

---

## 📝 FILE CONTENTS TEMPLATES

### 1. `.implementation-todo`
```markdown
<!-- Copy from template-examples/implementation-todo-template.md -->
<!-- Replace {XX} and {Name} with actual values -->
```

### 2. `.implementation-knowledge`
```markdown
<!-- Copy from template-examples/implementation-knowledge-template.md -->
<!-- Replace {XX} and {Name} with actual values -->
```

### 3. `algorithm_main.py`
```python
"""
Algorithm {XX}: {Name}
{Brief description of algorithm purpose and functionality}

This module implements {detailed description} for the GattoNero AI Assistant.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path

# Project imports
from ..core.file_handler import load_image, save_image
from ..utils import validate_input_image

logger = logging.getLogger(__name__)

class Algorithm{XX}:
    """
    {Algorithm Name} implementation.
    
    This class implements {detailed description of algorithm}.
    
    Attributes:
        config (Dict): Algorithm configuration parameters
        
    Example:
        >>> algorithm = Algorithm{XX}(param1=value1)
        >>> result_image, metadata = algorithm.process(input_image)
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'param1': {
            'default': 'default_value',
            'type': str,
            'description': 'Parameter description'
        },
        # Add more parameters as needed
    }
    
    def __init__(self, **kwargs):
        """
        Initialize Algorithm{XX} with configuration.
        
        Args:
            **kwargs: Configuration parameters (see DEFAULT_CONFIG)
        """
        self.config = self._validate_config(kwargs)
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    def process(self, input_image: np.ndarray, **params) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main processing function for Algorithm{XX}.
        
        Args:
            input_image (np.ndarray): Input image as numpy array (H, W, C)
            **params: Additional processing parameters
            
        Returns:
            Tuple[np.ndarray, Dict]: (processed_image, metadata)
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If processing fails
        """
        start_time = time.time()
        
        # 1. Validate input
        self._validate_input(input_image, params)
        
        # 2. Core processing
        try:
            result_image, metadata = self._core_algorithm(input_image, params)
        except Exception as e:
            logger.error(f"Core algorithm failed: {e}")
            raise RuntimeError(f"Algorithm{XX} processing failed: {e}")
        
        # 3. Validate output
        self._validate_output(result_image, metadata)
        
        # 4. Add processing metadata
        processing_time = time.time() - start_time
        metadata.update({
            'algorithm': f'algorithm_{XX:02d}',
            'processing_time': processing_time,
            'input_shape': input_image.shape,
            'output_shape': result_image.shape
        })
        
        logger.info(f"Algorithm{XX} completed in {processing_time:.3f}s")
        return result_image, metadata
    
    def _core_algorithm(self, image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Core algorithm implementation.
        
        This method contains the main algorithm logic.
        
        Args:
            image (np.ndarray): Input image
            params (Dict): Processing parameters
            
        Returns:
            Tuple[np.ndarray, Dict]: (result_image, algorithm_metadata)
        """
        # TODO: Implement specific algorithm logic
        # This is where the main algorithm implementation goes
        
        # Placeholder implementation
        result_image = image.copy()
        metadata = {
            'method': 'placeholder',
            'parameters': params
        }
        
        return result_image, metadata
    
    def _validate_config(self, config: Dict) -> Dict:
        """
        Validate and merge configuration with defaults.
        
        Args:
            config (Dict): User-provided configuration
            
        Returns:
            Dict: Validated configuration
        """
        validated_config = {}
        
        for key, default_info in self.DEFAULT_CONFIG.items():
            if key in config:
                # Use provided value
                value = config[key]
                # TODO: Add type validation if needed
                validated_config[key] = value
            else:
                # Use default value
                validated_config[key] = default_info['default']
        
        return validated_config
    
    def _validate_input(self, image: np.ndarray, params: Dict) -> None:
        """
        Validate input parameters.
        
        Args:
            image (np.ndarray): Input image to validate
            params (Dict): Parameters to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not validate_input_image(image):
            raise ValueError("Invalid input image")
        
        if image.ndim not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")
        
        if image.dtype != np.uint8:
            logger.warning(f"Input image dtype is {image.dtype}, expected uint8")
        
        # TODO: Add algorithm-specific validation
    
    def _validate_output(self, image: np.ndarray, metadata: Dict) -> None:
        """
        Validate output quality and consistency.
        
        Args:
            image (np.ndarray): Output image to validate
            metadata (Dict): Algorithm metadata to validate
            
        Raises:
            RuntimeError: If output validation fails
        """
        if image is None or image.size == 0:
            raise RuntimeError("Generated empty result image")
        
        if not isinstance(metadata, dict):
            raise RuntimeError("Metadata must be a dictionary")
        
        # TODO: Add algorithm-specific output validation


# API Integration Function
def process_algorithm_{xx}(input_path: str, output_path: str, **params) -> Dict[str, Any]:
    """
    API endpoint function for Algorithm{XX}.
    
    This function provides the interface between the Flask API and the algorithm.
    
    Args:
        input_path (str): Path to input image file
        output_path (str): Path for output image file
        **params: Algorithm parameters from API request
        
    Returns:
        Dict[str, Any]: Processing result with status and metadata
    """
    try:
        # Load image
        logger.info(f"Loading image from {input_path}")
        image = load_image(input_path)
        
        # Initialize and run algorithm
        algorithm = Algorithm{XX}(**params)
        result_image, metadata = algorithm.process(image, **params)
        
        # Save result
        logger.info(f"Saving result to {output_path}")
        save_image(result_image, output_path)
        
        return {
            "status": "success",
            "algorithm": f"algorithm_{XX:02d}",
            "output_path": output_path,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Algorithm{XX} API processing failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "algorithm": f"algorithm_{XX:02d}"
        }


# Convenience functions for common use cases
def quick_process(image: np.ndarray, **params) -> np.ndarray:
    """
    Quick processing function that returns only the result image.
    
    Args:
        image (np.ndarray): Input image
        **params: Algorithm parameters
        
    Returns:
        np.ndarray: Processed image
    """
    algorithm = Algorithm{XX}(**params)
    result_image, _ = algorithm.process(image, **params)
    return result_image


if __name__ == "__main__":
    # Quick test of the algorithm
    import sys
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output.tif"
        
        result = process_algorithm_{xx}(input_path, output_path)
        print(f"Result: {result}")
    else:
        print("Usage: python algorithm_main.py input_image.tif [output_image.tif]")
```

### 4. `__init__.py`
```python
"""
Algorithm {XX}: {Name}
"""

from .algorithm_main import Algorithm{XX}, process_algorithm_{xx}, quick_process

__all__ = [
    'Algorithm{XX}',
    'process_algorithm_{xx}',
    'quick_process'
]

__version__ = '1.0.0'
__algorithm_id__ = {XX}
__algorithm_name__ = '{name}'
```

### 5. `README.md`
```markdown
# Algorithm {XX}: {Name}

{Brief description of what this algorithm does and its main benefits}

## Overview

- **Purpose:** {Main goal of algorithm}
- **Input:** {Input requirements - image format, size constraints}
- **Output:** {Output description - what the algorithm produces}
- **Performance:** ~{X} seconds for typical 1024x1024 images
- **Quality:** {Quality metrics or characteristics}

## Quick Start

### Python API
```python
from app.algorithms.algorithm_{xx}_{name} import Algorithm{XX}

# Basic usage
algorithm = Algorithm{XX}()
result_image, metadata = algorithm.process(input_image)

# With parameters
algorithm = Algorithm{XX}(param1=value1, param2=value2)
result_image, metadata = algorithm.process(input_image)
```

### HTTP API
```bash
curl -X POST http://localhost:5000/api/algorithm_{xx} \
  -F "image=@input.tif" \
  -F "param1=value1" \
  -F "param2=value2"

# Response: success,algorithm_{xx},output_filename.tif
```

### Photoshop JSX
1. Open Photoshop with your source document
2. Run the JSX script: `app/scripts/algorithm_{xx}.jsx`
3. Configure parameters in the dialog
4. Click OK to process

## Parameters

### Required Parameters
- **input_image**: Input image as numpy array or file path

### Optional Parameters
- **param1** (`{type}`, default: `{default}`): {Description of parameter}
  - Range: {min} to {max}
  - Example: `param1=50`
- **param2** (`{type}`, default: `{default}`): {Description of parameter}
  - Options: {option1, option2, option3}
  - Example: `param2="option1"`

## Examples

### Basic Example
```python
import cv2
from app.algorithms.algorithm_{xx}_{name} import quick_process

# Load image
image = cv2.imread('input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process
result = quick_process(image)

# Save
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite('output.jpg', result_bgr)
```

### Advanced Example
```python
from app.algorithms.algorithm_{xx}_{name} import Algorithm{XX}

# Custom configuration
algorithm = Algorithm{XX}(
    param1=custom_value,
    param2=custom_option
)

# Process with additional parameters
result_image, metadata = algorithm.process(
    input_image,
    extra_param=extra_value
)

print(f"Processing time: {metadata['processing_time']:.3f}s")
print(f"Algorithm details: {metadata}")
```

## Technical Details

### Algorithm Overview
{Brief technical explanation of how the algorithm works}

### Performance Characteristics
- **Time Complexity:** O({complexity})
- **Space Complexity:** O({complexity})
- **Typical Processing Time:** {X} seconds for 1024x1024 images
- **Memory Usage:** ~{X} MB for typical operations

### Supported Image Formats
- **Input:** TIFF, PNG, JPEG (RGB, BGR, Grayscale)
- **Output:** Same format as input
- **Color Depth:** 8-bit per channel (uint8)
- **Size Limits:** {minimum} to {maximum} pixels

## Testing

### Run Tests
```bash
# All tests
pytest app/algorithms/algorithm_{xx}_{name}/tests/ -v

# Specific test categories
pytest app/algorithms/algorithm_{xx}_{name}/tests/test_algorithm.py -v
pytest app/algorithms/algorithm_{xx}_{name}/tests/test_performance.py -v
```

### Performance Benchmark
```bash
python -m app.algorithms.algorithm_{xx}_{name}.tests.test_performance
```

## Known Issues & Limitations

- **Issue 1:** {Description} - {Status/Workaround}
- **Issue 2:** {Description} - {Status/Workaround}

### Future Improvements
- {Planned enhancement 1}
- {Planned enhancement 2}

## API Reference

### Class: Algorithm{XX}
Main algorithm implementation class.

#### Methods
- `__init__(**kwargs)`: Initialize with configuration
- `process(image, **params)`: Main processing method
- `_core_algorithm(image, params)`: Core implementation (private)

### Function: process_algorithm_{xx}(input_path, output_path, **params)
API integration function for HTTP endpoints.

### Function: quick_process(image, **params)
Convenience function for simple processing.

## Dependencies

### Required
- numpy >= 1.19.0
- opencv-python >= 4.5.0
- {additional dependencies}

### Optional
- scikit-image >= 0.18.0 (for advanced features)
- {optional dependencies}

## Version History

- **v1.0.0**: Initial implementation
- **v1.1.0**: {Future version notes}

---

*This algorithm is part of the GattoNero AI Assistant project for advanced image processing in Photoshop integration.*
```

### 6. `tests/__init__.py`
```python
"""
Tests for Algorithm {XX}: {Name}
"""
```

### 7. `tests/test_algorithm.py`
```python
"""
Unit tests for Algorithm {XX}
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# Import algorithm components
from ..algorithm_main import Algorithm{XX}, process_algorithm_{xx}, quick_process


class TestAlgorithm{XX}:
    """Test suite for Algorithm{XX} class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def algorithm(self):
        """Create algorithm instance with default config."""
        return Algorithm{XX}()
    
    @pytest.fixture
    def algorithm_custom(self):
        """Create algorithm instance with custom config."""
        return Algorithm{XX}(param1="custom_value")
    
    def test_initialization_default(self):
        """Test algorithm initialization with default parameters."""
        algo = Algorithm{XX}()
        assert algo.config is not None
        assert isinstance(algo.config, dict)
    
    def test_initialization_custom(self):
        """Test algorithm initialization with custom parameters."""
        custom_params = {'param1': 'test_value'}
        algo = Algorithm{XX}(**custom_params)
        assert algo.config['param1'] == 'test_value'
    
    def test_process_valid_input(self, algorithm, sample_image):
        """Test processing with valid input."""
        result_image, metadata = algorithm.process(sample_image)
        
        # Check result image
        assert result_image is not None
        assert isinstance(result_image, np.ndarray)
        assert result_image.shape == sample_image.shape
        assert result_image.dtype == sample_image.dtype
        
        # Check metadata
        assert isinstance(metadata, dict)
        assert 'processing_time' in metadata
        assert 'algorithm' in metadata
        assert metadata['algorithm'] == f'algorithm_{XX:02d}'
    
    def test_process_grayscale_image(self, algorithm):
        """Test processing with grayscale image."""
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result_image, metadata = algorithm.process(gray_image)
        
        assert result_image is not None
        assert result_image.shape == gray_image.shape
    
    def test_process_invalid_input_none(self, algorithm):
        """Test processing with None input."""
        with pytest.raises(ValueError):
            algorithm.process(None)
    
    def test_process_invalid_input_empty(self, algorithm):
        """Test processing with empty array."""
        empty_image = np.array([])
        with pytest.raises(ValueError):
            algorithm.process(empty_image)
    
    def test_process_invalid_input_wrong_dimensions(self, algorithm):
        """Test processing with wrong dimensions."""
        wrong_dim_image = np.random.randint(0, 255, (100, 100, 100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            algorithm.process(wrong_dim_image)
    
    def test_process_with_parameters(self, algorithm, sample_image):
        """Test processing with additional parameters."""
        params = {'extra_param': 'test_value'}
        result_image, metadata = algorithm.process(sample_image, **params)
        
        assert result_image is not None
        assert isinstance(metadata, dict)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        algo = Algorithm{XX}(param1="valid")
        assert algo.config['param1'] == "valid"
        
        # Invalid config should use defaults
        algo = Algorithm{XX}()
        assert 'param1' in algo.config


class TestAPIIntegration:
    """Test API integration functions."""
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary input and output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.tif"
            output_path = temp_path / "output.tif"
            
            # Create sample input file
            sample_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            # TODO: Save sample_image to input_path using appropriate method
            
            yield str(input_path), str(output_path)
    
    def test_process_algorithm_api_success(self, temp_files):
        """Test API processing function with valid inputs."""
        input_path, output_path = temp_files
        
        result = process_algorithm_{xx}(input_path, output_path)
        
        assert result['status'] == 'success'
        assert result['algorithm'] == f'algorithm_{XX:02d}'
        assert 'output_path' in result
        assert 'metadata' in result
    
    def test_process_algorithm_api_invalid_input(self):
        """Test API processing with invalid input path."""
        result = process_algorithm_{xx}("nonexistent.tif", "output.tif")
        
        assert result['status'] == 'error'
        assert 'error' in result
    
    def test_quick_process_function(self):
        """Test quick processing convenience function."""
        sample_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        result = quick_process(sample_image)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 🚀 USAGE INSTRUCTIONS

### 1. Creating New Algorithm
```bash
# 1. Create algorithm folder
mkdir app/algorithms/algorithm_XX_name

# 2. Copy template files
cp -r doc/WORKING-ON/RULES/template-examples/algorithm-folder-template/* app/algorithms/algorithm_XX_name/

# 3. Replace placeholders
# Edit all files and replace {XX}, {Name}, etc. with actual values

# 4. Implement algorithm logic
# Edit algorithm_main.py and implement _core_algorithm method

# 5. Write tests
# Edit tests/test_algorithm.py and add specific tests

# 6. Update API routes
# Add new endpoint to app/api/routes.py

# 7. Create JSX script
# Create app/scripts/algorithm_XX.jsx
```

### 2. Template Replacement Guide
Replace these placeholders in all template files:

- `{XX}` → Algorithm number (e.g., `01`, `02`, `03`)
- `{Name}` → Algorithm name (e.g., `Palette Matching`)
- `{name}` → Algorithm name lowercase with underscores (e.g., `palette_matching`)
- `{xx}` → Algorithm number lowercase (e.g., `01`, `02`, `03`)
- `{Date}` → Current date (e.g., `2024-12-19`)
- `{time_complexity}` → Big O time complexity (e.g., `n*m`)
- `{space_complexity}` → Big O space complexity (e.g., `n`)

### 3. Validation Checklist
After creating new algorithm from template:

- [ ] All placeholder values replaced
- [ ] Algorithm folder follows naming convention
- [ ] `.implementation-*` files are hidden (start with dot)
- [ ] `algorithm_main.py` compiles without errors
- [ ] Tests can be imported (even if not implemented)
- [ ] README.md is readable and informative
- [ ] API integration function exists
- [ ] Proper import structure in `__init__.py`

---

*This template provides the complete structure for implementing new algorithms in the GattoNero AI Assistant following established patterns and best practices.*
