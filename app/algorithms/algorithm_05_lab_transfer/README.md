---
version: "1.0"
last_updated: 2025-06-14
author: lucastoma & Cascade
interface_stable: true
stop_deep_scan: false
tags:
  - api
  - module
  - interface
  - image_processing
  - color_transfer
  - lab_color
  - gpu_acceleration
aliases:
  - "LAB Color Transfer"
  - "algorithm_05_lab_transfer"
links:
  - "[[README.concepts]]"
  - "[[README.todo]]"
cssclasses:
  - readme-template
---

# algorithm_05_lab_transfer

Performs color transfer between images using the LAB color space. It supports multiple transfer algorithms and features GPU acceleration via OpenCL for enhanced performance, with a fallback to CPU for broader compatibility.

## 1. Overview & Quick Start

### Co to jest
This module is responsible for transferring the color characteristics from a source image to a target image. It operates in the LAB color space, which is known for its perceptual uniformity, allowing for more natural-looking color transformations. The module is a key component of the GattoNeroPhotoshop application's image processing pipeline and can be used for tasks like style transfer or color correction. It includes GPU-accelerated versions of its algorithms for speed, and CPU fallbacks if a GPU is not available or `pyopencl` is not installed.

### Szybki start
```bash
# Installation (ensure Python 3.8+ is installed)
# Core dependencies:
pip install numpy Pillow scikit-image

# Optional for GPU acceleration:
pip install pyopencl
# Note: pyopencl requires appropriate OpenCL drivers for your GPU.
# If pyopencl is not installed or fails to initialize, the module will automatically use CPU implementations.

# Ensure test images are generated (if running tests or examples that need them):
# python app/algorithms/algorithm_05_lab_transfer/tests/regenerate_test_images.py
```

### Struktura katalogu

## 2. Implementation Status & Gaps

This table provides a high-level overview of the current implementation status for each processing mode across CPU and GPU platforms. It highlights existing gaps between the expected functionality and the actual code.

| Mode / Feature | Expected Functionality (from UI/Config) | CPU Status (`core.py`) | GPU Status (`gpu_core.py` + `kernels.cl`) | Key Actions Required |
| :--- | :--- | :--- | :--- | :--- |
| **Basic** | Standard statistical transfer of L, a, b channels. | ✅ **Complete** | ✅ **Complete** | None. |
| **Linear Blend** | Weighted transfer using `weights` for L, a, b. | ✅ **Complete** | ✅ **Complete** | None. |
| **Selective** | Transfer on `selective_channels` with a `mask` and `blend_factor`. | ✅ **Complete** | ❌ **Partial**. Kernel only supports a simple 'preserve L' flag. No support for mask, blend_factor, or dynamic channels. | **Critical**: Enhance `gpu_core.py` and kernel to support mask, blending, and channel selection. |
| **Adaptive** | Segmentation using `num_segments`, `delta_e_threshold`, `min_segment_size_perc`. | ✅ **Complete** | ❌ **Partial/Incorrect**. Kernel uses a hardcoded 3-segment logic and ignores configuration parameters. | **Critical**: Modify GPU implementation to dynamically handle `num_segments` and other parameters. |
| **Hybrid** | Logic is not clearly defined in the UI. `advanced.py` suggests it's a wrapper. | ❌ **Not Integrated**. `process_image_hybrid` exists but is not connected to the main `METHOD_MAP`. | ❌ **Not Implemented**. | **High**: Integrate into `METHOD_MAP` for testing. Define its role and decide on a potential GPU implementation. |

```
/app/algorithms/algorithm_05_lab_transfer/
├── core.py                 # Main LABColorTransfer class, CPU implementations
├── gpu_core.py             # LABColorTransferGPU class, OpenCL integration
├── kernels.cl              # OpenCL kernel code for GPU computations
├── config.py               # Configuration management for algorithm parameters
├── processor.py            # ImageProcessor class for single/batch processing
├── logger.py               # Logging setup
├── exceptions.py           # Custom exceptions
├── utils.py                # Utility functions
├── tests/                  # Unit and integration tests
│   ├── test_lab_transfer_basic.py
│   ├── test_lab_transfer_adaptive.py
│   ├── ... (other test files)
│   ├── test_images/          # Test images (.npy format)
│   └── regenerate_test_images.py # Script to generate .npy test images
├── __init__.py
├── README.md               # This file
├── README.concepts.md      # Conceptual details and design decisions
└── README.todo.md          # Roadmap and pending tasks
```

### Wymagania
- Python 3.8+
- Core Libraries:
  - `numpy`
  - `Pillow`
  - `scikit-image`
- Optional for GPU Acceleration:
  - `pyopencl`
  - Working OpenCL drivers for your GPU (NVIDIA, AMD, Intel)
- For testing:
  - `pytest`

### Najczęstsze problemy
- **`ModuleNotFoundError: No module named 'pyopencl'`**: `pyopencl` is not installed. Install it (`pip install pyopencl`) if you want GPU acceleration. Otherwise, the module will automatically use CPU fallback (if `use_gpu` was True but OpenCL failed to init) or run on CPU (if `use_gpu` was False).
- **OpenCL errors (e.g., `clBuildProgram` failure, `LogicError: clEnqueueNDRangeKernel failed: MEM_OBJECT_ALLOCATION_FAILURE`):**
    - Ensure your GPU drivers are up-to-date and support OpenCL.
    - The GPU might not have enough memory for the operation, especially with very large images. Try smaller images or ensure other GPU-intensive applications are closed.
    - Kernel compilation errors might indicate issues with the OpenCL code or driver compatibility.
- **`FileNotFoundError` for test images**: Run `regenerate_test_images.py` located in the `tests/` directory to create the necessary `.npy` files.
- **Incorrect color results or artifacts**:
    - Check input image formats (should be RGB).
    - Experiment with different transfer methods (`basic`, `adaptive`, `selective`, `weighted`) and their parameters in `config.py`.
    - Ensure source and target images are suitable for color transfer (e.g., not too dissimilar in content if using adaptive methods).
- **Type errors related to configuration**: Ensure parameters in `config.py` or passed dynamically are of the correct numeric types (float/int). The module attempts to cast them, but direct configuration should be correct.

---

## 2. API Documentation

### Klasy dostępne

#### `LABColorTransfer`
**Przeznaczenie:** Main class for performing LAB color transfer. Orchestrates CPU and GPU implementations.
Located in: `app/algorithms/algorithm_05_lab_transfer/core.py`

##### Konstruktor
```python
LABColorTransfer(config: Optional[dict] = None, use_gpu: bool = True)
```
**Parametry:**
- `config` (dict, optional): A dictionary containing configuration parameters. If `None`, default configuration from `config.py` is used. See `config.py` for available parameters (e.g., `adaptive_segment_threshold`, `selective_hue_weight`).
- `use_gpu` (bool, optional, default=True): If `True`, attempts to use GPU acceleration. Falls back to CPU if GPU initialization fails or `pyopencl` is not available. If `False`, forces CPU execution.

##### Główne metody

**`basic_transfer(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray`**
- **Description:** Performs a global color transfer by matching the mean and standard deviation of LAB channels from source to target.
- **Input:**
    - `source_rgb` (np.ndarray): Source image in RGB format (H x W x 3, uint8).
    - `target_rgb` (np.ndarray): Target image in RGB format (H x W x 3, uint8).
- **Output:** `np.ndarray`: Resulting image in RGB format (H x W x 3, uint8).

**`adaptive_transfer(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray`**
- **Description:** Performs adaptive color transfer using image segmentation and local statistics. Can be computationally intensive. GPU acceleration is hybrid.
- **Input:**
    - `source_rgb` (np.ndarray): Source image in RGB format.
    - `target_rgb` (np.ndarray): Target image in RGB format.
- **Output:** `np.ndarray`: Resulting image in RGB format.
- **Configurable via:** `adaptive_num_segments`, `adaptive_segment_threshold`, etc. in `config.py`.

**`selective_transfer(source_rgb: np.ndarray, target_rgb: np.ndarray, mask_rgb: Optional[np.ndarray] = None) -> np.ndarray`**
- **Description:** Transfers color selectively, potentially guided by a mask. If no mask is provided, it behaves like basic transfer but might use different internal weighting (see `kernels.cl` if GPU).
- **Input:**
    - `source_rgb` (np.ndarray): Source image in RGB format.
    - `target_rgb` (np.ndarray): Target image in RGB format.
    - `mask_rgb` (np.ndarray, optional): A binary or grayscale mask (H x W or H x W x 1, uint8) indicating regions for transfer. Currently, the direct use of a mask to *limit* transfer to specific areas in the GPU kernel might require custom kernel modification or CPU-side pre/post-processing. The `selective_` prefix in GPU implies specific weighting strategies.
- **Output:** `np.ndarray`: Resulting image in RGB format.
- **Configurable via:** `selective_color_weight`, `selective_spatial_weight` etc. in `config.py`.

**`weighted_transfer(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray`**
- **Description:** Similar to basic transfer but allows for weighted influence of source color statistics.
- **Input:**
    - `source_rgb` (np.ndarray): Source image in RGB format.
    - `target_rgb` (np.ndarray): Target image in RGB format.
- **Output:** `np.ndarray`: Resulting image in RGB format.
- **Configurable via:** `weighted_L_source_influence`, `weighted_A_source_influence`, `weighted_B_source_influence` in `config.py`.

#### `ImageProcessor`
**Przeznaczenie:** Handles image loading, processing (single or batch), and saving, using `LABColorTransfer`.
Located in: `app/algorithms/algorithm_05_lab_transfer/processor.py`

##### Konstruktor
```python
ImageProcessor(config: Optional[dict] = None, use_gpu: bool = True)
```
**Parametry:**
- `config` (dict, optional): Configuration for `LABColorTransfer`.
- `use_gpu` (bool, optional, default=True): Passed to `LABColorTransfer`.

##### Główne metody

**`process_image(method: str, source_image_data: Union[str, np.ndarray, Image.Image], target_image_data: Union[str, np.ndarray, Image.Image], mask_image_data: Optional[Union[str, np.ndarray, Image.Image]] = None) -> np.ndarray`**
- **Description:** Processes a single pair of source/target images using the specified method.
- **Input:**
    - `method` (str): Transfer method ('basic', 'adaptive', 'selective', 'weighted').
    - `source_image_data`: Path to source image, NumPy array, or PIL Image.
    - `target_image_data`: Path to target image, NumPy array, or PIL Image.
    - `mask_image_data` (optional): Path to mask image, NumPy array, or PIL Image (for 'selective' method, if applicable).
- **Output:** `np.ndarray`: Resulting image in RGB format.

**`process_batch(image_operations: List[dict]) -> List[np.ndarray]`**
- **Description:** Processes a batch of image operations.
- **Input:** `image_operations` (List[dict]): A list of dictionaries, each specifying an operation:
    ```python
    [
        {
            'method': 'basic',
            'source': 'path/to/source1.jpg', # or np.ndarray or PIL.Image
            'target': 'path/to/target1.jpg', # or np.ndarray or PIL.Image
            # 'mask': 'path/to/mask1.png' (optional)
        },
        # ... more operations
    ]
    ```
- **Output:** `List[np.ndarray]`: List of resulting images in RGB format.

#### `LABColorTransferGPU` (Internal/Advanced Use)
**Przeznaczenie:** Handles the GPU-specific implementations using OpenCL. Generally used internally by `LABColorTransfer`.
Located in: `app/algorithms/algorithm_05_lab_transfer/gpu_core.py`
- **API:** Similar methods to `LABColorTransfer` (e.g., `basic_transfer_gpu`) but directly execute GPU code. Requires OpenCL context and queue.

### Klasy dostępne

#### [[ClassName]]
**Przeznaczenie:** Waliduje i przetwarza dane użytkownika zgodnie z regułami biznesowymi

##### Konstruktor
```python
ClassName(config_path: str, timeout: int = 30, cache_enabled: bool = True)
```
**Parametry:**
- `config_path` (str, required): Ścieżka do pliku z regułami walidacji (.json)
- `timeout` (int, optional, default=30): Timeout dla operacji w sekundach (1-300)
- `cache_enabled` (bool, optional, default=True): Czy używać cache'a wyników

##### Główne metody

**[[process()]]**
```python
result = instance.process(data: dict, options: list = []) -> ProcessResult
```
- **Input:** `data` musi zawierać klucze: ['user_id', 'email', 'profile_data']
- **Input:** `options` lista z dozwolonych: ['strict_mode', 'auto_fix', 'generate_report']
- **Output:** `ProcessResult` obiekt z polami:
  - `.status` (str): 'success'|'error'|'warning'|'partial'
  - `.data` (dict): przetworzone dane z dodatkowymi polami
  - `.errors` (list[dict]): lista {'code': str, 'message': str, 'field': str}
  - `.warnings` (list[str]): lista ostrzeżeń
  - `.metadata` (dict): statystyki przetwarzania

**[[validate_single()]]**
```python
is_valid = instance.validate_single(item: dict, rule_set: str = 'default') -> ValidationResult
```
- **Input:** `item` dict z danymi do walidacji
- **Input:** `rule_set` nazwa zestawu reguł ('default', 'strict', 'minimal')
- **Output:** `ValidationResult` z polami:
  - `.is_valid` (bool): czy przeszło walidację
  - `.errors` (list): lista błędów walidacji
  - `.score` (float): wynik walidacji 0.0-1.0

**[[get_stats()]]**
```python
stats = instance.get_stats() -> dict
```
- **Output:** Słownik ze statystykami:
  - `processed_count` (int): liczba przetworzonych elementów
  - `success_rate` (float): procent sukcesu
  - `avg_processing_time` (float): średni czas przetwarzania w ms

### Typowe użycie

```python
from PIL import Image
import numpy as np
from app.algorithms.algorithm_05_lab_transfer.core import LABColorTransfer
from app.algorithms.algorithm_05_lab_transfer.config import get_config, update_config

# --- Scenario 1: Basic transfer with default settings ---
try:
    source_pil = Image.open('source.jpg').convert('RGB')
    target_pil = Image.open('target.jpg').convert('RGB')
except FileNotFoundError:
    print("Ensure source.jpg and target.jpg exist in the current directory.")
    exit()


source_rgb = np.array(source_pil)
target_rgb = np.array(target_pil)

config = get_config()
transfer_agent = LABColorTransfer(config=config, use_gpu=True) # Tries GPU, falls to CPU if needed

if source_rgb.ndim == 3 and source_rgb.shape[2] == 3 and \
   target_rgb.ndim == 3 and target_rgb.shape[2] == 3:
    result_basic_rgb = transfer_agent.basic_transfer(source_rgb, target_rgb)
    Image.fromarray(result_basic_rgb).save('result_basic.jpg')
    print("Basic transfer saved to result_basic.jpg")
else:
    print("Error: Source and target images must be RGB.")

# --- Scenario 2: Adaptive transfer with custom settings ---

# Update configuration (optional)
update_config(config, {
    'adaptive_num_segments': 16,  # Increase number of segments for more detail
    'adaptive_segment_threshold': 0.2,  # Adjust threshold for segmentation
})

# Perform adaptive transfer
if source_rgb.ndim == 3 and source_rgb.shape[2] == 3 and \
   target_rgb.ndim == 3 and target_rgb.shape[2] == 3:
    result_adaptive_rgb = transfer_agent.adaptive_transfer(source_rgb, target_rgb)
    Image.fromarray(result_adaptive_rgb).save('result_adaptive.jpg')
    print("Adaptive transfer saved to result_adaptive.jpg")

# --- Scenario 3: Process a batch of images ---
from app.algorithms.algorithm_05_lab_transfer.processor import ImageProcessor

# Initialize processor
processor = ImageProcessor(config=config, use_gpu=True)

# Prepare batch operations
image_operations = [
    {
        'method': 'basic',
        'source': source_rgb,  # Could also be file paths or PIL Images
        'target': target_rgb,
    },
    {
        'method': 'adaptive',
        'source': source_rgb,
        'target': target_rgb,
    },
    {
        'method': 'weighted',
        'source': source_rgb,
        'target': target_rgb,
    },
]

# Process batch
results = processor.process_batch(image_operations)

# Save results
for i, result_rgb in enumerate(results):
    Image.fromarray(result_rgb).save(f'batch_result_{i}.jpg')
print(f"Processed {len(results)} images in batch.")
```

### Error Handling

#### Wyjątki (Exceptions)
- **`LABTransferError`**: Base exception for all module-specific errors.
- **`GPUInitializationError`**: Raised when GPU initialization fails but was requested.
- **`InvalidImageFormatError`**: Raised when input images have incorrect format or dimensions.
- **`ConfigurationError`**: Raised for invalid configuration values.

#### Obsługa błędów
```python
from app.algorithms.algorithm_05_lab_transfer.exceptions import (
    LABTransferError, GPUInitializationError, InvalidImageFormatError, ConfigurationError
)

try:
    # Attempt to use GPU-accelerated transfer
    transfer_agent = LABColorTransfer(use_gpu=True)
    result = transfer_agent.basic_transfer(source_rgb, target_rgb)
    
    # Process the result...
    
except GPUInitializationError as e:
    print(f"GPU initialization failed, falling back to CPU: {e}")
    # Optionally continue with CPU-only mode
    transfer_agent = LABColorTransfer(use_gpu=False)
    result = transfer_agent.basic_transfer(source_rgb, target_rgb)
    
except InvalidImageFormatError as e:
    print(f"Invalid image format: {e}")
    # Handle invalid image format (e.g., wrong dimensions, not RGB, etc.)
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle invalid configuration (e.g., out-of-range values)
    
except LABTransferError as e:
    print(f"Color transfer error: {e}")
    # Handle other module-specific errors

except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

### Dependencies
**Import:**
```python
# Main classes
from app.algorithms.algorithm_05_lab_transfer.core import LABColorTransfer
from app.algorithms.algorithm_05_lab_transfer.processor import ImageProcessor

# Configuration
from app.algorithms.algorithm_05_lab_transfer.config import get_config, update_config, reset_config

# Exceptions
from app.algorithms.algorithm_05_lab_transfer.exceptions import (
    LABTransferError, GPUInitializationError, 
    InvalidImageFormatError, ConfigurationError
)
```

**External dependencies:**
```txt
# Core requirements
numpy>=1.19.0
Pillow>=8.0.0
scikit-image>=0.18.0

# Optional for GPU acceleration
pyopencl>=2021.1.1  # Requires compatible OpenCL drivers and hardware

# For testing
pytest>=6.0.0
pytest-cov>=2.8.0
```

**Python version compatibility:**
- Python 3.8 or higher is required.
- For GPU acceleration, a compatible OpenCL implementation is required (e.g., NVIDIA CUDA, AMD ROCm, or Intel OpenCL).

### File locations
- **Core Implementation:**
  - `core.py` - Main `LABColorTransfer` class with CPU implementations
  - `gpu_core.py` - GPU-accelerated implementations using OpenCL
  - `kernels.cl` - OpenCL kernel code for GPU acceleration
  - `config.py` - Configuration management and default settings
  - `processor.py` - `ImageProcessor` class for batch processing
  - `exceptions.py` - Custom exception classes
  - `logger.py` - Logging configuration
  - `utils.py` - Utility functions and helpers

- **Tests:**
  - `tests/` - Contains all test files
    - `test_lab_transfer_basic.py` - Basic transfer method tests
    - `test_lab_transfer_adaptive.py` - Adaptive transfer method tests
    - `test_lab_transfer_selective.py` - Selective transfer method tests
    - `test_lab_transfer_weighted.py` - Weighted transfer method tests
    - `test_processor.py` - ImageProcessor tests
    - `test_images/` - Test images in .npy format
    - `regenerate_test_images.py` - Script to generate test images

- **Documentation:**
  - `README.md` - This file (main documentation)
  - `README.concepts.md` - Conceptual documentation and design decisions
  - `README.todo.md` - Roadmap and future work

### Configuration

The module uses a centralized configuration system defined in `config.py`. You can customize the behavior of the color transfer algorithms by modifying these parameters.

#### Default Configuration

```python
# Default configuration values
DEFAULT_CONFIG = {
    # Basic transfer parameters
    'basic_L_preserve': True,  # Preserve L channel (luminance) in basic transfer
    'basic_AB_preserve': True,  # Preserve AB channels in basic transfer
    
    # Adaptive transfer parameters
    'adaptive_num_segments': 8,  # Number of segments for adaptive transfer
    'adaptive_segment_threshold': 0.1,  # Threshold for segment merging
    'adaptive_use_gpu': True,  # Use GPU for adaptive transfer if available
    
    # Selective transfer parameters
    'selective_color_weight': 0.5,  # Weight for color similarity in selective transfer
    'selective_spatial_weight': 0.5,  # Weight for spatial distance in selective transfer
    'selective_use_gpu': True,  # Use GPU for selective transfer if available
    
    # Weighted transfer parameters
    'weighted_L_source_influence': 0.5,  # Influence of source L channel (0-1)
    'weighted_A_source_influence': 0.5,  # Influence of source A channel (0-1)
    'weighted_B_source_influence': 0.5,  # Influence of source B channel (0-1)
    'weighted_use_gpu': True,  # Use GPU for weighted transfer if available
    
    # General parameters
    'use_gpu': True,  # Global GPU usage flag
    'gpu_device_type': 'GPU',  # 'GPU', 'CPU', or 'ACCELERATOR'
    'gpu_platform_name': None,  # Specific platform name or None for auto-select
    'max_image_size': 4096,  # Maximum width/height for GPU processing
}
```

#### Modifying Configuration

You can modify the configuration in several ways:

1. **Temporarily for a single instance:**
   ```python
   from app.algorithms.algorithm_05_lab_transfer.core import LABColorTransfer
   
   custom_config = {
       'adaptive_num_segments': 16,
       'adaptive_segment_threshold': 0.2,
   }
   
   transfer_agent = LABColorTransfer(config=custom_config)
   ```

2. **Update global configuration:**
   ```python
   from app.algorithms.algorithm_05_lab_transfer.config import update_config, get_config
   
   # Get current config
   config = get_config()
   
   # Update specific values
   update_config(config, {
       'adaptive_num_segments': 16,
       'adaptive_segment_threshold': 0.2,
   })
   
   # Now all new instances will use the updated config
   transfer_agent = LABColorTransfer()
   ```

3. **Reset to defaults:**
   ```python
   from app.algorithms.algorithm_05_lab_transfer.config import reset_config, get_config
   
   # Reset to default values
   reset_config()
   
   # Verify default values
   config = get_config()
   print(config['adaptive_num_segments'])  # Will print 8 (default)
   ```

#### GPU Configuration

For GPU acceleration, you can specify which OpenCL device to use:

```python
# Example: Force using a specific OpenCL platform/device
import pyopencl as cl

# List available platforms and devices
for platform in cl.get_platforms():
    print(f"Platform: {platform.name}")
    for device in platform.get_devices():
        print(f"  Device: {device.name}")

# Use a specific device by platform name
transfer_agent = LABColorTransfer(config={
    'gpu_platform_name': 'NVIDIA CUDA',  # Or 'Intel(R) OpenCL', 'AMD Accelerated Parallel Processing', etc.
    'gpu_device_type': 'GPU'  # 'GPU', 'CPU', or 'ACCELERATOR'
})
```

### Environment Variables

- `LAB_TRANSFER_USE_GPU`: Set to '0' to disable GPU acceleration globally
- `LAB_TRANSFER_DEBUG`: Set to '1' to enable debug logging
- `PYOPENCL_CTX`: Set to control PyOpenCL context creation (e.g., '0' for first GPU, '1' for second GPU, etc.)

### Logging

The module uses Python's built-in `logging` module. You can configure logging as follows:

```python
import logging

# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(level=logging.INFO)

# Or configure specific logger
logger = logging.getLogger('algorithm_05_lab_transfer')
logger.setLevel(logging.DEBUG)
```

### Rozdział 1 
- klasyczne README - szybki start, overview, podstawowa orientacja

### Rozdział 2 
{{ ... }}

**Details:**
- Konkretne API - dokładne sygnatury metod z typami
- Kompletne przykłady - AI widzi jak używać bez czytania kodu
- Error handling - wszystkie możliwe błędy i kody
- Dependencies - dokładnie co importować
- Lokalizacje - gdzie znajdzie kod jeśli jednak musi
- Cel: AI agent może użyć modułu bez oglądania kodu źródłowego.