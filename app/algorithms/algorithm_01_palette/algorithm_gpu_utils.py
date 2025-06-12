# /app/algorithms/algorithm_01_palette/algorithm-gpu-utils.py
# Moduł zawierający podstawowe, współdzielone komponenty i definicje.

import logging
import numpy as np
from enum import Enum
from typing import Any, TYPE_CHECKING, Tuple, List

# --- Project Imports with Fallbacks ---
# Umożliwia działanie modułu nawet poza główną strukturą projektu.
try:
    if TYPE_CHECKING:
        from ...core.development_logger import DevelopmentLogger
except ImportError:
    # Definicje zastępcze, jeśli główne moduły nie są dostępne
    DevelopmentLogger = logging.Logger
    
    class PerformanceProfiler:
        def profile_operation(self, *args, **kwargs):
            import contextlib
            return contextlib.nullcontext()

# --- Fallback Logger and Profiler ---

def get_logger() -> Any:
    """Zwraca instancję loggera, zapewniając fallback, jeśli główny system logowania jest niedostępny."""
    try:
        from ...core.development_logger import get_logger as get_core_logger
        return get_core_logger()
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        return logging.getLogger("fallback_logger")

def get_profiler() -> Any:
    """Zwraca instancję profilera, zapewniając fallback."""
    try:
        from ...core.performance_profiler import get_profiler as get_core_profiler
        return get_core_profiler()
    except ImportError:
        return PerformanceProfiler()

# --- Enhanced Custom Exceptions ---

# --- Validation Utilities ---

def validate_image_array(image_array):
    """Validates input image array and returns its dimensions.
    
    Args:
        image_array: Input image as numpy array
        
    Returns:
        Tuple of (height, width, channels)
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if image_array.dtype != np.uint8:
        raise ValueError("Input array must have dtype uint8")
    
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Input must be a 3D array with shape (H,W,3)")
    
    return image_array.shape

def validate_palette(palette):
    """Validates palette and converts it to numpy array.
    
    Args:
        palette: List of [R,G,B] colors with values 0-255
        
    Returns:
        numpy.ndarray: Palette as float32 array with values 0-1
        
    Raises:
        ValueError: If validation fails
    """
    if not palette or not all(isinstance(c, (list, tuple)) and len(c) == 3 for c in palette):
        raise ValueError("Palette must be a non-empty list of [R,G,B] lists")
    
    palette_np = np.array(palette, dtype=np.float32)
    
    if np.any((palette_np < 0) | (palette_np > 255)):
        raise ValueError("Palette values must be in range [0, 255]")
    
    return palette_np / 255.0

# --- Acceleration Strategy Enum ---

class AccelerationStrategy(Enum):
    """Definiuje strategię wyboru backendu do przetwarzania."""
    CPU = 0
    GPU_SMALL = 1    # Małe palety, prosty algorytm
    GPU_MEDIUM = 2   # Średnia złożoność
    GPU_LARGE = 3    # Pełny potok GPU z przetwarzaniem wsadowym
