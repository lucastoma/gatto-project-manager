# /app/algorithms/algorithm_01_palette/algorithm-gpu-utils.py
# Moduł zawierający podstawowe, współdzielone komponenty i definicje.

import logging
from enum import Enum
from typing import Any, TYPE_CHECKING

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

class GPUProcessingError(Exception):
    """Niestandardowy wyjątek dla błędów przetwarzania na GPU."""
    pass

class GPUMemoryError(GPUProcessingError):
    """Szczególny wyjątek dla problemów z pamięcią GPU."""
    pass

class ImageProcessingError(Exception):
    """Wyjątek dla błędów ładowania lub przetwarzania obrazów."""
    pass

# --- Acceleration Strategy Enum ---

class AccelerationStrategy(Enum):
    """Definiuje strategię wyboru backendu do przetwarzania."""
    CPU = 0
    GPU_SMALL = 1    # Małe palety, prosty algorytm
    GPU_MEDIUM = 2   # Średnia złożoność
    GPU_LARGE = 3    # Pełny potok GPU z przetwarzaniem wsadowym
