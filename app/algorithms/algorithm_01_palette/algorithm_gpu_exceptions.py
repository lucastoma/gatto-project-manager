# /app/algorithms/algorithm_01_palette/algorithm_gpu_exceptions.py
# Module containing custom exceptions for the GPU algorithm.

class GPUProcessingError(Exception):
    """Custom exception for GPU processing errors."""
    pass

class GPUMemoryError(GPUProcessingError):
    """Specific exception for GPU memory issues."""
    pass

class ImageProcessingError(Exception):
    """Exception for image loading and processing errors."""
    pass
