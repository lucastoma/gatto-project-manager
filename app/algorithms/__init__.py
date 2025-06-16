"""
GattoNero AI Assistant - Algorithm Modules
==========================================

This package contains modular algorithm implementations for color matching
and image processing. Each algorithm is self-contained with comprehensive
monitoring, testing, and documentation.

Available Algorithms:
- Algorithm 01: Palette Mapping (K-means based color palette extraction)
- Algorithm 02: Statistical Transfer (LAB color space statistical matching)
- Algorithm 03: Histogram Matching (Luminance channel histogram specification)
"""

# Import algorithm factories for easy access
from .algorithm_01_palette import (
    create_palette_mapping_algorithm
)
from .algorithm_02_statistical import (
    create_statistical_transfer_algorithm,
    basic_statistical_transfer
)
from .algorithm_03_histogram import (
    create_histogram_matching_algorithm,
    simple_histogram_matching
)
from .algorithm_05_lab_transfer import (
    create_lab_transfer_algorithm
)

# Algorithm registry for dynamic access
ALGORITHM_REGISTRY = {
    'algorithm_01_palette': create_palette_mapping_algorithm,
    'algorithm_02_statistical': create_statistical_transfer_algorithm,
    'algorithm_03_histogram': create_histogram_matching_algorithm,
    'algorithm_05_lab_transfer': create_lab_transfer_algorithm,
}

# Legacy function mapping for backward compatibility
LEGACY_FUNCTIONS = {
    'method2': basic_statistical_transfer,
    'method3': simple_histogram_matching,
}

def get_algorithm(algorithm_id: str):
    """Get algorithm instance by ID."""
    if algorithm_id in ALGORITHM_REGISTRY:
        return ALGORITHM_REGISTRY[algorithm_id]()
    raise ValueError(f"Unknown algorithm: {algorithm_id}")

def get_legacy_function(method: str):
    """Get legacy function by method name."""
    if method in LEGACY_FUNCTIONS:
        return LEGACY_FUNCTIONS[method]
    raise ValueError(f"Unknown method: {method}")

__all__ = [
    # Algorithm factories
    'create_palette_mapping_algorithm',
    'create_statistical_transfer_algorithm', 
    'create_histogram_matching_algorithm',
    'create_lab_transfer_algorithm',
    
    # Legacy compatibility functions
    'basic_statistical_transfer',
    'simple_histogram_matching',
    
    # Dynamic access
    'get_algorithm',
    'get_legacy_function',
    'ALGORITHM_REGISTRY',
    'LEGACY_FUNCTIONS'
]
