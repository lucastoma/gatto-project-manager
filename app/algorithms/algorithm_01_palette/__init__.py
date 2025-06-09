"""
Algorithm 01: Palette Mapping
============================

This module provides palette-based color matching functionality using K-means clustering.
"""

from .algorithm import (
    PaletteMappingAlgorithm,
    create_palette_mapping_algorithm,
    simple_palette_mapping
)

__all__ = [
    'PaletteMappingAlgorithm',
    'create_palette_mapping_algorithm',
    'simple_palette_mapping'
]
