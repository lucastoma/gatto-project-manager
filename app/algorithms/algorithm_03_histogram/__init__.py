"""
Algorithm 03: Histogram Matching
===============================

This module provides histogram matching functionality focusing on luminance channels.
"""

from .algorithm import (
    HistogramMatchingAlgorithm,
    create_histogram_matching_algorithm,
    simple_histogram_matching
)

__all__ = [
    'HistogramMatchingAlgorithm',
    'create_histogram_matching_algorithm',
    'simple_histogram_matching'
]
