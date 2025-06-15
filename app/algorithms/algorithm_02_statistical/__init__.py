"""
Algorithm 02: Statistical Transfer
=================================

This module provides statistical color transfer functionality using LAB color space.
"""

from .algorithm import (
    StatisticalTransferAlgorithm,
    create_statistical_transfer_algorithm,
    basic_statistical_transfer
)

__all__ = [
    'StatisticalTransferAlgorithm',
    'create_statistical_transfer_algorithm', 
    'basic_statistical_transfer'
]
