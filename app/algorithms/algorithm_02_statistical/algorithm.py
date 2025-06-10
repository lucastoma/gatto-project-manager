"""
Algorithm 02: Statistical Transfer
=================================

Enhanced modular implementation of statistical color transfer algorithm.
Operates in LAB color space for better perceptual accuracy.

Design Philosophy: "Bezpiecznie = Szybko"
- Clear separation of concerns
- Comprehensive error handling  
- Performance monitoring integration
- Easy testing and validation
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from ...core.development_logger import get_logger
# Poprawka: Dodano bezpośredni import klas, aby pomóc Pylance w analizie typów
from ...core.performance_profiler import get_profiler, PerformanceProfiler 
from ...core.file_handler import get_result_path


class StatisticalTransferAlgorithm:
    """
    Enhanced Statistical Transfer Algorithm
    
    Core functionality:
    1. Convert images to LAB color space for perceptual accuracy
    2. Calculate statistical moments (mean, std) for each channel
    3. Transfer master's statistics to target image
    4. Apply proper LAB range clipping and convert back to RGB
    """
    
    def __init__(self, algorithm_id: str = "algorithm_02_statistical"):
        self.algorithm_id = algorithm_id
        self.logger = get_logger()
        # Poprawka: Dodano jawną adnotację typu, aby rozwiązać problemy Pylance
        self.profiler: PerformanceProfiler = get_profiler()
        
        # LAB color space ranges
        self.lab_ranges = {
            'L': (0, 100),    # Lightness: 0-100
            'a': (-127, 127), # Green-Red: -127 to 127  
            'b': (-127, 127)  # Blue-Yellow: -127 to 127
        }
        
        self.logger.info(f"Initialized {self.algorithm_id}")
    
    def convert_to_lab(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to LAB color space."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_convert_to_lab"):
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            self.logger.debug(f"Converted image to LAB: {lab_image.shape}")
            return lab_image
    
    def convert_to_bgr(self, lab_image: np.ndarray) -> np.ndarray:
        """Convert LAB image back to BGR color space."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_convert_to_bgr"):
            # Ensure proper LAB range clipping before conversion
            clipped_lab = self.clip_lab_ranges(lab_image)
            bgr_image = cv2.cvtColor(clipped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            self.logger.debug(f"Converted LAB back to BGR: {bgr_image.shape}")
            return bgr_image
    
    def clip_lab_ranges(self, lab_image: np.ndarray) -> np.ndarray:
        """Apply proper LAB range clipping to prevent conversion artifacts."""
        clipped = lab_image.copy()
        clipped[:, :, 0] = np.clip(clipped[:, :, 0], self.lab_ranges['L'][0], self.lab_ranges['L'][1])  # L channel
        clipped[:, :, 1] = np.clip(clipped[:, :, 1], self.lab_ranges['a'][0], self.lab_ranges['a'][1])  # a channel
        clipped[:, :, 2] = np.clip(clipped[:, :, 2], self.lab_ranges['b'][0], self.lab_ranges['b'][1])  # b channel
        return clipped
    
    def calculate_statistics(self, lab_image: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate mean and standard deviation for each LAB channel."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_calculate_stats"):
            stats = {}
            channel_names = ['L', 'a', 'b']
            
            for i, channel in enumerate(channel_names):
                channel_data = lab_image[:, :, i]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                stats[channel] = (mean, std)
                self.logger.debug(f"Channel {channel}: mean={mean:.2f}, std={std:.2f}")
            
            return stats
    
    def transfer_statistics(self, target_lab: np.ndarray, master_stats: Dict[str, Tuple[float, float]], 
                          target_stats: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Transfer statistical properties from master to target image."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_transfer_stats"):
            result_lab = target_lab.copy()
            channel_names = ['L', 'a', 'b']
            
            for i, channel in enumerate(channel_names):
                master_mean, master_std = master_stats[channel]
                target_mean, target_std = target_stats[channel]
                
                # Apply statistical transfer: normalize and rescale
                if target_std > 0:
                    result_lab[:, :, i] = (target_lab[:, :, i] - target_mean) * (master_std / target_std) + master_mean
                else:
                    # If target std is 0, just shift to master mean
                    result_lab[:, :, i] = master_mean
                
                self.logger.debug(f"Transferred {channel} channel statistics")
            
            return result_lab
    
    def process(self, master_path: str, target_path: str) -> str:
        """
        Main processing method - applies statistical transfer algorithm.
        
        Args:
            master_path: Path to master image (source of color statistics)
            target_path: Path to target image (will be color-matched)
            
        Returns:
            Path to result image file
            
        Raises:
            FileNotFoundError: If input images don't exist
            RuntimeError: If processing fails
        """
        with self.profiler.profile_operation(f"{self.algorithm_id}_process"):
            # Set algorithm context for logging
            self.logger.set_algorithm_context(self.algorithm_id)
            
            # Validate input files
            if not os.path.exists(master_path):
                raise FileNotFoundError(f"Master image not found: {master_path}")
                
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Target image not found: {target_path}")
            
            self.logger.info("Starting statistical transfer")
            self.logger.debug(f"Master: {master_path}")
            self.logger.debug(f"Target: {target_path}")
            
            try:
                # Load images
                master_image = cv2.imread(master_path)
                target_image = cv2.imread(target_path)
                
                if master_image is None:
                    raise RuntimeError(f"Failed to load master image: {master_path}")
                if target_image is None:
                    raise RuntimeError(f"Failed to load target image: {target_path}")
                
                self.logger.debug(f"Master shape: {master_image.shape}")
                self.logger.debug(f"Target shape: {target_image.shape}")
                
                # Convert to LAB color space
                master_lab = self.convert_to_lab(master_image)
                target_lab = self.convert_to_lab(target_image)
                
                # Calculate statistics for both images
                master_stats = self.calculate_statistics(master_lab)
                target_stats = self.calculate_statistics(target_lab)
                
                # Transfer statistics from master to target
                result_lab = self.transfer_statistics(target_lab, master_stats, target_stats)
                
                # Convert back to BGR
                result_image = self.convert_to_bgr(result_lab)
                
                # Save result
                result_path = get_result_path(os.path.basename(target_path))
                success = cv2.imwrite(result_path, result_image)
                
                if not success:
                    raise RuntimeError(f"Failed to save result image: {result_path}")
                
                self.logger.success(f"Statistical transfer completed: {result_path}")
                return result_path
                
            except Exception as e:
                # Poprawka: Dodano exc_info=True dla pełnego tracebacku w logach
                self.logger.error(f"Statistical transfer failed: {str(e)}", exc_info=True)
                raise RuntimeError(f"Algorithm processing failed: {str(e)}") from e
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information for monitoring and documentation."""
        return {
            'algorithm_id': self.algorithm_id,
            'name': 'Statistical Transfer',
            'description': 'LAB color space statistical moment matching',
            'version': '2.0.0',
            'color_space': 'LAB',
            'parameters': {
                'statistical_moments': ['mean', 'standard_deviation'],
                'channels': ['L', 'a', 'b']
            },
            'supported_formats': ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'],
            'complexity': 'O(n)',
            'memory_usage': 'O(n)'
        }


# Factory function for easy algorithm creation
def create_statistical_transfer_algorithm() -> StatisticalTransferAlgorithm:
    """Create and return a new statistical transfer algorithm instance."""
    return StatisticalTransferAlgorithm()


# Legacy compatibility function
def basic_statistical_transfer(master_path: str, target_path: str) -> str:
    """
    Legacy compatibility function for existing API.
    
    This maintains backward compatibility with existing code while using
    the new modular algorithm implementation.
    """
    algorithm = create_statistical_transfer_algorithm()
    return algorithm.process(master_path, target_path)
