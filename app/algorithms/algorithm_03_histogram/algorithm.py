"""
Algorithm 03: Histogram Matching
===============================

Enhanced modular implementation of histogram matching algorithm.
Focuses on luminance channel matching for natural-looking results.

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

from app.core.development_logger import get_logger
from app.core.performance_profiler import get_profiler
from app.core.file_handler import get_result_path


class HistogramMatchingAlgorithm:
    """
    Enhanced Histogram Matching Algorithm
    
    Core functionality:
    1. Convert images to LAB color space
    2. Extract luminance (L) channel histograms
    3. Build cumulative distribution functions (CDF)
    4. Create lookup table for histogram matching
    5. Apply transformation and convert back to RGB
    """
    
    def __init__(self, algorithm_id: str = "algorithm_03_histogram"):
        self.algorithm_id = algorithm_id
        self.logger = get_logger()
        self.profiler = get_profiler()
        
        # Histogram parameters
        self.histogram_bins = 256
        # Poprawka: `range` w np.histogram oczekuje krotki (tuple)
        self.histogram_range: Tuple[int, int] = (0, 256)
        
        self.logger.info(f"Initialized {self.algorithm_id}")
    
    # Poprawka: Zmieniono typ zwracany na krotkę
    def extract_luminance_channel(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract luminance (L) channel from BGR image via LAB conversion."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_extract_luminance"):
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            luminance = lab_image[:, :, 0]  # L channel
            self.logger.debug(f"Extracted luminance channel: {luminance.shape}")
            return lab_image, luminance
    
    def compute_histogram(self, channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram and cumulative distribution function."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_compute_histogram"):
            # Calculate histogram
            hist, bins = np.histogram(channel.flatten(), self.histogram_bins, self.histogram_range)
            
            # Calculate cumulative distribution function (CDF)
            cdf = hist.cumsum()
            
            # Normalize CDF to [0, 1] range
            cdf_normalized = cdf / cdf[-1] if cdf[-1] > 0 else cdf
            
            self.logger.debug(f"Computed histogram: {len(hist)} bins, CDF max: {cdf[-1]}")
            return hist, cdf_normalized
    
    def create_lookup_table(self, master_cdf: np.ndarray, target_cdf: np.ndarray) -> np.ndarray:
        """Create lookup table for histogram matching transformation."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_create_lookup"):
            lookup_table = np.zeros(self.histogram_bins, dtype=np.uint8)
            
            for i in range(self.histogram_bins):
                # Find closest value in master CDF for each target CDF value
                differences = np.abs(master_cdf - target_cdf[i])
                closest_idx = np.argmin(differences)
                lookup_table[i] = closest_idx
            
            self.logger.debug(f"Created lookup table with {self.histogram_bins} entries")
            return lookup_table
    
    def apply_histogram_matching(self, lab_image: np.ndarray, luminance: np.ndarray, 
                               lookup_table: np.ndarray) -> np.ndarray:
        """Apply histogram matching using lookup table to luminance channel."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_apply_matching"):
            # Apply lookup table to luminance channel
            result_lab = lab_image.copy()
            result_lab[:, :, 0] = lookup_table[luminance]
            
            # Convert back to BGR
            result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
            
            self.logger.debug(f"Applied histogram matching to luminance channel")
            return result_bgr
    
    def process(self, master_path: str, target_path: str) -> str:
        """
        Main processing method - applies histogram matching algorithm.
        
        Args:
            master_path: Path to master image (source of histogram)
            target_path: Path to target image (will be histogram-matched)
            
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
            
            self.logger.info("Starting histogram matching")
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
                
                # Extract luminance channels
                master_lab, master_luminance = self.extract_luminance_channel(master_image)
                target_lab, target_luminance = self.extract_luminance_channel(target_image)
                
                # Compute histograms and CDFs
                master_hist, master_cdf = self.compute_histogram(master_luminance)
                target_hist, target_cdf = self.compute_histogram(target_luminance)
                
                # Create lookup table for histogram matching
                lookup_table = self.create_lookup_table(master_cdf, target_cdf)
                
                # Apply histogram matching
                result_image = self.apply_histogram_matching(target_lab, target_luminance, lookup_table)
                
                # Save result
                result_path = get_result_path(os.path.basename(target_path))
                success = cv2.imwrite(result_path, result_image)
                
                if not success:
                    raise RuntimeError(f"Failed to save result image: {result_path}")
                
                self.logger.success(f"Histogram matching completed: {result_path}")
                return result_path
                
            except Exception as e:
                self.logger.error(f"Histogram matching failed: {str(e)}", exc_info=True)
                raise RuntimeError(f"Algorithm processing failed: {str(e)}") from e
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information for monitoring and documentation."""
        return {
            'algorithm_id': self.algorithm_id,
            'name': 'Histogram Matching',
            'description': 'Luminance channel histogram specification',
            'version': '2.0.0',
            'color_space': 'LAB (L channel only)',
            'parameters': {
                'histogram_bins': self.histogram_bins,
                'histogram_range': list(self.histogram_range), # Zwróć jako listę dla JSON
                'target_channel': 'luminance'
            },
            'supported_formats': ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'],
            'complexity': 'O(n + bins)',
            'memory_usage': 'O(n + bins)'
        }


# Factory function for easy algorithm creation
def create_histogram_matching_algorithm() -> HistogramMatchingAlgorithm:
    """Create and return a new histogram matching algorithm instance."""
    return HistogramMatchingAlgorithm()


# Legacy compatibility function
def simple_histogram_matching(master_path: str, target_path: str) -> str:
    """
    Legacy compatibility function for existing API.
    
    This maintains backward compatibility with existing code while using
    the new modular algorithm implementation.
    """
    algorithm = create_histogram_matching_algorithm()
    return algorithm.process(master_path, target_path)

