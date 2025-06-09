"""
Algorithm 01: Palette Mapping
============================

Enhanced modular implementation of palette-based color matching algorithm.
Extracted from legacy code with improved structure, monitoring, and testability.

Design Philosophy: "Bezpiecznie = Szybko"
- Clear separation of concerns
- Comprehensive error handling  
- Performance monitoring integration
- Easy testing and validation
"""

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from app.core.development_logger import get_logger
# Poprawka: Dodano bezpośredni import klas, aby pomóc Pylance w analizie typów
from app.core.performance_profiler import get_profiler, PerformanceProfiler
from app.core.file_handler import get_result_path


class PaletteMappingAlgorithm:
    """
    Enhanced Palette Mapping Algorithm
    
    Core functionality:
    1. Extract dominant colors from master image using K-means
    2. Map target image pixels to closest master palette colors
    3. Generate result with master's color scheme applied to target's content
    """
    
    def __init__(self, algorithm_id: str = "algorithm_01_palette"):
        self.algorithm_id = algorithm_id
        self.logger = get_logger()
        # Poprawka: Dodano jawną adnotację typu, aby rozwiązać problemy Pylance
        self.profiler: PerformanceProfiler = get_profiler()
        
        # Default parameters
        self.default_params = {
            'k_colors': 8,
            'random_state': 42,
            'n_init': 10,
            'max_iter': 300,
            'tol': 1e-4
        }
        
        self.logger.info(f"Initialized {self.algorithm_id}")
    
    def extract_palette(self, image: np.ndarray, k_colors: int) -> np.ndarray:
        """Extract dominant colors from image using K-means clustering."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_extract_palette"):
            self.logger.set_algorithm_context(self.algorithm_id)
            
            # Reshape image to pixel array
            pixels = image.reshape(-1, 3).astype(np.float32)
            self.logger.debug(f"Processing {len(pixels)} pixels for palette extraction")
            
            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=k_colors,
                random_state=self.default_params['random_state'],
                n_init=self.default_params['n_init'],
                max_iter=self.default_params['max_iter'],
                tol=self.default_params['tol']
            )
            
            kmeans.fit(pixels)
            palette = kmeans.cluster_centers_
            
            self.logger.success(f"Extracted {k_colors} colors from palette")
            return palette
    
    def map_colors(self, target_image: np.ndarray, master_palette: np.ndarray, k_colors: int) -> np.ndarray:
        """Map target image colors to closest master palette colors."""
        with self.profiler.profile_operation(f"{self.algorithm_id}_map_colors"):
            # Reshape target to pixels
            target_pixels = target_image.reshape(-1, 3).astype(np.float32)
            
            # Extract target palette for mapping
            kmeans_target = KMeans(
                n_clusters=k_colors,
                random_state=self.default_params['random_state'],
                n_init=self.default_params['n_init']
            )
            target_labels = kmeans_target.fit_predict(target_pixels)
            target_palette = kmeans_target.cluster_centers_
            
            # Create mapping from target palette to master palette
            mapped_pixels = np.zeros_like(target_pixels)
            
            for i, target_color in enumerate(target_palette):
                # Find closest color in master palette using Euclidean distance
                distances = np.sum((master_palette - target_color) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                mapped_pixels[target_labels == i] = master_palette[closest_idx]
            
            # Reshape back to image dimensions
            result = mapped_pixels.reshape(target_image.shape).astype(np.uint8)
            
            self.logger.success(f"Mapped colors for {len(target_pixels)} pixels")
            return result
    
    def process(self, master_path: str, target_path: str, k_colors: int = 8) -> str:
        """
        Main processing method - applies palette mapping algorithm.
        
        Args:
            master_path: Path to master image (source of color palette)
            target_path: Path to target image (will be color-matched)
            k_colors: Number of colors in palette (4-32)
            
        Returns:
            Path to result image file
            
        Raises:
            FileNotFoundError: If input images don't exist
            ValueError: If k_colors is out of valid range
            RuntimeError: If processing fails
        """
        with self.profiler.profile_operation(f"{self.algorithm_id}_process"):
            # Set algorithm context for logging
            self.logger.set_algorithm_context(self.algorithm_id)
            
            # Validate parameters
            if not (4 <= k_colors <= 32):
                raise ValueError(f"k_colors must be between 4 and 32, got {k_colors}")
            
            if not os.path.exists(master_path):
                raise FileNotFoundError(f"Master image not found: {master_path}")
                
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Target image not found: {target_path}")
            
            self.logger.info(f"Starting palette mapping: k={k_colors}")
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
                
                # Extract palette from master image
                master_palette = self.extract_palette(master_image, k_colors)
                
                # Map target image to master palette
                result_image = self.map_colors(target_image, master_palette, k_colors)
                
                # Save result
                result_path = get_result_path(os.path.basename(target_path))
                success = cv2.imwrite(result_path, result_image)
                
                if not success:
                    raise RuntimeError(f"Failed to save result image: {result_path}")
                
                self.logger.success(f"Palette mapping completed: {result_path}")
                return result_path
                
            except Exception as e:
                self.logger.error(f"Palette mapping failed: {str(e)}", exc_info=True)
                raise RuntimeError(f"Algorithm processing failed: {str(e)}") from e
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information for monitoring and documentation."""
        return {
            'algorithm_id': self.algorithm_id,
            'name': 'Palette Mapping',
            'description': 'K-means based color palette extraction and mapping',
            'version': '2.0.0',
            'parameters': self.default_params.copy(),
            'supported_formats': ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'],
            'color_space': 'RGB',
            'complexity': 'O(n*k*iterations)',
            'memory_usage': 'O(n + k)'
        }


# Factory function for easy algorithm creation
def create_palette_mapping_algorithm() -> PaletteMappingAlgorithm:
    """Create and return a new palette mapping algorithm instance."""
    return PaletteMappingAlgorithm()


# Legacy compatibility function
def simple_palette_mapping(master_path: str, target_path: str, k_colors: int = 8) -> str:
    """
    Legacy compatibility function for existing API.
    
    This maintains backward compatibility with existing code while using
    the new modular algorithm implementation.
    """
    algorithm = create_palette_mapping_algorithm()
    return algorithm.process(master_path, target_path, k_colors)
