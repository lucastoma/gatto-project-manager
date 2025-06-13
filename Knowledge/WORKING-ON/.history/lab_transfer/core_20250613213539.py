import os
import numpy as np
from PIL import Image
import skimage.color
from functools import lru_cache
from typing import Optional, Dict

from .config import LABTransferConfig
from .metrics import calculate_delta_e_lab
from .logger import get_logger

class LABColorTransfer:
    """
    Base class implementing core LAB color transfer methods.
    """
    def __init__(self, config: LABTransferConfig = None):
        self.logger = get_logger()
        self.config = config or LABTransferConfig()

    def rgb_to_lab_optimized(self, rgb_array: np.ndarray) -> np.ndarray:
        """
        Convert an RGB image array to LAB color space with caching.
        """
        from functools import lru_cache
        
        @lru_cache(maxsize=32)
        def _rgb_to_lab_cached(rgb_bytes):
            rgb_array = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(self._last_rgb_shape)
            return skimage.color.rgb2lab(rgb_array)

        # Store shape to reshape bytes back to array
        self._last_rgb_shape = rgb_array.shape
        # Convert to bytes for caching (lru_cache requires hashable arguments)
        rgb_bytes = rgb_array.astype(np.uint8).tobytes()
        return _rgb_to_lab_cached(rgb_bytes)

    def lab_to_rgb_optimized(self, lab_array: np.ndarray) -> np.ndarray:
        """
        Convert a LAB image array back to RGB color space.
        """
        rgb_result = skimage.color.lab2rgb(lab_array)
        # Convert to 0-255 range and uint8 type
        return (rgb_result * 255).astype(np.uint8)

    def basic_lab_transfer(self, source_lab, target_lab):
        """Basic statistical LAB transfer."""
        # Ensure 2D arrays are treated as single-channel
        if len(source_lab.shape) == 2:
            source_lab = source_lab[..., np.newaxis]
            target_lab = target_lab[..., np.newaxis]
            
        result = np.copy(source_lab).astype(np.float64)
        for i in range(source_lab.shape[-1]):  # Handle variable channels
            source_channel = source_lab[..., i]
            target_channel = target_lab[..., i]
            source_mean = np.mean(source_channel)
            source_std = np.std(source_channel)
            target_mean = np.mean(target_channel)
            target_std = np.std(target_channel)
            
            if source_std > 0:
                result[..., i] = (source_channel - source_mean) * (target_std / source_std)
            result[..., i] = result[..., i] + target_mean
        return result.squeeze()

    def weighted_lab_transfer(self, source: np.ndarray, target: np.ndarray,
                            l_weight: float = None, a_weight: float = None, 
                            b_weight: float = None, weights: Dict[str, float] = None) -> np.ndarray:
        """
        Performs LAB color transfer with user-defined weights for L, a, and b channels.
        Can accept weights as either individual parameters or as a dictionary.

        Args:
            source: Source image in LAB space (H x W x 3).
            target: Target image in LAB space (H x W x 3).
            l_weight: Weight for the L channel (0-1).
            a_weight: Weight for the a channel (0-1).
            b_weight: Weight for the b channel (0-1).
            weights: Dictionary of weights {'L': float, 'a': float, 'b': float}.
                    Overrides individual weight parameters if provided.

        Returns:
            Transferred image in LAB space.

        Raises:
            ValueError: If weights are invalid or don't sum to 1.0
        """
        # Handle dictionary weights if provided
        if weights is not None:
            if not isinstance(weights, dict):
                raise ValueError("Weights must be provided as a dictionary")
            l_weight = weights.get('L', 0.0)
            a_weight = weights.get('a', 0.0)
            b_weight = weights.get('b', 0.0)
        else:
            # Use individual weights with defaults if not provided
            if l_weight is None: l_weight = 0.5
            if a_weight is None: a_weight = 0.25
            if b_weight is None: b_weight = 0.25

        # Input validation: weights must be numeric and sum to 1.0
        try:
            weight_sum = float(l_weight) + float(a_weight) + float(b_weight)
            if not np.isclose(weight_sum, 1.0, rtol=1e-5):
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        except (TypeError, ValueError) as e:
            raise ValueError("Weights must be numeric values") from e

        # Ensure arrays are float for calculations
        source_lab = source.astype(np.float64)
        target_lab = target.astype(np.float64)
        
        result = np.zeros_like(source_lab)
        
        # Apply weighted transfer for each channel with respective weights
        result[..., 0] = source_lab[..., 0] * (1 - l_weight) + target_lab[..., 0] * l_weight
        result[..., 1] = source_lab[..., 1] * (1 - a_weight) + target_lab[..., 1] * a_weight
        result[..., 2] = source_lab[..., 2] * (1 - b_weight) + target_lab[..., 2] * b_weight
        
        return result

    def selective_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """Transfer only a and b channels, preserving source L channel."""
        result = np.copy(source_lab)
        
        # Ensure arrays are at least 3D
        if source_lab.ndim == 1:
            source_lab = source_lab[np.newaxis, np.newaxis, :]
        if target_lab.ndim == 1:
            target_lab = target_lab[np.newaxis, np.newaxis, :]
            
        # Only transfer a and b channels
        result[..., 1:] = self.basic_lab_transfer(source_lab, target_lab)[..., 1:]
        return result

    def adaptive_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """Adaptive transfer based on luminance segmentation."""
        result = np.copy(source_lab)
        
        # Segment based on luminance
        l_channel = source_lab[..., 0] if source_lab.shape[-1] == 3 else source_lab
        thresholds = np.percentile(l_channel, [33, 66])
        
        # Apply different transfers per segment
        for i, (low, high) in enumerate(zip([0] + thresholds.tolist(), thresholds.tolist() + [100])):
            mask = (l_channel >= low) & (l_channel <= high)
            if np.any(mask):
                # Process each channel separately to maintain shape
                for c in range(source_lab.shape[-1]):
                    src = source_lab[..., c][mask]
                    tgt = target_lab[..., c][mask]
                    result[..., c][mask] = self.basic_lab_transfer(src, tgt)
        
        return result

    def blend_tile_overlap(self, tile: np.ndarray, overlap_size: int = 32) -> np.ndarray:
        """
        Apply linear alpha blending to tile edges to smooth overlaps.

        Args:
            tile: Input tile (H x W x 3)
            overlap_size: Size of overlap region to blend

        Returns:
            Blended tile with smoothed edges
        """
        if overlap_size <= 0:
            return tile.copy()

        # Convert to float for blending if needed
        if tile.dtype == np.uint8:
            blended = tile.astype(np.float32) / 255.0
        else:
            blended = tile.copy()
        height, width = tile.shape[:2]

        # Create alpha gradient for blending
        alpha = np.linspace(0, 1, overlap_size)

        # Always blend all edges if possible, even for small tiles
        # For small tiles, blend all available pixels
        if width >= overlap_size:
            blended[:, :overlap_size] *= alpha[np.newaxis, :, np.newaxis]
            blended[:, -overlap_size:] *= alpha[::-1][np.newaxis, :, np.newaxis]
        else:
            # If tile is smaller than overlap, blend across the whole width
            alpha_w = np.linspace(0, 1, width)
            blended *= alpha_w[np.newaxis, :, np.newaxis]

        if height >= overlap_size:
            blended[:overlap_size, :] *= alpha[:, np.newaxis, np.newaxis]
            blended[-overlap_size:, :] *= alpha[::-1][:, np.newaxis, np.newaxis]
        else:
            # If tile is smaller than overlap, blend across the whole height
            alpha_h = np.linspace(0, 1, height)
            blended *= alpha_h[:, np.newaxis, np.newaxis]

        # Convert back to original type
        if tile.dtype == np.uint8:
            blended = (blended * 255).clip(0, 255).astype(np.uint8)
        return blended

    def process_large_image(self, source: np.ndarray, target: np.ndarray, 
                          method: str = 'basic', tile_size: int = 256, 
                          overlap: int = 32) -> np.ndarray:
        """
        Process large image by tiling and applying color transfer with blending.
        
        Args:
            source: Source image (H x W x 3)
            target: Target image (H x W x 3)
            method: Transfer method ('basic', 'weighted', 'selective', 'adaptive')
            tile_size: Size of processing tiles
            overlap: Overlap size between tiles
            
        Returns:
            Result image after color transfer
        """
        if method not in ['basic', 'weighted', 'selective', 'adaptive']:
            raise ValueError(f"Invalid method '{method}'. Must be one of: basic, weighted, selective, adaptive")
            
        height, width = source.shape[:2]
        result = np.zeros_like(source)
        
        # Process in tiles with overlap
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Get source and target tiles
                src_tile = source[y:y+tile_size, x:x+tile_size]
                tgt_tile = target[y:y+tile_size, x:x+tile_size]
                
                # Apply selected transfer method
                if method == 'basic':
                    transfer_tile = self.basic_lab_transfer(src_tile, tgt_tile)
                elif method == 'weighted':
                    transfer_tile = self.weighted_lab_transfer(src_tile, tgt_tile)
                elif method == 'selective':
                    transfer_tile = self.selective_lab_transfer(src_tile, tgt_tile)
                else:  # adaptive
                    transfer_tile = self.adaptive_lab_transfer(src_tile, tgt_tile)
                
                # Blend tile edges
                blended_tile = self.blend_tile_overlap(transfer_tile, overlap_size=overlap)
                
                # Paste into result
                result[y:y+tile_size, x:x+tile_size] = blended_tile
                
        return result
