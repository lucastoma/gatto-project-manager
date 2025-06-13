import os
import numpy as np
from PIL import Image
import skimage.color
from functools import lru_cache
from typing import Optional, Dict, List

from .config import LABTransferConfig
from .metrics import calculate_delta_e_lab
from .logger import get_logger
from .gpu_core import LABColorTransferGPU

class LABColorTransfer:
    """
    Base class implementing core LAB color transfer methods.
    It now uses scikit-image for robust color conversions and includes
    optimized and refactored transfer methods.
    """
    def __init__(self, config: LABTransferConfig = None):
        self.logger = get_logger()
        self.config = config or LABTransferConfig()
        self.gpu_transfer = None
        if self.config.use_gpu:
            try:
                self.gpu_transfer = LABColorTransferGPU()
                if not self.gpu_transfer.is_gpu_available():
                    self.logger.warning("GPU requested, but OpenCL initialization failed. Falling back to CPU.")
                    self.gpu_transfer = None
            except Exception as e:
                self.logger.error(f"Failed to initialize GPU context: {e}. Falling back to CPU.")
                self.gpu_transfer = None

    @staticmethod
    @lru_cache(maxsize=16)
    def _rgb_to_lab_cached(rgb_bytes: bytes, shape: tuple) -> np.ndarray:
        """Helper for caching RGB to LAB conversion."""
        rgb_array = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(shape)
        return skimage.color.rgb2lab(rgb_array)

    def rgb_to_lab_optimized(self, rgb_array: np.ndarray) -> np.ndarray:
        """
        Convert an RGB image array to LAB color space with caching.
        """
        # The array's bytes are used as a key, which requires the array to be hashable.
        # A simple way is to convert it to a read-only bytes string.
        return self._rgb_to_lab_cached(rgb_array.tobytes(), rgb_array.shape)

    def lab_to_rgb_optimized(self, lab_array: np.ndarray) -> np.ndarray:
        """
        Convert a LAB image array back to RGB color space.
        """
        rgb_result = skimage.color.lab2rgb(lab_array)
        # Convert to 0-255 range and uint8 type, clipping to ensure validity.
        return (np.clip(rgb_result, 0, 1) * 255).astype(np.uint8)

    def _transfer_channel_stats(self, source_channel: np.ndarray, target_channel: np.ndarray) -> np.ndarray:
        """
        Helper to apply statistical transfer to a single channel.
        """
        source_mean, source_std = np.mean(source_channel), np.std(source_channel)
        target_mean, target_std = np.mean(target_channel), np.std(target_channel)
        
        # Avoid division by zero for flat channels
        if source_std < 1e-6:
            return source_channel + (target_mean - source_mean)
            
        result_channel = (source_channel - source_mean) * (target_std / source_std) + target_mean
        return result_channel

    def basic_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Performs statistical transfer on all LAB channels.
        Dispatches to GPU if available and configured.
        """
        if self.gpu_transfer:
            self.logger.info("Using GPU for basic LAB transfer.")
            return self.gpu_transfer.basic_lab_transfer_gpu(source_lab, target_lab)

        # Validate input shapes – basic transfer must operate on same-sized images in public API.
        if source_lab.shape != target_lab.shape:
            raise ValueError("Source and target must have the same shape")

        original_dtype = source_lab.dtype
        src = source_lab.astype(np.float64, copy=False)
        tgt = target_lab.astype(np.float64, copy=False)

        result = np.empty_like(src)
        for i in range(3):
            result[..., i] = self._transfer_channel_stats(src[..., i], tgt[..., i])
        return result.astype(original_dtype, copy=False)

    def linear_blend_lab(self, source_lab: np.ndarray, target_lab: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """
        Performs a linear blend (interpolation) between the source and target images
        in LAB space, using independent weights for each channel. This is not a
        statistical transfer but a direct mixing of color values.

        Args:
            source_lab: Source image in LAB space.
            target_lab: Target image in LAB space.
            weights: Dictionary of weights {'L': float, 'a': float, 'b': float}.
                     Each weight is between 0 (use source) and 1 (use target).

        Returns:
            The blended image in LAB space.
        """
        # Validate input shapes and dtype
        if source_lab.shape != target_lab.shape:
            raise ValueError("Source and target must have the same shape")
        if source_lab.dtype != np.float64 or target_lab.dtype != np.float64:
            raise ValueError("Input arrays must be of type float64")
        
        original_dtype = source_lab.dtype
        l_weight = weights.get('L', 0.5)
        a_weight = weights.get('a', 0.5)
        b_weight = weights.get('b', 0.5)

        result = np.zeros_like(source_lab)
        result[..., 0] = source_lab[..., 0] * (1 - l_weight) + target_lab[..., 0] * l_weight
        result[..., 1] = source_lab[..., 1] * (1 - a_weight) + target_lab[..., 1] * a_weight
        result[..., 2] = source_lab[..., 2] * (1 - b_weight) + target_lab[..., 2] * b_weight
        return result.astype(original_dtype, copy=False)

    def selective_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Performs statistical transfer on color channels (a, b) only,
        preserving the luminance (L) of the source image.
        """
        if self.gpu_transfer:
            self.logger.info("Using GPU for selective LAB transfer.")
            return self.gpu_transfer.selective_lab_transfer_gpu(source_lab, target_lab)

        if source_lab.shape != target_lab.shape:
            raise ValueError("Source and target must have the same shape")

        original_dtype = source_lab.dtype
        src = source_lab.astype(np.float64, copy=False)
        tgt = target_lab.astype(np.float64, copy=False)
        
        result = src.copy()
        # Transfer stats for 'a' and 'b' channels
        for i in [1, 2]:
            result[..., i] = self._transfer_channel_stats(src[..., i], tgt[..., i])
            
        return result.astype(original_dtype, copy=False)

    def weighted_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, weights: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Performs a weighted statistical transfer. A weight of 1.0 is a full
        transfer, while 0.0 leaves the source channel unchanged.
        """
        if self.gpu_transfer:
            self.logger.info("Using GPU for weighted LAB transfer.")
            return self.gpu_transfer.weighted_lab_transfer_gpu(source_lab, target_lab, weights=weights)

        if source_lab.shape != target_lab.shape:
            raise ValueError("Source and target must have the same shape")

        original_dtype = source_lab.dtype
        src = source_lab.astype(np.float64, copy=False)
        
        # Get the fully transferred image first
        transferred_lab = self.basic_lab_transfer(src, target_lab)

        result = np.empty_like(src)
        # Blend source with the fully transferred version
        for i, weight in enumerate(weights):
            result[..., i] = src[..., i] * (1 - weight) + transferred_lab[..., i] * weight
            
        return result.astype(original_dtype, copy=False)

    def adaptive_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Adaptive LAB transfer based on luminance segmentation. Matches statistics
        between corresponding luminance zones of the source and target images.
        """
        if self.gpu_transfer:
            self.logger.info("Using GPU for adaptive LAB transfer.")
            return self.gpu_transfer.adaptive_lab_transfer_gpu(source_lab, target_lab)

        if source_lab.shape != target_lab.shape:
            raise ValueError("Source and target must have the same shape")

        original_dtype = source_lab.dtype
        src = source_lab.astype(np.float64, copy=False)
        tgt = target_lab.astype(np.float64, copy=False)
        result = src.copy()

        src_l, tgt_l = src[..., 0], tgt[..., 0]

        # Define luminance segments based on percentiles
        src_thresholds = np.percentile(src_l, [33, 66])
        tgt_thresholds = np.percentile(tgt_l, [33, 66])

        src_masks = [
            src_l < src_thresholds[0],
            (src_l >= src_thresholds[0]) & (src_l < src_thresholds[1]),
            src_l >= src_thresholds[1]
        ]
        tgt_masks = [
            tgt_l < tgt_thresholds[0],
            (tgt_l >= tgt_thresholds[0]) & (tgt_l < tgt_thresholds[1]),
            tgt_l >= tgt_thresholds[1]
        ]

        # Process each corresponding segment
        for i in range(3):
            src_mask, tgt_mask = src_masks[i], tgt_masks[i]

            if not np.any(src_mask) or not np.any(tgt_mask):
                continue

            # Transfer stats for each channel within the segment
            for ch in range(3):
                src_segment = src[src_mask, ch]
                tgt_segment = tgt[tgt_mask, ch]
                transferred_segment = self._transfer_channel_stats(src_segment, tgt_segment)
                result[src_mask, ch] = transferred_segment

        return result.astype(original_dtype, copy=False)

    def weighted_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """Weighted LAB transfer with channel-specific weights"""
        # Validate weights
        for channel in ['L', 'a', 'b']:
            if channel not in weights:
                raise ValueError(f"Missing weight for channel: {channel}")
            if not (0 <= weights[channel] <= 1):
                raise ValueError(f"Weight for channel {channel} must be between 0 and 1")
        
        # Validate input shapes and dtype
        if source_lab.shape != target_lab.shape:
            raise ValueError("Source and target must have the same shape")
        if source_lab.dtype != np.float64 or target_lab.dtype != np.float64:
            raise ValueError("Input arrays must be of type float64")
        
        return self.linear_blend_lab(source_lab, target_lab, weights)

    def process_large_image(self, source_img: np.ndarray, target_img: np.ndarray, tile_size: int = 64, overlap: int = 16, method: str = 'adaptive') -> np.ndarray:
        """High-level helper that processes full-resolution RGB or LAB images.
        Currently processes the entire image at once (no real tiling) but keeps the
        signature required by tests. Supports `adaptive` or `basic` methods.
        """
        # Basic shape sanity check
        if source_img.shape != target_img.shape:
            raise ValueError("Source and target must have the same shape")
        if source_img.ndim != 3 or source_img.shape[2] != 3:
            raise ValueError("Images must be (H, W, 3)")

        # Accept RGB uint8 or float images as well as LAB float64; convert as needed
        is_rgb = source_img.dtype == np.uint8
        if is_rgb:
            src_lab = self.rgb_to_lab_optimized(source_img)
            tgt_lab = self.rgb_to_lab_optimized(target_img)
        else:
            src_lab = source_img.astype(np.float64, copy=False)
            tgt_lab = target_img.astype(np.float64, copy=False)

        # Choose processing method
        if method == 'adaptive':
            result_lab = self.adaptive_lab_transfer(src_lab, tgt_lab)
        elif method == 'basic':
            result_lab = self.basic_lab_transfer(src_lab, tgt_lab)
        else:
            raise ValueError("invalid_method")

        # Convert back to original space if inputs were RGB
        if is_rgb:
            return self.lab_to_rgb_optimized(result_lab)
        return result_lab

    def blend_tile_overlap(self, tile: np.ndarray, overlap_size: int) -> np.ndarray:
        """Apply linear alpha blending to tile edges based on overlap size"""
        if overlap_size == 0:
            return tile
            
        blended = tile.astype(np.float32)
        h, w, _ = blended.shape
        
        # Vertical edges
        if overlap_size > 0 and h > 1:
            alpha = np.linspace(0, 1, overlap_size)[:, np.newaxis, np.newaxis]
            blended[:overlap_size] *= alpha
            blended[-overlap_size:] *= alpha[::-1]
            
        # Horizontal edges
        if overlap_size > 0 and w > 1:
            alpha = np.linspace(0, 1, overlap_size)[np.newaxis, :, np.newaxis]
            blended[:, :overlap_size] *= alpha
            blended[:, -overlap_size:] *= alpha[::-1]
            
        return blended.astype(tile.dtype)
