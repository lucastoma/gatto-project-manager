import os
import numpy as np
from PIL import Image
import skimage.color
from functools import lru_cache
from typing import Optional, Dict, List

from .config import LABTransferConfig
from .metrics import calculate_delta_e_lab
from .logger import get_logger

class LABColorTransfer:
    """
    Base class implementing core LAB color transfer methods.
    It now uses scikit-image for robust color conversions and includes
    optimized and refactored transfer methods.
    """
    def __init__(self, config: LABTransferConfig = None):
        self.logger = get_logger()
        self.config = config or LABTransferConfig()

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
        """
        result = np.copy(source_lab)
        for i in range(3):
            result[..., i] = self._transfer_channel_stats(source_lab[..., i], target_lab[..., i])
        return result

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
        l_weight = weights.get('L', 0.5)
        a_weight = weights.get('a', 0.5)
        b_weight = weights.get('b', 0.5)

        result = np.zeros_like(source_lab)
        result[..., 0] = source_lab[..., 0] * (1 - l_weight) + target_lab[..., 0] * l_weight
        result[..., 1] = source_lab[..., 1] * (1 - a_weight) + target_lab[..., 1] * a_weight
        result[..., 2] = source_lab[..., 2] * (1 - b_weight) + target_lab[..., 2] * b_weight
        return result

    def selective_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, channels: List[str]) -> np.ndarray:
        """
        Performs statistical transfer on a selected list of channels, preserving the others.
        This is an optimized version that only computes what's necessary.
        """
        result = np.copy(source_lab)
        channel_map = {'L': 0, 'a': 1, 'b': 2}
        for channel_name in channels:
            if channel_name in channel_map:
                idx = channel_map[channel_name]
                result[..., idx] = self._transfer_channel_stats(source_lab[..., idx], target_lab[..., idx])
        return result

    def adaptive_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Adaptive transfer based on luminance segmentation. This improved version uses
        statistics from the entire target image for each source segment, making the
        transfer more robust and predictable.
        """
        result = np.copy(source_lab)
        l_channel = source_lab[..., 0]
        
        # Define luminance segments based on percentiles of the source image
        thresholds = np.percentile(l_channel, [33, 66])
        segments = [(0, thresholds[0]), (thresholds[0], thresholds[1]), (thresholds[1], 100)]

        for low, high in segments:
            mask = (l_channel >= low) & (l_channel < high)
            if np.any(mask):
                # Apply basic transfer to the masked region of the source,
                # using stats from the *entire* target image for stability.
                segment_transfer = self.basic_lab_transfer(source_lab[mask], target_lab)
                result[mask] = segment_transfer
        
        return result

    def blend_tile_overlap(self, tile: np.ndarray, result_so_far: np.ndarray, x: int, y: int, overlap: int) -> np.ndarray:
        """
        Blends a new tile with the existing result on the overlapping region.
        """
        if overlap <= 0:
            return tile

        h, w, _ = tile.shape
        blended = tile.astype(np.float32)
        
        # Vertical blending
        if y > 0:
            top_overlap_data = result_so_far[y : y + overlap, x : x + w].astype(np.float32)
            alpha_y = np.linspace(0, 1, overlap)[:, np.newaxis, np.newaxis]
            blended[:overlap, :] = top_overlap_data * (1 - alpha_y) + blended[:overlap, :] * alpha_y

        # Horizontal blending
        if x > 0:
            left_overlap_data = result_so_far[y : y + h, x : x + overlap].astype(np.float32)
            alpha_x = np.linspace(0, 1, overlap)[np.newaxis, :, np.newaxis]
            blended[:, :overlap] = left_overlap_data * (1 - alpha_x) + blended[:, :overlap] * alpha_x

        return blended.astype(np.uint8)

