"""
Core classes for LAB Color Transfer algorithm.
"""
import os
import numpy as np
from PIL import Image
import skimage.color
from functools import lru_cache

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

    def basic_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Perform basic statistical LAB transfer (match mean and std).
        """
        result_lab = np.copy(source_lab).astype(np.float64)
        
        for i in range(3): # Iterate over L, a, b channels
            source_channel = source_lab[:, :, i]
            target_channel = target_lab[:, :, i]
            
            source_mean = np.mean(source_channel)
            source_std = np.std(source_channel)
            target_mean = np.mean(target_channel)
            target_std = np.std(target_channel)
            
            # Apply transformation
            # Avoid division by zero if std is very small
            if source_std == 0:
                result_lab[:, :, i] = target_mean
            else:
                result_lab[:, :, i] = (source_channel - source_mean) * (target_std / source_std) + target_mean
                
        return result_lab

    def weighted_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray, weights: dict = None) -> np.ndarray:
        """
        Perform weighted LAB transfer using channel-specific weights.
        """
        result_lab = np.copy(source_lab).astype(np.float64)
        
        # Default weights if none are provided
        if weights is None:
            weights = {'L': 1.0, 'a': 1.0, 'b': 1.0}

        channel_map = {0: 'L', 1: 'a', 2: 'b'}

        for i in range(3): # Iterate over L, a, b channels
            channel_name = channel_map[i]
            weight = weights.get(channel_name, 1.0) # Get weight, default to 1.0

            source_channel = source_lab[:, :, i]
            target_channel = target_lab[:, :, i]
            
            source_mean = np.mean(source_channel)
            source_std = np.std(source_channel)
            target_mean = np.mean(target_channel)
            target_std = np.std(target_channel)
            
            # Apply transformation with weighting
            if source_std == 0:
                transferred_channel = np.full_like(source_channel, target_mean)
            else:
                transferred_channel = (source_channel - source_mean) * (target_std / source_std) + target_mean
            
            # Blend the transferred channel with the original based on weight
            result_lab[:, :, i] = source_channel * (1.0 - weight) + transferred_channel * weight
                
        return result_lab

    def selective_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Perform selective LAB transfer (e.g., only chromaticity).
        """
        result_lab = np.copy(source_lab).astype(np.float64)
        
        # Preserve L channel from source
        result_lab[:, :, 0] = source_lab[:, :, 0]
        
        # Apply basic transfer to 'a' and 'b' channels
        for i in range(1, 3): # Iterate over a, b channels
            source_channel = source_lab[:, :, i]
            target_channel = target_lab[:, :, i]
            
            source_mean = np.mean(source_channel)
            source_std = np.std(source_channel)
            target_mean = np.mean(target_channel)
            target_std = np.std(target_channel)
            
            if source_std == 0:
                result_lab[:, :, i] = target_mean
            else:
                result_lab[:, :, i] = (source_channel - source_mean) * (target_std / source_std) + target_mean
                
        return result_lab

    def adaptive_lab_transfer(self, source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        """
        Perform adaptive LAB transfer based on image content.
        """
        result_lab = np.copy(source_lab).astype(np.float64)
        
        # Define luminance ranges (example: 3 ranges)
        luminance_ranges = [
            (0, 33),  # Darks
            (34, 66), # Mids
            (67, 100) # Lights
        ]

        for min_l, max_l in luminance_ranges:
            # Create a mask for the current luminance range
            mask = (source_lab[:, :, 0] >= min_l) & (source_lab[:, :, 0] <= max_l)
            
            # Apply basic LAB transfer only to the masked region
            # Extract relevant parts of source and target LAB arrays
            source_lab_masked = source_lab[mask]
            target_lab_masked = target_lab[mask]

            if source_lab_masked.size == 0 or target_lab_masked.size == 0:
                continue # Skip if no pixels in this range

            # Perform basic transfer for the masked region
            # This is a simplified approach, a more robust solution might involve
            # calculating statistics for the target image's corresponding luminance range
            # For now, we'll use the overall target stats for simplicity
            
            for i in range(3): # Iterate over L, a, b channels
                source_channel = source_lab_masked[:, i]
                target_channel = target_lab_masked[:, i]
                
                source_mean = np.mean(source_channel)
                source_std = np.std(source_channel)
                target_mean = np.mean(target_channel)
                target_std = np.std(target_channel)
                
                if source_std == 0:
                    transferred_channel_part = np.full_like(source_channel, target_mean)
                else:
                    transferred_channel_part = (source_channel - source_mean) * (target_std / source_std) + target_mean
                
                # Assign back to the result_lab using the mask
                result_lab[:, :, i][mask] = transferred_channel_part
                
        return result_lab

    def blend_tile_overlap(self, tile_array: np.ndarray, result_array: np.ndarray, x: int, y: int, overlap: int) -> np.ndarray:
        """
        Blend overlapping tiles in a large image processing context.
        """
        # Pobierz istniejący fragment z obrazu wynikowego
        h, w, _ = tile_array.shape
        
        # Blending pionowy (jeśli jest overlap z góry)
        if y > 0:
            # Ensure the slice is within bounds for result_array
            top_overlap_region = result_array[y : y + overlap, x : x + w]
            for i in range(overlap):
                alpha = i / (overlap - 1) if overlap > 1 else 0.5 # waga od 0 do 1
                tile_array[i, :] = (1 - alpha) * top_overlap_region[i, :] + alpha * tile_array[i, :]

        # Blending poziomy (jeśli jest overlap z lewej)
        if x > 0:
            # Ensure the slice is within bounds for result_array
            left_overlap_region = result_array[y : y + h, x : x + overlap]
            for i in range(overlap):
                alpha = i / (overlap - 1) if overlap > 1 else 0.5
                tile_array[:, i] = (1 - alpha) * left_overlap_region[:, i] + alpha * tile_array[:, i]
        
        return tile_array

    def process_large_image(self, source_path: str, target_path: str, output_path: str,
                             tile_size: int = 512, overlap: int = 64) -> None:
        """
        Process a large image by tiling, applying LAB transfer, and blending overlaps.
        """
        # Wczytaj target
        target_image = Image.open(target_path).convert('RGB')
        target_lab = self.rgb_to_lab_optimized(np.array(target_image))
        
        # Otwórz source image
        source_image = Image.open(source_path).convert('RGB')
        width, height = source_image.size
        
        # Utwórz output image
        result_image_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Przetwarzaj w kafelkach
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Wytnij kafelek
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                
                tile = source_image.crop((x, y, x_end, y_end))
                tile_array = np.array(tile)
                
                # Przetwórz kafelek
                tile_lab = self.rgb_to_lab_optimized(tile_array)
                result_lab = self.basic_lab_transfer(tile_lab, target_lab) # Using basic for now, can be dynamic
                result_tile_rgb = self.lab_to_rgb_optimized(result_lab)
                
                # Apply blending if there's an overlap
                if overlap > 0 and (x > 0 or y > 0):
                    # Create a temporary array for the current tile's region in the result_image_array
                    # This is needed because blend_tile_overlap expects result_array to be the full image
                    current_result_region = result_image_array[y:y_end, x:x_end]
                    result_tile_rgb = self.blend_tile_overlap(
                        result_tile_rgb, current_result_region, 0, 0, overlap # x, y are 0,0 for the tile's internal coords
                    )
                
                # Paste the processed tile into the result image array
                result_image_array[y:y_end, x:x_end] = result_tile_rgb[:y_end-y, :x_end-x]

        # Save the final image
        Image.fromarray(result_image_array).save(output_path)

    def calculate_delta_e(self, lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
        """
        Calculate perceptual color difference (CIEDE2000).
        """
        return calculate_delta_e_lab(lab1, lab2)
