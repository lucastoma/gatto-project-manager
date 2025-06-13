"""
Color difference and histogram matching metrics for LAB Color Transfer.
"""
import numpy as np
from skimage.color import deltaE_ciede2000
from typing import List

def calculate_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Calculate perceptual color difference (CIEDE2000) between two LAB images.
    
    Args:
        lab1: First LAB image (H x W x 3)
        lab2: Second LAB image (H x W x 3)
    Returns:
        Delta E map (H x W)
    """
    # Reshape for scikit-image function if needed, but it handles 3D arrays well.
    return deltaE_ciede2000(lab1, lab2)


def calculate_delta_e_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Alias for calculate_delta_e, for consistency with core API.
    """
    return calculate_delta_e(lab1, lab2)


def histogram_matching(source: np.ndarray, target: np.ndarray, channels: List[str] = None) -> np.ndarray:
    """
    Match the histogram of the source image to the target image for specified channels.
    
    Args:
        source: Source image in LAB space (H x W x 3).
        target: Target image in LAB space (H x W x 3).
        channels: A list of channels to match, e.g., ['L', 'a', 'b'].
                  If None, defaults to matching all channels.
        
    Returns:
        Matched image in LAB space.
    """
    if channels is None:
        channels = ['L', 'a', 'b']

    channel_map = {'L': 0, 'a': 1, 'b': 2}
    matched = np.copy(source).astype(np.float64)

    for channel_name in channels:
        if channel_name not in channel_map:
            continue
            
        idx = channel_map[channel_name]
        
        source_channel = source[..., idx].astype(np.float64)
        target_channel = target[..., idx].astype(np.float64)
        
        source_flat = source_channel.ravel()
        target_flat = target_channel.ravel()
        
        # Get the sorted unique values from the source channel
        source_values, bin_idx, source_counts = np.unique(source_flat, return_inverse=True, return_counts=True)
        
        # Calculate the cumulative distribution functions (CDFs)
        source_cdf = np.cumsum(source_counts).astype(np.float64) / source_flat.size
        
        target_values, target_counts = np.unique(target_flat, return_counts=True)
        target_cdf = np.cumsum(target_counts).astype(np.float64) / target_flat.size
        
        # Interpolate to map the source CDF to the target value range
        interp_values = np.interp(source_cdf, target_cdf, target_values)
        
        # Map the interpolated values back to the original image shape
        mapped_channel = interp_values[bin_idx].reshape(source_channel.shape)
        
        # Clip the values to stay within the valid LAB range
        if channel_name == 'L':
            matched[..., idx] = np.clip(mapped_channel, 0, 100)
        else: # For 'a' and 'b' channels
            matched[..., idx] = np.clip(mapped_channel, -128, 127)

    return matched
