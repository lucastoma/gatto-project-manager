"""
Color difference and histogram matching metrics for LAB Color Transfer.
"""
import numpy as np
from skimage.color import deltaE_ciede2000
from skimage.exposure import match_histograms
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
    """Matches the histogram of the source image to the target image for specified channels
    using skimage.exposure.match_histograms for robustness and performance.
    
    Args:
        source: Source image (H x W x 3) in LAB color space.
        target: Target image (H x W x 3) in LAB color space.
        channels: List of channels to match (e.g., ['L', 'a', 'b']). 
                  Defaults to ['L', 'a', 'b'] if None.

    Returns:
        The source image with histograms matched to the target for the specified channels.
    """
    if channels is None:
        channels = ['L', 'a', 'b']  # Default to all LAB channels

    channel_map = {'L': 0, 'a': 1, 'b': 2}
    matched_image = np.copy(source)

    for channel_name in channels:
        if channel_name not in channel_map:
            # Optionally, log a warning or raise an error for invalid channel names
            continue

        idx = channel_map[channel_name]
        
        # Ensure the channel exists in the source and target
        if source.shape[2] <= idx or target.shape[2] <= idx:
            # Optionally, log a warning or raise an error
            continue

        source_ch = source[..., idx]
        target_ch = target[..., idx]
        
        # match_histograms expects 2D images or 3D with multichannel=True
        # We are processing channel by channel, so they are 2D.
        matched_channel = match_histograms(source_ch, target_ch, channel_axis=None) # Explicitly set channel_axis
        matched_image[..., idx] = matched_channel
    
    return matched_image
