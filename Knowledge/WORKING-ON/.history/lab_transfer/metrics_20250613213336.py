# Color difference metrics implementation

"""
Color difference and histogram matching metrics for LAB Color Transfer.
"""
import numpy as np
from skimage.color import deltaE_ciede2000


def calculate_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Calculate perceptual color difference (CIEDE2000) between two LAB images.
    
    Args:
        lab1: First LAB image (H x W x 3)
        lab2: Second LAB image (H x W x 3)
    Returns:
        Delta E map (H x W)
    """
    lab1_reshaped = lab1.reshape(-1, 3)
    lab2_reshaped = lab2.reshape(-1, 3)
    delta = deltaE_ciede2000(lab1_reshaped, lab2_reshaped)
    return delta.reshape(lab1.shape[:2])


def calculate_delta_e_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Alias for calculate_delta_e, for consistency with core API.
    """
    return calculate_delta_e(lab1, lab2)


def histogram_matching(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Match the histogram of source image to target image in LAB space using quantile mapping.
    
    Args:
        source: Source image in LAB space (H x W x 3)
        target: Target image in LAB space (H x W x 3)
        
    Returns:
        Matched image in LAB space with L channel histogram matched to target
    """
    # Create output array
    matched = np.copy(source).astype(np.float64)
    
    # Only match L channel (preserve a and b)
    source_l = source[..., 0].astype(np.float64)
    target_l = target[..., 0].astype(np.float64)
    
    # Flatten the L channels
    source_flat = source_l.ravel()
    target_flat = target_l.ravel()
    
    # Get all unique source values and their counts
    source_unique, source_inverse = np.unique(source_flat, return_inverse=True)
    
    # Calculate percentiles for source values
    source_percentiles = np.percentile(source_flat, np.linspace(0, 100, len(source_unique)))
    
    # Calculate target values at the same percentiles
    target_values = np.percentile(target_flat, np.linspace(0, 100, len(source_unique)))
    
    # Create mapping from source to target values
    value_map = dict(zip(source_percentiles, target_values))
    
    # Apply the mapping to the source L channel
    matched_l = np.interp(source_flat, 
                         sorted(value_map.keys()), 
                         [value_map[k] for k in sorted(value_map.keys())])
    
    # Reshape and assign back
    matched[..., 0] = matched_l.reshape(source_l.shape)
    
    # Ensure we don't go out of LAB bounds
    matched[..., 0] = np.clip(matched[..., 0], 0, 100)
    
    return matched
