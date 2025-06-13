"""
Configuration module for LAB Color Transfer algorithm.
"""
from typing import Dict, List, Optional

class LABTransferConfig:
    """
    Configuration for LAB Color Transfer, defining methods and parameters.
    """
    def __init__(
        self,
        method: str = 'basic',
        channel_weights: Optional[Dict[str, float]] = None,
        selective_channels: Optional[List[str]] = None,
        adaptation_method: str = 'none',
        tile_size: int = 512,
        overlap: int = 64,
        use_gpu: bool = False
    ):
        # Main processing method
        self.method = method

        # Parameters for 'linear_blend' method
        self.channel_weights = channel_weights or {'L': 0.5, 'a': 0.5, 'b': 0.5}
        
        # Parameters for 'selective' method
        self.selective_channels = selective_channels or ['a', 'b']
        
        # Parameters for 'adaptive' method (currently one type)
        self.adaptation_method = adaptation_method

        # Parameters for large image processing
        self.tile_size = tile_size
        self.overlap = overlap

        # GPU acceleration flag
        self.use_gpu = use_gpu

    def validate(self):
        """
        Validates the configuration values and raises ValueError if invalid.
        """
        # Added 'hybrid' and 'linear_blend', removed 'weighted'
        valid_methods = ['basic', 'linear_blend', 'selective', 'adaptive', 'hybrid']
        valid_adapt = ['none', 'luminance'] # Simplified to implemented methods
        errors = []

        if self.method not in valid_methods:
            errors.append(f"Invalid method: '{self.method}'. Must be one of {valid_methods}")

        if self.adaptation_method not in valid_adapt:
            errors.append(f"Invalid adaptation_method: '{self.adaptation_method}'. Must be one of {valid_adapt}")
        
        for ch in self.selective_channels:
            if ch not in ['L', 'a', 'b']:
                errors.append(f"Invalid channel in selective_channels: '{ch}'")
        
        for w in self.channel_weights.values():
            if not (0.0 <= w <= 1.0):
                errors.append(f"Channel weight must be between 0 and 1, but got {w}")

        if errors:
            raise ValueError('Invalid configuration: ' + '; '.join(errors))
