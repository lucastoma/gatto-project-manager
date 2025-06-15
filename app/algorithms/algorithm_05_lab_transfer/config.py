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
        blend_factor: float = 0.5,
        adaptation_method: str = 'none',
        num_segments: int = 16,
        delta_e_threshold: float = 12.0,
        min_segment_size_perc: float = 0.01,
        tile_size: int = 512,
        overlap: int = 64,
        use_gpu: bool = False,
        hybrid_pipeline: Optional[List[Dict]] = None # Added hybrid_pipeline
    ):
        # Main processing method
        self.method = method

        # Parameters for 'linear_blend' method
        self.channel_weights = channel_weights or {'L': 0.5, 'a': 0.5, 'b': 0.5}
        
        # Parameters for 'selective' method
        self.selective_channels = selective_channels or ['a', 'b']
        self.blend_factor = blend_factor

        # Parameters for 'adaptive' method
        self.adaptation_method = adaptation_method
        self.num_segments = num_segments
        self.delta_e_threshold = delta_e_threshold
        self.min_segment_size_perc = min_segment_size_perc

        # Parameters for large image processing
        self.tile_size = tile_size
        self.overlap = overlap

        # GPU acceleration flag
        self.use_gpu = use_gpu

        # Parameters for 'hybrid' method
        if hybrid_pipeline is None:
            self.hybrid_pipeline = [
                {"method": "adaptive", "params": {"num_segments": 8}},
                {"method": "selective",
                 "params": {
                     "mask": None, 
                     "blend_factor": 0.7,
                     "selective_channels": ["a", "b"]
                 }},
                {"method": "linear_blend",
                 "params": {"weights": [0.3, 0.5, 0.5]}}
            ]
        else:
            self.hybrid_pipeline = hybrid_pipeline

    def validate(self):
        """
        Validates the configuration values and raises ValueError if invalid.
        """
        # Added 'hybrid' and 'linear_blend', removed 'weighted'
        valid_methods = ['basic', 'linear_blend', 'selective', 'adaptive', 'hybrid']
        valid_adapt = ['none', 'luminance']  # Simplified to implemented methods
        errors = []

        if self.method not in valid_methods:
            errors.append(f"Invalid method: '{self.method}'. Must be one of {valid_methods}")

        if self.adaptation_method not in valid_adapt:
            errors.append(
                f"Invalid adaptation_method: '{self.adaptation_method}'. Must be one of {valid_adapt}")

        for ch in self.selective_channels:
            if ch not in ['L', 'a', 'b']:
                errors.append(f"Invalid channel in selective_channels: '{ch}'. Must be 'L', 'a', or 'b'.")

        if not (0.0 <= self.blend_factor <= 1.0):
            errors.append(f"Invalid blend_factor: {self.blend_factor}. Must be between 0.0 and 1.0.")

        if self.channel_weights:
            for ch, w in self.channel_weights.items():
                if ch not in ['L', 'a', 'b']:
                    errors.append(f"Invalid channel in channel_weights: '{ch}'.")
                if not (0.0 <= w <= 1.0):
                    errors.append(f"Invalid weight for channel '{ch}': {w}. Must be between 0.0 and 1.0.")

        if not (isinstance(self.num_segments, int) and self.num_segments > 0):
            errors.append(f"Invalid num_segments: {self.num_segments}. Must be a positive integer.")

        if not (isinstance(self.delta_e_threshold, (int, float)) and self.delta_e_threshold >= 0):
            errors.append(f"Invalid delta_e_threshold: {self.delta_e_threshold}. Must be a non-negative number.")

        if not (0.0 <= self.min_segment_size_perc <= 1.0):
            errors.append(f"Invalid min_segment_size_perc: {self.min_segment_size_perc}. Must be between 0.0 and 1.0.")

        if not isinstance(self.hybrid_pipeline, list):
            errors.append(f"'hybrid_pipeline' must be a list, got {type(self.hybrid_pipeline)}.")
        else:
            for idx, step in enumerate(self.hybrid_pipeline):
                if not isinstance(step, dict):
                    errors.append(f"Hybrid pipeline step {idx} must be a dictionary.")
                    continue
                if "method" not in step:
                    errors.append(f"Hybrid pipeline step {idx} is missing 'method' key.")
                # Further validation of methods and params within pipeline can be added here
                # For now, we assume core/gpu_core will validate specific methods and their params

        if errors:
            raise ValueError("Configuration errors: " + "; ".join(errors))
