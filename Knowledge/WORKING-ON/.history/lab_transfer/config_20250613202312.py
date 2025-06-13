"""
Configuration module for LAB Color Transfer algorithm.
"""

class LABTransferConfig:
    """
    Configuration for LAB Color Transfer.
    """
    def __init__(
        self,
        method: str = 'basic',
        channel_weights: dict = None,
        adaptation_method: str = 'none',
        tile_size: int = 512,
        overlap: int = 64
    ):
        self.method = method
        self.channel_weights = channel_weights or {'L': 1.0, 'a': 1.0, 'b': 1.0}
        self.adaptation_method = adaptation_method
        self.tile_size = tile_size
        self.overlap = overlap

    def validate(self):
        """
        Validates the configuration values and raises ValueError if invalid.
        """
        valid_methods = ['basic', 'weighted', 'selective', 'adaptive']
        valid_adapt = ['none', 'luminance', 'saturation', 'gradient']
        errors = []
        if self.method not in valid_methods:
            errors.append(f"Invalid method: {self.method}")
        if self.adaptation_method not in valid_adapt:
            errors.append(f"Invalid adaptation_method: {self.adaptation_method}")
        if errors:
            raise ValueError('Invalid configuration: ' + '; '.join(errors))
