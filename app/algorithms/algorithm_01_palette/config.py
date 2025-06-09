"""
Algorithm 01: Palette Mapping Configuration
===========================================

Configuration parameters, validation, and defaults for the palette mapping algorithm.
"""

from typing import Dict, Any, Union
from dataclasses import dataclass, field
import json


@dataclass
class PaletteMappingConfig:
    """Configuration class for Palette Mapping Algorithm."""
    
    # Core algorithm parameters
    k_colors: int = 8              # Number of colors in palette (4-32)
    random_state: int = 42         # Random seed for reproducible results
    n_init: int = 10              # Number of K-means initializations
    max_iter: int = 300           # Maximum K-means iterations
    tol: float = 1e-4             # K-means convergence tolerance
    
    # Performance parameters
    enable_monitoring: bool = True  # Enable performance monitoring
    memory_limit_mb: int = 512     # Memory limit for processing
    timeout_seconds: int = 300     # Processing timeout
    
    # Quality parameters
    min_image_size: int = 32       # Minimum image dimension
    max_image_size: int = 4096     # Maximum image dimension
    downsample_threshold: int = 2048  # Downsample images larger than this
    
    # Output parameters
    output_format: str = "tiff"    # Output file format
    compression_quality: int = 95   # JPEG quality or similar
    preserve_alpha: bool = True    # Preserve alpha channel if present
    
    # Advanced parameters
    color_space: str = "RGB"       # Color space for processing
    distance_metric: str = "euclidean"  # Color distance metric
    enable_edge_preservation: bool = False  # Experimental edge preservation
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []
        
        # Validate k_colors
        if not (4 <= self.k_colors <= 32):
            errors.append(f"k_colors must be between 4 and 32, got {self.k_colors}")
        
        # Validate random_state
        if not isinstance(self.random_state, int) or self.random_state < 0:
            errors.append(f"random_state must be non-negative integer, got {self.random_state}")
        
        # Validate n_init
        if not (1 <= self.n_init <= 100):
            errors.append(f"n_init must be between 1 and 100, got {self.n_init}")
        
        # Validate max_iter
        if not (10 <= self.max_iter <= 1000):
            errors.append(f"max_iter must be between 10 and 1000, got {self.max_iter}")
        
        # Validate tolerance
        if not (1e-6 <= self.tol <= 1e-1):
            errors.append(f"tol must be between 1e-6 and 1e-1, got {self.tol}")
        
        # Validate memory limit
        if not (64 <= self.memory_limit_mb <= 8192):
            errors.append(f"memory_limit_mb must be between 64 and 8192, got {self.memory_limit_mb}")
        
        # Validate timeout
        if not (10 <= self.timeout_seconds <= 3600):
            errors.append(f"timeout_seconds must be between 10 and 3600, got {self.timeout_seconds}")
        
        # Validate image sizes
        if not (16 <= self.min_image_size <= 256):
            errors.append(f"min_image_size must be between 16 and 256, got {self.min_image_size}")
        
        if not (512 <= self.max_image_size <= 16384):
            errors.append(f"max_image_size must be between 512 and 16384, got {self.max_image_size}")
        
        if self.min_image_size >= self.max_image_size:
            errors.append("min_image_size must be less than max_image_size")
        
        # Validate downsample threshold
        if not (256 <= self.downsample_threshold <= self.max_image_size):
            errors.append(f"downsample_threshold must be between 256 and max_image_size")
        
        # Validate output format
        valid_formats = ["tiff", "png", "jpg", "jpeg", "bmp"]
        if self.output_format.lower() not in valid_formats:
            errors.append(f"output_format must be one of {valid_formats}, got {self.output_format}")
        
        # Validate compression quality
        if not (1 <= self.compression_quality <= 100):
            errors.append(f"compression_quality must be between 1 and 100, got {self.compression_quality}")
        
        # Validate color space
        valid_color_spaces = ["RGB", "BGR", "LAB", "HSV"]
        if self.color_space not in valid_color_spaces:
            errors.append(f"color_space must be one of {valid_color_spaces}, got {self.color_space}")
        
        # Validate distance metric
        valid_metrics = ["euclidean", "manhattan", "cosine"]
        if self.distance_metric not in valid_metrics:
            errors.append(f"distance_metric must be one of {valid_metrics}, got {self.distance_metric}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'k_colors': self.k_colors,
            'random_state': self.random_state,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'enable_monitoring': self.enable_monitoring,
            'memory_limit_mb': self.memory_limit_mb,
            'timeout_seconds': self.timeout_seconds,
            'min_image_size': self.min_image_size,
            'max_image_size': self.max_image_size,
            'downsample_threshold': self.downsample_threshold,
            'output_format': self.output_format,
            'compression_quality': self.compression_quality,
            'preserve_alpha': self.preserve_alpha,
            'color_space': self.color_space,
            'distance_metric': self.distance_metric,
            'enable_edge_preservation': self.enable_edge_preservation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaletteMappingConfig':
        """Create configuration from dictionary."""
        return cls(**data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PaletteMappingConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Predefined configurations for different use cases
PRESET_CONFIGS = {
    'fast': PaletteMappingConfig(
        k_colors=6,
        n_init=3,
        max_iter=100,
        downsample_threshold=1024,
        enable_monitoring=False
    ),
    
    'balanced': PaletteMappingConfig(
        k_colors=8,
        n_init=10,
        max_iter=300,
        downsample_threshold=2048,
        enable_monitoring=True
    ),
    
    'quality': PaletteMappingConfig(
        k_colors=16,
        n_init=20,
        max_iter=500,
        downsample_threshold=4096,
        enable_monitoring=True,
        enable_edge_preservation=True
    ),
    
    'artistic': PaletteMappingConfig(
        k_colors=4,
        n_init=15,
        max_iter=300,
        downsample_threshold=2048,
        enable_monitoring=True
    ),
    
    'photorealistic': PaletteMappingConfig(
        k_colors=24,
        n_init=25,
        max_iter=400,
        downsample_threshold=4096,
        enable_monitoring=True,
        enable_edge_preservation=True
    )
}


def get_config(preset: str = 'balanced') -> PaletteMappingConfig:
    """Get a predefined configuration preset."""
    if preset not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    return PRESET_CONFIGS[preset]


def create_config_from_api_params(k_colors: int = 8, **kwargs) -> PaletteMappingConfig:
    """Create configuration from API parameters (for backward compatibility)."""
    config = get_config('balanced')
    config.k_colors = k_colors
    
    # Apply any additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    config.validate()
    return config
