from .algorithm import PaletteMappingAlgorithm
from .algorithm import create_palette_mapping_algorithm
PaletteMappingAlgorithmCPU = PaletteMappingAlgorithm
create_palette_mapping_algorithm_cpu = create_palette_mapping_algorithm

try:
    from .algorithm_gpu import PaletteMappingAlgorithmGPU
    from .algorithm_gpu import create_palette_mapping_algorithm_gpu
    OPENCL_AVAILABLE = True
except (ImportError, RuntimeError):
    PaletteMappingAlgorithmGPU = None
    create_palette_mapping_algorithm_gpu = None
    OPENCL_AVAILABLE = False
    
__all__ = [
    'PaletteMappingAlgorithm',
    'create_palette_mapping_algorithm',
    'PaletteMappingAlgorithmCPU',
    'create_palette_mapping_algorithm_cpu',
]
if OPENCL_AVAILABLE:
    __all__.extend([
        'PaletteMappingAlgorithmGPU',
        'create_palette_mapping_algorithm_gpu',
    ])