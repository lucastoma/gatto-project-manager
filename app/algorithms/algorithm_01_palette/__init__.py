# Ten plik definiuje, co jest publicznie dostępne z pakietu 'algorithm_01_palette'.

# Import z oryginalnej wersji CPU
from .algorithm import PaletteMappingAlgorithm
from .algorithm import create_palette_mapping_algorithm

# Tworzymy aliasy dla spójności, ale zostawiamy też oryginalne nazwy
PaletteMappingAlgorithmCPU = PaletteMappingAlgorithm
create_palette_mapping_algorithm_cpu = create_palette_mapping_algorithm

# Import z nowej wersji GPU (OpenCL)
# Upewniamy się, że import nie wywali się, jeśli OpenCL nie jest dostępne.
try:
    from .algorithm_gpu import PaletteMappingAlgorithmGPU
    from .algorithm_gpu import create_palette_mapping_algorithm_gpu
    from .algorithm_gpu import algorithm_gpu_taichi_init  # Dodano import
    OPENCL_AVAILABLE = True
except (ImportError, RuntimeError):
    PaletteMappingAlgorithmGPU = None
    create_palette_mapping_algorithm_gpu = None
    algorithm_gpu_taichi_init = None  # Dodano fallback
    OPENCL_AVAILABLE = False


# Definiujemy, co jest eksportowane. To naprawi błąd importu w nadrzędnym __init__.py.
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
        'algorithm_gpu_taichi_init'  # Dodano do eksportu
    ])
