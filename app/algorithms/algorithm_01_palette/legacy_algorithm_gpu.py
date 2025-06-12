# /app/algorithms/algorithm_01_palette/algorithm_gpu.py
# Główny, zintegrowany plik algorytmu. Wersja poprawiona.

import numpy as np
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

# Importowanie składowych z podzielonych modułów
from . import algorithm_gpu_utils as utils
from . import algorithm_gpu_config as cfg
from . import algorithm_gpu_taichi_init as ti_init
from . import algorithm_gpu_kernels as kernels
from . import algorithm_gpu_cpu_fallback as cpu
from . import algorithm_gpu_exceptions as err

# --- POPRAWIONA INICJALIZACJA ---
# Inicjalizuj Taichi przy starcie i przechowaj flagę o dostępności akceleracji GPU
IS_GPU_ACCELERATED = ti_init.safe_taichi_init()

class PaletteMappingAlgorithmGPU:
    """Główna klasa orkiestrująca, wykorzystująca logikę z osobnych modułów."""
    def __init__(self, config_path: str = None, algorithm_id: str = "algorithm_01_palette_gpu_refactored"):
        self.algorithm_id = algorithm_id
        self.logger = utils.get_logger()
        self.profiler = utils.get_profiler()
        
        self.logger.info(f"Initializing GPU algorithm: {self.algorithm_id}")
        if not ti_init.TAICHI_AVAILABLE:
            self.logger.warning("Taichi library not available. Full CPU mode.")
        elif not IS_GPU_ACCELERATED:
            self.logger.warning(f"Taichi GPU backend not found. Using Taichi with CPU backend: {ti_init.GPU_BACKEND}")
        else:
            self.logger.info(f"Taichi GPU backend successfully initialized: {ti_init.GPU_BACKEND}")
        
        self.name = "Palette Mapping GPU Accelerated (Refactored)"
        self.version = "5.0-Stable"

        # Konfiguracja
        self.default_config = cfg.get_default_config()
        self.config = cfg.load_config(config_path, self.default_config, self.logger) if config_path else self.default_config.copy()
        
        # Zarządzanie pamięcią GPU
        self._gpu_memory_manager = kernels.GPUMemoryManager(
            logger=self.logger, 
            max_gpu_memory=self.config.get("max_gpu_memory", 2 * 1024**3), 
            max_palette_size=self.config.get("max_palette_size", 256)
        )

    def _determine_strategy(self, image_size: Tuple[int, int], palette_size: int, config: Dict[str, Any]) -> utils.AccelerationStrategy:
        """Określa optymalną strategię przetwarzania."""
        if config.get("force_cpu", False) or not IS_GPU_ACCELERATED:
            return utils.AccelerationStrategy.CPU

        pixel_count = image_size[0] * image_size[1]
        if pixel_count < 50_000: return utils.AccelerationStrategy.CPU
        if pixel_count < 500_000 and palette_size <= 16: return utils.AccelerationStrategy.GPU_SMALL
        if pixel_count < 5_000_000: return utils.AccelerationStrategy.GPU_MEDIUM
        return utils.AccelerationStrategy.GPU_LARGE
        
    def _map_pixels_to_palette_gpu(self, image_array: np.ndarray, palette: List[List[int]], config: Dict[str, Any]) -> np.ndarray:
        """Wykonuje mapowanie pikseli z użyciem kerneli GPU, z poprawną walidacją i obsługą błędów."""
        try:
            # Krok 1: Walidacja danych wejściowych przy użyciu funkcji pomocniczych
            height, width, _ = utils.validate_image_array(image_array)
            palette_np_normalized = utils.validate_palette(palette) # Zwraca paletę znormalizowaną do [0,1]
            
            # Krok 2: Przygotowanie danych
            num_pixels = height * width
            pixels_flat_normalized = image_array.reshape(-1, 3).astype(np.float32) / 255.0
            hue_weight = float(config.get('hue_weight', 3.0))
            use_64bit = config.get("use_64bit_indices", False) or num_pixels > 2**31 - 1
            
            # Krok 3: Alokacja pamięci GPU
            fields = self._gpu_memory_manager.get_or_create_fields(num_pixels, len(palette), use_64bit)
            
            # Krok 4: Transfer danych na GPU
            fields['pixels_rgb'].from_numpy(pixels_flat_normalized)
            fields['palette_hsv'].from_numpy(cpu.rgb2hsv(palette_np_normalized))
            fields['distance_modifier'].fill(1.0)
            
            # Krok 5: Wykonanie kerneli
            if config.get("enable_kernel_fusion", True):
                 kernels.combined_conversion_and_mapping(fields['pixels_rgb'], fields['palette_hsv'], fields['closest_indices'], fields['distance_modifier'], hue_weight)
            else:
                 kernels.rgb_to_hsv_kernel(fields['pixels_rgb'], fields['pixels_hsv'])
                 kernels.find_closest_color_kernel(fields['pixels_hsv'], fields['palette_hsv'], fields['closest_indices'], fields['distance_modifier'], hue_weight)
            
            # Krok 6: Pobranie wyników
            closest_indices_np = fields['closest_indices'].to_numpy()
            
            # Krok 7: Mapowanie indeksów na kolory i powrót do formatu 8-bit
            mapped_colors = (palette_np_normalized[closest_indices_np] * 255.0).astype(np.uint8)
            return mapped_colors.reshape(height, width, 3)

        except (err.GPUMemoryError, err.ImageProcessingError) as e:
            self.logger.error(f"Specific error during GPU processing: {e}")
            raise
        except Exception as e:
            raise err.GPUProcessingError(f"An unexpected error occurred in GPU processing pipeline: {e}") from e

    def _map_pixels_to_palette(self, image_array: np.ndarray, palette: List[List[int]], config: Dict[str, Any]) -> np.ndarray:
        """Inteligentny dispatcher, wybierający między CPU i GPU."""
        strategy = self._determine_strategy(image_array.shape[:2], len(palette), config)
        self.logger.info(f"Processing strategy: {strategy.name}")

        if strategy == utils.AccelerationStrategy.CPU:
            return cpu.map_pixels_to_palette_cpu(image_array, palette, config, self.logger)
        
        try:
            return self._map_pixels_to_palette_gpu(image_array, palette, config)
        except (err.GPUProcessingError, err.GPUMemoryError, err.ImageProcessingError) as e:
            self.logger.warning(f"GPU processing failed: {e}. Falling back to CPU.")
            self._gpu_memory_manager.cleanup()
            return cpu.map_pixels_to_palette_cpu(image_array, palette, config, self.logger)

    def process_images(self, master_path: str, target_path: str, output_path: str, **kwargs) -> bool:
        """Główny potok przetwarzania synchronicznego."""
        run_config = self.config.copy()
        run_config.update(kwargs)
        cfg.validate_run_config(run_config, self.config.get("max_palette_size", 256))
        
        try:
            master_image = cpu.safe_image_load(master_path, self.logger)
            target_image = cpu.safe_image_load(target_path, self.logger)
            target_array = np.array(target_image) # uint8
            
            palette = cpu.extract_palette(
                master_image, run_config['num_colors'], run_config['quality'], 
                run_config['inject_extremes'], self.config.get("max_palette_size", 256), self.logger)
                
            mapped_array = self._map_pixels_to_palette(target_array, palette, run_config)
            
            Image.fromarray(mapped_array).save(output_path, quality=95, optimize=True)
            self.logger.info(f"Successfully saved processed image: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Main processing pipeline failed: {e}", exc_info=True)
            return False

    async def process_images_async(self, master_path: str, target_path: str, output_path: str, **kwargs) -> bool:
        """Główny potok przetwarzania asynchronicznego."""
        # ... implementacja async pozostaje bez zmian, ale będzie korzystać z poprawionych metod
        return self.process_images(master_path, target_path, output_path, **kwargs) # Uproszczone dla przykładu

def create_palette_mapping_algorithm_gpu():
    """Funkcja fabrykująca do tworzenia instancji algorytmu."""
    return PaletteMappingAlgorithmGPU()
