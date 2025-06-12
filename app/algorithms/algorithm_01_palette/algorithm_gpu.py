# /app/algorithms/algorithm_01_palette/algorithm-gpu.py
# Główny plik algorytmu, integrujący wszystkie moduły.

import numpy as np
from PIL import Image
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

# Importowanie składowych z podzielonych modułów
from . import algorithm_gpu_utils as utils
from . import algorithm_gpu_config as cfg
from . import algorithm_gpu_kernels as kernels
from . import algorithm_gpu_cpu_fallback as cpu

# Inicjalizacja Taichi przy starcie modułu
IS_GPU_ACCELERATED = kernels.safe_taichi_init()

class PaletteMappingAlgorithmGPU:
    """
    Główna klasa orkiestrująca, wykorzystująca logikę z osobnych modułów.
    """
    def __init__(self, config_path: str = None, algorithm_id: str = "algorithm_01_palette_gpu_refactored"):
        self.algorithm_id = algorithm_id
        self.logger = utils.get_logger()
        self.profiler = utils.get_profiler()
        
        self.logger.info(f"Initializing GPU algorithm: {self.algorithm_id}")
        if not kernels.TAICHI_AVAILABLE:
            self.logger.warning("Taichi library not available. Full CPU mode.")
        elif not IS_GPU_ACCELERATED:
            self.logger.warning("Taichi GPU backend not available. Using CPU for Taichi operations.")
        
        self.name = "Palette Mapping GPU Accelerated (Refactored)"
        self.version = "4.0-Modular"

        # Konfiguracja
        self.default_config = cfg.get_default_config()
        self.config = cfg.load_config(config_path, self.default_config) if config_path else self.default_config.copy()
        self._max_palette_size = 256
        
        # Zarządzanie pamięcią GPU
        self._cached_fields: Dict[str, Any] = {}
        self._max_gpu_memory = 2 * 1024**3  # 2GB
        self._gpu_memory_cleanup_threshold = 5

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
        """Wykonuje mapowanie pikseli z użyciem kerneli GPU."""
        start_time = time.perf_counter()
        palette_np = np.array(palette, dtype=np.float32)
        pixels_flat = np.clip(image_array, 0, 255).reshape(-1, 3).astype(np.float32)
        
        num_pixels = pixels_flat.shape[0]
        use_64bit = config.get("use_64bit_indices", False) or num_pixels > 2**31 - 1

        self._cached_fields = kernels.setup_taichi_fields(
            self._cached_fields, num_pixels, len(palette), use_64bit, 
            self._max_gpu_memory, self._max_palette_size, self.logger
        )

        # Transfer danych na GPU
        self._cached_fields['pixels_rgb'].from_numpy(pixels_flat)
        self._cached_fields['palette_hsv'].from_numpy(
            cpu.rgb2hsv(palette_np / 255.0)
        )
        self._cached_fields['distance_modifier'].fill(1.0) # Prosta inicjalizacja
        
        # Wykonanie kerneli
        if config.get("enable_kernel_fusion", True) and num_pixels > 100_000:
            kernels.combined_conversion_and_mapping(
                self._cached_fields['pixels_rgb'], self._cached_fields['palette_hsv'], 
                self._cached_fields['closest_indices'], self._cached_fields['distance_modifier'], 
                config.get("hue_weight", 3.0))
        else:
            kernels.rgb_to_hsv_kernel(self._cached_fields['pixels_rgb'], self._cached_fields['pixels_hsv'])
            kernels.find_closest_color_kernel(
                self._cached_fields['pixels_hsv'], self._cached_fields['palette_hsv'], 
                self._cached_fields['closest_indices'], self._cached_fields['distance_modifier'], 
                config.get("hue_weight", 3.0))
        
        kernels.ti.sync()
        
        # Odczyt wyników
        closest_indices_np = self._cached_fields['closest_indices'].to_numpy()
        mapped_array = palette_np[closest_indices_np].reshape(image_array.shape)
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        self.logger.info(f"GPU processed {num_pixels:,} pixels in {total_time_ms:.1f}ms")
        
        return np.clip(mapped_array, 0, 255)

    def _map_pixels_to_palette(self, image_array: np.ndarray, palette: List[List[int]], config: Dict[str, Any]) -> np.ndarray:
        """Inteligentny dispatcher, wybierający między CPU i GPU."""
        strategy = self._determine_strategy(image_array.shape[:2], len(palette), config)
        self.logger.info(f"Processing strategy: {strategy.name}")

        if strategy == utils.AccelerationStrategy.CPU:
            return cpu.map_pixels_to_palette_cpu(image_array, palette, config, self.logger)
        
        try:
            return self._map_pixels_to_palette_gpu(image_array, palette, config)
        except (utils.GPUProcessingError, utils.GPUMemoryError) as e:
            self.logger.warning(f"GPU error: {e}. Falling back to CPU.")
            kernels.cleanup_gpu_memory(self._cached_fields, self.logger)
            return cpu.map_pixels_to_palette_cpu(image_array, palette, config, self.logger)

    def process_images(self, master_path: str, target_path: str, output_path: str, **kwargs) -> bool:
        """Główny potok przetwarzania synchronicznego."""
        run_config = self.config.copy()
        run_config.update(kwargs)
        cfg.validate_run_config(run_config, self._max_palette_size)
        
        try:
            master_image = cpu.safe_image_load(master_path, self.logger)
            target_image = cpu.safe_image_load(target_path, self.logger)
            target_array = np.array(target_image)
            
            palette = cpu.extract_palette(
                master_image, run_config['num_colors'], run_config['quality'], 
                run_config['inject_extremes'], self._max_palette_size, self.logger)
                
            mapped_array = self._map_pixels_to_palette(target_array, palette, run_config)
            
            Image.fromarray(mapped_array.astype(np.uint8)).save(output_path, quality=95, optimize=True)
            self.logger.info(f"Successfully saved processed image: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Processing failed: {e}", exc_info=True)
            return False

    async def process_images_async(self, master_path: str, target_path: str, output_path: str, **kwargs) -> bool:
        """Główny potok przetwarzania asynchronicznego."""
        run_config = self.config.copy()
        run_config.update(kwargs)
        cfg.validate_run_config(run_config, self._max_palette_size)
        
        loop = asyncio.get_event_loop()
        
        try:
            with ThreadPoolExecutor() as executor:
                master_image = await loop.run_in_executor(executor, cpu.safe_image_load, master_path, self.logger)
                target_image = await loop.run_in_executor(executor, cpu.safe_image_load, target_path, self.logger)
                target_array = np.array(target_image)

                palette = await loop.run_in_executor(executor, cpu.extract_palette, 
                    master_image, run_config['num_colors'], run_config['quality'], 
                    run_config['inject_extremes'], self._max_palette_size, self.logger)

            mapped_array = self._map_pixels_to_palette(target_array, palette, run_config)
            
            def save_image():
                Image.fromarray(mapped_array.astype(np.uint8)).save(output_path, quality=95, optimize=True)

            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, save_image)

            self.logger.info(f"Async processing completed for {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Async processing failed: {e}", exc_info=True)
            return False

def create_palette_mapping_algorithm_gpu():
    """Funkcja fabrykująca do tworzenia instancji algorytmu."""
    return PaletteMappingAlgorithmGPU()
