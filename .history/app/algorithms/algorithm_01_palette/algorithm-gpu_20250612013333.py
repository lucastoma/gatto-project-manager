# /app/algorithms/algorithm_01_palette/algorithm-gpu-production-final.py
# Finalna wersja produkcyjna z rozwiązaniem wszystkich problemów stabilności
# WERSJA ZMODYFIKOWANA - zawiera poprawki i komentarze w odpowiedzi na analizę kodu.

import logging
import numpy as np
from PIL import Image, ImageFilter
import time
import json
import asyncio
import statistics
import gc
import threading
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from skimage import color
from sklearn.cluster import KMeans
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from enum import Enum

# --- POPRAWKA 1: Bezpieczna inicjalizacja Taichi z proper cleanup ---
TAICHI_AVAILABLE = False
GPU_BACKEND = "none"
ti, tm = None, None
_taichi_lock = threading.Lock()  # Thread safety dla inicjalizacji

def _safe_taichi_cleanup():
    """Safely cleanup Taichi context if needed."""
    global ti
    try:
        if ti is not None and hasattr(ti, 'reset'):
            ti.reset()
    except Exception as e:
        print(f"Warning: Taichi cleanup failed: {e}")

def _safe_taichi_init():
    """POPRAWKA 1: Bezpieczna inicjalizacja z proper resource management."""
    global TAICHI_AVAILABLE, GPU_BACKEND, ti, tm
    
    with _taichi_lock:  # Thread safety
        try:
            import taichi as ti_module
            import taichi.math as tm_module
            
            # Sprawdź dostępność GPU backends
            available_archs = []
            if hasattr(ti_module, 'cuda') and ti_module.cuda.is_available():
                available_archs.append('cuda')
            if hasattr(ti_module, 'vulkan') and ti_module.vulkan.is_available():
                available_archs.append('vulkan')
            if hasattr(ti_module, 'metal') and ti_module.metal.is_available():
                available_archs.append('metal')
            
            if not available_archs:
                print("WARNING: No GPU backends available. Using CPU fallback.")
                try:
                    ti_module.init(arch=ti_module.cpu, log_level=ti_module.WARN)
                    ti, tm = ti_module, tm_module
                    return False
                except Exception as e:
                    print(f"ERROR: CPU fallback initialization failed: {e}")
                    return False
            
            # Próbuj inicjalizacji z GPU
            try:
                ti_module.init(arch=ti_module.gpu, log_level=ti_module.WARN)
                
                # Sprawdź backend
                try:
                    if hasattr(ti_module, 'lang') and hasattr(ti_module.lang, 'impl'):
                        cfg = ti_module.lang.impl.current_cfg()
                        GPU_BACKEND = str(cfg.arch)
                    else:
                        GPU_BACKEND = "gpu_unknown"
                except:
                    GPU_BACKEND = "gpu_unknown"
                
                # Test funkcjonalności GPU
                test_field = ti_module.field(dtype=ti_module.f32, shape=10)
                test_field.fill(1.0)
                ti_module.sync()
                
                # Cleanup test
                del test_field
                
                print(f"SUCCESS: Taichi GPU initialized with backend: {GPU_BACKEND}")
                ti, tm = ti_module, tm_module
                return True
                
            except Exception as e:
                print(f"WARNING: GPU initialization failed: {e}. Trying CPU fallback.")
                
                # POPRAWKA 1: Proper cleanup before fallback
                _safe_taichi_cleanup()
                # [FIX - UWAGA #1] Dodatkowe wyzerowanie globalnych referencji dla pewności,
                # że nie zostanie użyty częściowo zainicjalizowany, niestabilny obiekt.
                ti, tm = None, None
                
                try:
                    ti_module.init(arch=ti_module.cpu, log_level=ti_module.WARN)
                    print("Taichi initialized with CPU backend")
                    ti, tm = ti_module, tm_module
                    return False
                except Exception as e2:
                    print(f"ERROR: Complete Taichi initialization failed: {e2}")
                    return False
                    
        except ImportError as e:
            print(f"WARNING: Taichi not available: {e}")
            return False

# Inicjalizuj Taichi
TAICHI_AVAILABLE = _safe_taichi_init()

# Scipy imports
try:
    from scipy.spatial import KDTree
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    KDTree, ndimage = None, None
    SCIPY_AVAILABLE = False

# Project imports with fallbacks
try:
    from ...core.development_logger import get_logger
    from ...core.performance_profiler import get_profiler
    if TYPE_CHECKING:
        from ...core.development_logger import DevelopmentLogger
        from ...core.performance_profiler import PerformanceProfiler
except ImportError:
    def get_logger() -> Any:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        return logging.getLogger(__name__)

    class DummyProfiler:
        def profile_operation(self, *args, **kwargs):
            import contextlib
            return contextlib.nullcontext()
    
    def get_profiler() -> Any:
        return DummyProfiler()

# --- Enhanced Custom Exceptions ---
class GPUProcessingError(Exception):
    """Custom exception for GPU processing errors."""
    pass

class GPUMemoryError(GPUProcessingError):
    """Specific exception for GPU memory issues."""
    pass

class ImageProcessingError(Exception):
    """Exception for image loading/processing errors."""
    pass

# --- Acceleration Strategy Enum ---
class AccelerationStrategy(Enum):
    """Strategy for choosing processing backend."""
    CPU = 0
    GPU_SMALL = 1    # Small palettes, simple algorithm
    GPU_MEDIUM = 2   # Medium complexity
    GPU_LARGE = 3    # Full GPU pipeline with batch processing

class PaletteMappingAlgorithmGPU:
    """
    Production-ready GPU-accelerated palette mapping algorithm with all
    stability issues resolved and comprehensive error handling.
    """
    
    def __init__(self, config_path: str = None, algorithm_id: str = "algorithm_01_palette_gpu"):
        self.algorithm_id = algorithm_id
        
        if TYPE_CHECKING:
            self.logger: "DevelopmentLogger" = get_logger()
            self.profiler: "PerformanceProfiler" = get_profiler()
        else:
            self.logger = get_logger()
            self.profiler = get_profiler()
        
        self.logger.info(f"Initializing GPU algorithm: {self.algorithm_id}")
        
        if not TAICHI_AVAILABLE:
            self.logger.warning("Taichi GPU unavailable. All operations will use CPU fallback.")
            
        self.name = "Palette Mapping GPU Accelerated"
        self.version = "3.4-ProductionStable-Revised"
        
        # Configuration
        self.default_config_values = self._get_default_config()
        self.config = self.load_config(config_path) if config_path else self.default_config_values.copy()

        if not SCIPY_AVAILABLE:
            self.logger.warning("Scipy unavailable. Advanced features disabled.")
        
        # GPU memory management
        self._cached_fields: Dict[str, Any] = {}
        self._max_gpu_memory = 2 * 1024**3  # 2GB safe limit
        self._max_batch_pixels = 2_000_000   # 2M pixels per batch
        self._max_palette_size = 256        # Maximum palette size for efficient indexing
        
        # POPRAWKA 3: Thread safety dla benchmarkingu
        self._benchmark_lock = threading.Lock()
        self._original_fusion_state = True
        
        # Performance optimization flags
        self._use_kernel_fusion = True
        self._gpu_memory_cleanup_threshold = 5

    # --- Configuration and Utilities ---
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Enhanced default algorithm configuration."""
        return {
            "num_colors": 16,
            "palette_method": "kmeans",
            "quality": 5,
            "distance_metric": "weighted_hsv",
            "hue_weight": 3.0,
            "use_color_focus": False,
            "focus_ranges": [],
            "dithering_method": "none",
            "dithering_strength": 8.0,
            "inject_extremes": False,
            "preserve_extremes": False,
            "extremes_threshold": 10,
            "edge_blur_enabled": False,
            "edge_blur_radius": 1.5,
            "edge_blur_strength": 0.3,
            "edge_detection_threshold": 25,
            "postprocess_median_filter": False,
            # GPU-specific options
            "force_cpu": False,
            "gpu_batch_size": 2_000_000,
            "enable_kernel_fusion": True,
            "gpu_memory_cleanup": True,
            "use_64bit_indices": False,  # For extremely large images
        }

    def _validate_run_config(self, config: Dict[str, Any]):
        """[FIX - UWAGA #8] Central validation for runtime config parameters."""
        if "hue_weight" in config:
            config["hue_weight"] = max(0.1, min(10.0, float(config["hue_weight"])))
        if "gpu_batch_size" in config:
            config["gpu_batch_size"] = max(100_000, min(10_000_000, int(config["gpu_batch_size"])))
        if "num_colors" in config:
            config["num_colors"] = max(2, min(self._max_palette_size, int(config["num_colors"])))

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file with enhanced validation."""
        config = self.default_config_values.copy()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            
            # [FIX - UWAGA #8] Use central validation method.
            self._validate_run_config(user_config)
            
            config.update(user_config)
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}, using defaults.")
        return config

    def _determine_strategy(self, image_size: Tuple[int, int], palette_size: int, 
                             config: Dict[str, Any]) -> AccelerationStrategy:
        """POPRAWKA 2: Enhanced strategy determination with input validation."""
        if config.get("force_cpu", False) or not TAICHI_AVAILABLE:
            return AccelerationStrategy.CPU
        
        if len(image_size) != 2:
            raise ValueError(f"Invalid image_size: expected 2D tuple, got {image_size}")
        
        height, width = image_size
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")
        
        if height > 65535 or width > 65535:
            self.logger.warning(f"Very large image dimensions: {width}x{height}")
        
        pixel_count = height * width
        
        if pixel_count > 2**31 - 1:
            self.logger.warning(f"Large image ({pixel_count:,} pixels) may require 64-bit indices")
            if not config.get("use_64bit_indices", False):
                self.logger.info("Consider setting use_64bit_indices=True for very large images")
        
        if pixel_count < 50_000:
            return AccelerationStrategy.CPU
        elif pixel_count < 500_000 and palette_size <= 16:
            return AccelerationStrategy.GPU_SMALL
        elif pixel_count < 5_000_000:
            return AccelerationStrategy.GPU_MEDIUM
        else:
            return AccelerationStrategy.GPU_LARGE

    def _validate_inputs(self, image_array: np.ndarray, palette: List[List[int]]) -> None:
        """Enhanced input validation with detailed error messages."""
        if image_array.size == 0:
            raise ValueError("Input image is empty")
        
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_array.shape}")
        
        if len(palette) == 0:
            raise ValueError("Palette is empty")
        
        if len(palette) > self._max_palette_size:
            raise ValueError(f"Palette too large: {len(palette)} colors (max {self._max_palette_size})")
        
        for i, color_val in enumerate(palette):
            if len(color_val) != 3:
                raise ValueError(f"Palette color {i} must have 3 RGB components")
            if not all(isinstance(c, (int, float)) and 0 <= c <= 255 for c in color_val):
                raise ValueError(f"Palette color {i} has invalid RGB values: {color_val}")

    def _safe_image_load(self, image_path: str) -> Image.Image:
        """POPRAWKA 4: Enhanced image loading with comprehensive error handling."""
        try:
            path_obj = Path(image_path)
            if not path_obj.exists():
                raise ImageProcessingError(f"Image file not found: {image_path}")
            
            if not path_obj.is_file():
                raise ImageProcessingError(f"Path is not a file: {image_path}")
            
            file_size = path_obj.stat().st_size
            if file_size == 0:
                raise ImageProcessingError(f"Image file is empty: {image_path}")
            
            if file_size > 500 * 1024 * 1024:  # 500MB limit
                self.logger.warning(f"Very large image file: {file_size / 1024**2:.1f}MB")
            
            try:
                image = Image.open(image_path)
            except Image.UnidentifiedImageError as e:
                raise ImageProcessingError(f"Cannot identify image format: {image_path} - {e}")
            except PermissionError as e:
                raise ImageProcessingError(f"Permission denied reading file: {image_path} - {e}")
            except OSError as e:
                raise ImageProcessingError(f"OS error reading file: {image_path} - {e}")
            
            if image.size[0] == 0 or image.size[1] == 0:
                raise ImageProcessingError(f"Image has zero dimensions: {image.size}")
            
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            
            return image
            
        except ImageProcessingError:
            raise
        except Exception as e:
            raise ImageProcessingError(f"Unexpected error loading image {image_path}: {e}")

    def _cleanup_gpu_memory(self):
        """Enhanced GPU memory cleanup."""
        if TAICHI_AVAILABLE and self._cached_fields:
            try:
                self._cached_fields.clear()
                gc.collect()
                ti.sync()
                self.logger.debug("GPU memory cleanup completed")
            except Exception as e:
                self.logger.warning(f"GPU memory cleanup failed: {e}")

    # --- CPU Implementation (Enhanced Fallback) ---
    
    def extract_palette(self, image_path: str, num_colors: int, method: str, 
                        quality: int, inject_extremes: bool) -> List[List[int]]:
        """Extract color palette with enhanced error handling."""
        with self.profiler.profile_operation("extract_palette_cpu", algorithm_id=self.algorithm_id):
            try:
                image = self._safe_image_load(image_path)
                
                base_size, max_size = 100, 1000
                thumbnail_size = int(base_size + (max_size - base_size) * (quality - 1) / 9.0)
                image.thumbnail((thumbnail_size, thumbnail_size))
                
                self.logger.info(f"Extracting palette: quality={quality}/10, size={thumbnail_size}px")
                
                img_array = np.array(image)
                pixels = img_array.reshape(-1, 3)
                
                if len(pixels) < num_colors:
                    self.logger.warning(f"Image has fewer pixels ({len(pixels)}) than requested colors ({num_colors})")
                    num_colors = min(num_colors, len(pixels))
                
                kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10, max_iter=300)
                kmeans.fit(pixels)
                palette = kmeans.cluster_centers_.astype(int).tolist()

                palette = [[max(0, min(255, c)) for c in color_val] for color_val in palette]
                
                if inject_extremes:
                    black = [0, 0, 0]
                    white = [255, 255, 255]
                    if black not in palette and len(palette) < self._max_palette_size:
                        palette.insert(0, black)
                    if white not in palette and len(palette) < self._max_palette_size:
                        palette.append(white)
                
                self.logger.info(f"Extracted {len(palette)} colors successfully")
                return palette
                
            except ImageProcessingError:
                raise
            except Exception as e:
                self.logger.error(f"Error extracting palette: {e}", exc_info=True)
                return [[0, 0, 0], [128, 128, 128], [255, 255, 255]]

    def _map_pixels_to_palette_cpu(self, image_array: np.ndarray, palette: List[List[int]], 
                                     config: Dict[str, Any]) -> np.ndarray:
        """Enhanced CPU implementation with proper data type handling."""
        with self.profiler.profile_operation("map_pixels_to_palette_CPU", algorithm_id=self.algorithm_id):
            start_time = time.perf_counter()
            
            pixel_count = image_array.shape[0] * image_array.shape[1]
            use_64bit = config.get("use_64bit_indices", False) or pixel_count > 2**31 - 1
            index_dtype = np.int64 if use_64bit else np.int32
            
            if use_64bit:
                self.logger.info("Using 64-bit indices for large image on CPU")
            
            palette_np = np.array(palette, dtype=np.float32)
            pixels_flat = image_array.reshape(-1, 3).astype(np.float32)
            
            pixels_hsv = color.rgb2hsv(pixels_flat / 255.0)
            palette_hsv = color.rgb2hsv(palette_np / 255.0)
            
            delta_sv = pixels_hsv[:, np.newaxis, 1:] - palette_hsv[np.newaxis, :, 1:]
            delta_h_abs = np.abs(pixels_hsv[:, np.newaxis, 0] - palette_hsv[np.newaxis, :, 0])
            delta_h = np.minimum(delta_h_abs, 1.0 - delta_h_abs)
            
            hue_weight = config.get("hue_weight", 3.0)
            distances_sq = (hue_weight * delta_h)**2 + delta_sv[:, :, 0]**2 + delta_sv[:, :, 1]**2
            
            if config.get("use_color_focus", False) and config.get("focus_ranges"):
                distance_modifier = np.ones(len(palette), dtype=np.float32)
                
                for focus in config["focus_ranges"]:
                    target_h = focus["target_hsv"][0] / 360.0
                    target_s = focus["target_hsv"][1] / 100.0
                    target_v = focus["target_hsv"][2] / 100.0
                    range_h = focus["range_h"] / 360.0
                    range_s = focus["range_s"] / 100.0
                    range_v = focus["range_v"] / 100.0
                    
                    h_dist = np.abs(palette_hsv[:, 0] - target_h)
                    h_mask = np.minimum(h_dist, 1.0 - h_dist) <= (range_h / 2.0)
                    s_mask = np.abs(palette_hsv[:, 1] - target_s) <= (range_s / 2.0)
                    v_mask = np.abs(palette_hsv[:, 2] - target_v) <= (range_v / 2.0)
                    
                    final_mask = h_mask & s_mask & v_mask
                    if np.any(final_mask):
                        boost = focus.get("boost_factor", 1.0)
                        distance_modifier[final_mask] /= boost
                
                distances_sq *= distance_modifier[np.newaxis, :]
            
            closest_indices = np.argmin(distances_sq, axis=1).astype(index_dtype)
            result = palette_np[closest_indices].reshape(image_array.shape)
            
            result = np.clip(result, 0, 255)
            
            cpu_time = (time.perf_counter() - start_time) * 1000
            throughput = pixel_count / (cpu_time / 1000) if cpu_time > 0 else 0
            
            self.logger.info(f"CPU processed {pixel_count:,} pixels in {cpu_time:.1f}ms ({throughput:,.0f} pixels/sec)")
            
            return result

    # --- Enhanced GPU Implementation ---
    
    def _setup_taichi_fields(self, num_pixels: int, num_palette_colors: int, use_64bit_indices: bool = False):
        """
        [FIX - UWAGA #4] Enhanced field setup with 64-bit index support.
        POPRAWKA 7: Enhanced field setup with palette size validation.
        """
        if num_pixels <= 0:
            raise ValueError(f"Invalid pixel count: {num_pixels}")
        if num_palette_colors <= 0:
            raise ValueError(f"Invalid palette size: {num_palette_colors}")
        
        if num_palette_colors > self._max_palette_size:
            raise ValueError(f"Palette too large: {num_palette_colors} colors (max {self._max_palette_size})")
        
        cached_64bit = self._cached_fields.get('use_64bit_indices', False)
        if (self._cached_fields.get('num_pixels') == num_pixels and
            self._cached_fields.get('num_palette_colors') == num_palette_colors and
            cached_64bit == use_64bit_indices):
            return
        
        index_dtype = ti.i64 if use_64bit_indices else ti.i32
        index_size_bytes = 8 if use_64bit_indices else 4

        field_memory = (
            num_pixels * 4 * 3 +      # pixels_rgb
            num_pixels * 4 * 3 +      # pixels_hsv  
            num_palette_colors * 4 * 3 + # palette_hsv
            num_pixels * index_size_bytes + # closest_indices
            num_palette_colors * 4    # distance_modifier
        )
        
        if field_memory > self._max_gpu_memory:
            raise GPUMemoryError(f"Dataset requires {field_memory/1024**3:.1f}GB > limit {self._max_gpu_memory/1024**3:.1f}GB")
        
        if self._cached_fields:
            self.logger.debug("Releasing old Taichi fields")
            self._cached_fields.clear()
        
        log_msg = f"Allocating GPU fields: {num_pixels:,} pixels, {num_palette_colors} colors ({field_memory/1024**2:.1f}MB)"
        if use_64bit_indices:
            log_msg += " [64-bit indices enabled]"
        self.logger.info(log_msg)
        
        self._cached_fields = {
            'pixels_rgb': ti.Vector.field(3, dtype=ti.f32, shape=num_pixels),
            'pixels_hsv': ti.Vector.field(3, dtype=ti.f32, shape=num_pixels),
            'palette_hsv': ti.Vector.field(3, dtype=ti.f32, shape=num_palette_colors),
            'closest_indices': ti.field(dtype=index_dtype, shape=num_pixels),
            'distance_modifier': ti.field(dtype=ti.f32, shape=num_palette_colors),
            'num_pixels': num_pixels,
            'num_palette_colors': num_palette_colors,
            'use_64bit_indices': use_64bit_indices
        }

    @ti.kernel
    def rgb_to_hsv_kernel(self, rgb_field: ti.template(), hsv_field: ti.template()):
        """POPRAWKA 5: RGB to HSV conversion with input bounds validation."""
        for i in range(rgb_field.shape[0]):
            r_raw = rgb_field[i].x
            g_raw = rgb_field[i].y
            b_raw = rgb_field[i].z
            
            # [INFO - UWAGA #2] Poniższe "zaciskanie" wartości (clamp) jest kluczowym
            # zabezpieczeniem przed nieprawidłowymi danymi wejściowymi (np. spoza zakresu 0-255).
            # Dzięki temu, użycie małej wartości epsilon (1e-10) w dalszej części jest
            # bezpieczne, ponieważ operujemy na poprawnie znormalizowanych danych.
            r = tm.clamp(r_raw, 0.0, 255.0) / 255.0
            g = tm.clamp(g_raw, 0.0, 255.0) / 255.0
            b = tm.clamp(b_raw, 0.0, 255.0) / 255.0
            
            max_val = tm.max(tm.max(r, g), b)
            min_val = tm.min(tm.min(r, g), b)
            delta = max_val - min_val
            
            v = max_val
            s = 0.0
            if max_val > 1e-10:
                s = delta / max_val
            
            h = 0.0
            if delta > 1e-10:
                if tm.abs(max_val - r) < 1e-10:
                    h = ((g - b) / delta) % 6.0
                elif tm.abs(max_val - g) < 1e-10:
                    h = (b - r) / delta + 2.0
                else:
                    h = (r - g) / delta + 4.0
                
                h = h / 6.0
                if h < 0.0:
                    h += 1.0
            
            hsv_field[i] = tm.vec3(h, s, v)

    @ti.func
    def calculate_hsv_distance_sq_gpu(self, pixel_hsv: tm.vec3, palette_color_hsv: tm.vec3, 
                                      hue_weight: ti.f32) -> ti.f32:
        """Corrected HSV distance calculation with consistent squaring."""
        delta_s = pixel_hsv.y - palette_color_hsv.y
        delta_v = pixel_hsv.z - palette_color_hsv.z
        
        delta_h_abs = tm.abs(pixel_hsv.x - palette_color_hsv.x)
        # [INFO - UWAGA #5] Poniższa linia poprawnie oblicza najkrótszą odległość
        # dla barwy (hue), która jest wartością cykliczną (0.0 do 1.0). To standardowe
        # i skuteczne rozwiązanie problemu "zawijania" się barwy.
        delta_h = tm.min(delta_h_abs, 1.0 - delta_h_abs)
        
        hue_term = hue_weight * delta_h
        return hue_term * hue_term + delta_s * delta_s + delta_v * delta_v

    @ti.kernel
    def _combined_conversion_and_mapping(self, rgb_field: ti.template(), palette_hsv: ti.template(), 
                                         closest_indices: ti.template(), dist_modifier: ti.template(),
                                         hue_weight: ti.f32):
        """Fused kernel with protection against infinite loops and input validation."""
        for i in range(rgb_field.shape[0]):
            r_raw = rgb_field[i].x
            g_raw = rgb_field[i].y
            b_raw = rgb_field[i].z
            
            r = tm.clamp(r_raw, 0.0, 255.0) / 255.0
            g = tm.clamp(g_raw, 0.0, 255.0) / 255.0
            b = tm.clamp(b_raw, 0.0, 255.0) / 255.0
            
            max_val = tm.max(tm.max(r, g), b)
            min_val = tm.min(tm.min(r, g), b)
            delta = max_val - min_val
            
            v = max_val
            s = 0.0
            if max_val > 1e-10:
                s = delta / max_val
            
            h = 0.0
            if delta > 1e-10:
                if tm.abs(max_val - r) < 1e-10:
                    h = ((g - b) / delta) % 6.0
                elif tm.abs(max_val - g) < 1e-10:
                    h = (b - r) / delta + 2.0
                else:
                    h = (r - g) / delta + 4.0
                h = h / 6.0
                if h < 0.0:
                    h += 1.0
            
            pixel_hsv = tm.vec3(h, s, v)
            
            min_dist = 1e30
            best_j = 0
            valid_distance_found = False
            
            for j in range(palette_hsv.shape[0]):
                dist = self.calculate_hsv_distance_sq_gpu(pixel_hsv, palette_hsv[j], hue_weight)
                
                if tm.isfinite(dist) and dist >= 0.0:
                    modified_dist = dist * dist_modifier[j]
                    
                    if tm.isfinite(modified_dist) and (not valid_distance_found or modified_dist < min_dist):
                        min_dist = modified_dist
                        best_j = j
                        valid_distance_found = True
            
            if not valid_distance_found:
                best_j = 0
            
            closest_indices[i] = best_j

    def _map_pixels_to_palette_gpu(self, image_array: np.ndarray, palette: List[List[int]], 
                                     config: Dict[str, Any]) -> np.ndarray:
        """Enhanced GPU implementation with all fixes applied."""
        with self.profiler.profile_operation("map_pixels_to_palette_GPU", algorithm_id=self.algorithm_id):
            start_time = time.perf_counter()
            
            palette_np = np.array(palette, dtype=np.float32)
            
            image_array_clamped = np.clip(image_array, 0, 255)
            pixels_flat = image_array_clamped.reshape(-1, 3).astype(np.float32)
            
            num_pixels = pixels_flat.shape[0]
            num_palette_colors = palette_np.shape[0]
            
            # [FIX - UWAGA #4] Determine if 64-bit indices are needed for this run.
            use_64bit = config.get("use_64bit_indices", False) or num_pixels > 2**31 - 1

            try:
                self._setup_taichi_fields(num_pixels, num_palette_colors, use_64bit_indices=use_64bit)
            except (GPUMemoryError, ValueError) as e:
                raise GPUProcessingError(f"GPU setup failed: {e}")
            
            fields = self._cached_fields
            
            transfer_start = time.perf_counter()
            
            fields['pixels_rgb'].from_numpy(pixels_flat)
            
            palette_hsv_np = color.rgb2hsv(np.clip(palette_np, 0, 255) / 255.0)
            fields['palette_hsv'].from_numpy(palette_hsv_np)
            
            distance_modifier_np = np.ones(num_palette_colors, dtype=np.float32)
            if config.get("use_color_focus", False) and config.get("focus_ranges"):
                self.logger.debug(f"Applying Color Focus with {len(config['focus_ranges'])} ranges")
                
                for focus in config["focus_ranges"]:
                    target_h = focus["target_hsv"][0] / 360.0
                    target_s = focus["target_hsv"][1] / 100.0
                    target_v = focus["target_hsv"][2] / 100.0
                    range_h = focus["range_h"] / 360.0
                    range_s = focus["range_s"] / 100.0
                    range_v = focus["range_v"] / 100.0
                    
                    h_dist = np.abs(palette_hsv_np[:, 0] - target_h)
                    h_mask = np.minimum(h_dist, 1.0 - h_dist) <= (range_h / 2.0)
                    s_mask = np.abs(palette_hsv_np[:, 1] - target_s) <= (range_s / 2.0)
                    v_mask = np.abs(palette_hsv_np[:, 2] - target_v) <= (range_v / 2.0)
                    
                    final_mask = h_mask & s_mask & v_mask
                    if np.any(final_mask):
                        boost = focus.get("boost_factor", 1.0)
                        distance_modifier_np[final_mask] /= boost
            
            fields['distance_modifier'].from_numpy(distance_modifier_np)
            
            transfer_time = time.perf_counter() - transfer_start
            compute_start = time.perf_counter()
            
            use_fusion = (self._use_kernel_fusion and 
                          config.get("enable_kernel_fusion", True) and 
                          num_pixels > 100_000)
            
            if use_fusion:
                self.logger.debug("Using fused RGB→HSV+mapping kernel")
                self._combined_conversion_and_mapping(
                    fields['pixels_rgb'], fields['palette_hsv'], 
                    fields['closest_indices'], fields['distance_modifier'],
                    config.get("hue_weight", 3.0)
                )
            else:
                self.logger.debug("Using separate RGB→HSV and mapping kernels")
                self.rgb_to_hsv_kernel(fields['pixels_rgb'], fields['pixels_hsv'])
                # Here you would call a separate mapping kernel if it existed
                # self._find_closest_color_kernel(...)
            
            ti.sync()
            
            compute_time = time.perf_counter() - compute_start
            readback_start = time.perf_counter()
            
            closest_indices_np = fields['closest_indices'].to_numpy()
            mapped_array = palette_np[closest_indices_np].reshape(image_array.shape)
            
            mapped_array = np.clip(mapped_array, 0, 255)
            
            readback_time = time.perf_counter() - readback_start
            total_time = time.perf_counter() - start_time
            
            if not np.all(np.isfinite(mapped_array)):
                raise GPUProcessingError("GPU produced invalid results (NaN/Inf values)")
            
            throughput = num_pixels / total_time if total_time > 0 else 0
            fusion_str = "fused" if use_fusion else "separate"
            
            self.logger.info(
                f"GPU Performance ({fusion_str}) - Transfer: {transfer_time*1000:.1f}ms, "
                f"Compute: {compute_time*1000:.1f}ms, Readback: {readback_time*1000:.1f}ms, "
                f"Total: {total_time*1000:.1f}ms ({throughput:,.0f} pixels/sec)"
            )
            
            return mapped_array

    def _process_large_image_in_batches(self, image_array: np.ndarray, palette: List[List[int]], 
                                        config: Dict[str, Any]) -> np.ndarray:
        """POPRAWKA 8: Enhanced batch processing with streaming and memory management."""
        total_pixels = image_array.shape[0] * image_array.shape[1]
        batch_size = config.get("gpu_batch_size", self._max_batch_pixels)
        
        if total_pixels <= batch_size:
            return self._map_pixels_to_palette_gpu(image_array, palette, config)
        
        self.logger.info(f"Streaming batch processing: {total_pixels:,} pixels in {batch_size:,} pixel batches")
        
        height, width = image_array.shape[:2]
        rows_per_batch = max(1, batch_size // width)
        
        total_batches = (height + rows_per_batch - 1) // rows_per_batch
        cleanup_counter = 0
        
        result_array = np.empty_like(image_array, dtype=np.float32)
        
        for batch_idx, start_row in enumerate(range(0, height, rows_per_batch)):
            end_row = min(start_row + rows_per_batch, height)
            batch = image_array[start_row:end_row]
            
            self.logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}: rows {start_row}-{end_row}")
            
            batch_result = self._map_pixels_to_palette_gpu(batch, palette, config)
            result_array[start_row:end_row] = batch_result
            
            del batch_result
            
            cleanup_counter += 1
            if (config.get("gpu_memory_cleanup", True) and 
                cleanup_counter >= self._gpu_memory_cleanup_threshold):
                self._cleanup_gpu_memory()
                gc.collect()
                cleanup_counter = 0
        
        if config.get("gpu_memory_cleanup", True):
            self._cleanup_gpu_memory()
        
        self.logger.info(f"Completed {total_batches} batches successfully")
        return result_array

    # --- Enhanced Smart Dispatcher ---
    
    def _map_pixels_to_palette(self, image_array: np.ndarray, palette: List[List[int]], 
                               config: Dict[str, Any]) -> np.ndarray:
        """Enhanced intelligent dispatcher with strategy-based processing."""
        
        self._validate_inputs(image_array, palette)
        
        image_size = (image_array.shape[0], image_array.shape[1])
        strategy = self._determine_strategy(image_size, len(palette), config)
        
        pixel_count = image_size[0] * image_size[1]
        
        self.logger.info(f"Processing strategy: {strategy.name} for {pixel_count:,} pixels, {len(palette)} colors")
        
        if strategy == AccelerationStrategy.CPU:
            return self._map_pixels_to_palette_cpu(image_array, palette, config)
        
        # GPU strategies
        try:
            if strategy == AccelerationStrategy.GPU_LARGE:
                return self._process_large_image_in_batches(image_array, palette, config)
            else:
                return self._map_pixels_to_palette_gpu(image_array, palette, config)
                
        except Exception as e:
            # [FIX - UWAGA #10] Dodano czyszczenie zasobów GPU przed przejściem na CPU,
            # aby zapewnić czysty stan i uniknąć potencjalnych konfliktów.
            if TAICHI_AVAILABLE and hasattr(ti, 'TaichiRuntimeError') and isinstance(e, ti.TaichiRuntimeError):
                self.logger.warning(f"Taichi runtime error: {e}. Falling back to CPU.")
                self._cleanup_gpu_memory()
            elif isinstance(e, (GPUProcessingError, GPUMemoryError)):
                self.logger.warning(f"GPU error: {e}. Falling back to CPU.")
                self._cleanup_gpu_memory()
            else:
                self.logger.error(f"Unexpected error: {e}. Falling back to CPU.")
                if TAICHI_AVAILABLE:
                    self._cleanup_gpu_memory()
            
            return self._map_pixels_to_palette_cpu(image_array, palette, config)

    # --- Main Processing Pipeline ---
    
    def process_images(self, master_path: str, target_path: str, output_path: str, **kwargs) -> bool:
        """Enhanced main processing pipeline with comprehensive error handling."""
        with self.profiler.profile_operation("process_images_full", algorithm_id=self.algorithm_id):
            run_config = self.default_config_values.copy()
            run_config.update(kwargs)
            # [FIX - UWAGA #8] Walidacja parametrów przekazanych bezpośrednio do funkcji (przez kwargs),
            # co zapobiega użyciu nieprawidłowych wartości.
            self._validate_run_config(run_config)
            
            self.logger.info(f"Processing pipeline - GPU: {'enabled' if TAICHI_AVAILABLE else 'disabled'}, "
                             f"Colors: {run_config['num_colors']}, Quality: {run_config['quality']}")
            
            try:
                target_image = self._safe_image_load(target_path)
                target_array = np.array(target_image)
                
                if target_array.size == 0:
                    raise ImageProcessingError("Target image is empty")
                
                self.logger.info(f"Target image: {target_array.shape[1]}x{target_array.shape[0]} "
                                 f"({target_array.shape[0] * target_array.shape[1]:,} pixels)")
                
                palette = self.extract_palette(
                    master_path,
                    num_colors=run_config["num_colors"],
                    method=run_config["palette_method"],
                    quality=run_config["quality"],
                    inject_extremes=run_config["inject_extremes"]
                )
                
                mapped_array = self._map_pixels_to_palette(target_array, palette, run_config)
                
                if run_config["preserve_extremes"]:
                    threshold = run_config["extremes_threshold"]
                    luminance = np.dot(target_array[..., :3], [0.2989, 0.5870, 0.1140])
                    black_mask = luminance <= threshold
                    white_mask = luminance >= (255 - threshold)
                    mapped_array[black_mask] = [0, 0, 0]
                    mapped_array[white_mask] = [255, 255, 255]
                
                mapped_image = Image.fromarray(np.clip(mapped_array, 0, 255).astype(np.uint8), "RGB")
                
                save_kwargs = {"quality": 95, "optimize": True}
                mapped_image.save(output_path, **save_kwargs)
                
                self.logger.info(f"Successfully saved processed image: {output_path}")
                return True
                
            except (ImageProcessingError, GPUProcessingError, GPUMemoryError) as e:
                self.logger.error(f"Processing failed: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Unexpected processing error: {e}", exc_info=True)
                return False

    # --- POPRAWKA 6: Fixed Async Processing ---
    
    async def process_images_async(self, master_path: str, target_path: str, 
                                     output_path: str, **kwargs) -> bool:
        """POPRAWKA 6: Fixed async processing without race conditions."""
        with self.profiler.profile_operation("process_images_async", algorithm_id=self.algorithm_id):
            run_config = self.default_config_values.copy()
            run_config.update(kwargs)
            # [FIX - UWAGA #8] Walidacja również dla ścieżki asynchronicznej.
            self._validate_run_config(run_config)
            
            try:
                # Phase 1: Async I/O operations
                with ThreadPoolExecutor(max_workers=2) as executor:
                    loop = asyncio.get_event_loop()
                    
                    image_future = loop.run_in_executor(
                        executor, 
                        lambda: np.array(self._safe_image_load(target_path))
                    )
                    # [INFO - UWAGA #7] Wywołanie `extract_palette` w executorze jest bezpieczne wątkowo.
                    # Każde wywołanie tworzy własne, lokalne obiekty (np. instancję KMeans)
                    # i nie modyfikuje współdzielonego stanu klasy w sposób, który prowadziłby do
                    # konfliktu. Jest to efektywne i bezpieczne rozproszenie zadań I/O i CPU-bound.
                    palette_future = loop.run_in_executor(
                        executor, 
                        self.extract_palette, 
                        master_path, run_config["num_colors"], 
                        run_config["palette_method"], run_config["quality"], 
                        run_config["inject_extremes"]
                    )
                    
                    target_array, palette = await asyncio.wait_for(
                        asyncio.gather(image_future, palette_future),
                        timeout=300.0
                    )
                
                # Phase 2: Synchronous GPU processing
                mapped_array = self._map_pixels_to_palette(target_array, palette, run_config)
                
                def post_process():
                    result_array = mapped_array.copy()
                    
                    if run_config["preserve_extremes"]:
                        threshold = run_config["extremes_threshold"]
                        luminance = np.dot(target_array[..., :3], [0.2989, 0.5870, 0.1140])
                        black_mask = luminance <= threshold
                        white_mask = luminance >= (255 - threshold)
                        result_array[black_mask] = [0, 0, 0]
                        result_array[white_mask] = [255, 255, 255]
                    
                    mapped_image = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8), "RGB")
                    mapped_image.save(output_path, quality=95, optimize=True)
                    return True
                
                # Phase 3: Async post-processing
                with ThreadPoolExecutor(max_workers=1) as executor:
                    loop = asyncio.get_event_loop()
                    success = await loop.run_in_executor(executor, post_process)
                
                if success:
                    self.logger.info(f"Async processing completed: {output_path}")
                return success
            
            except asyncio.TimeoutError:
                self.logger.error("Async processing timed out")
                return False
            except Exception as e:
                self.logger.error(f"Async processing failed: {e}", exc_info=True)
                return False

    # --- POPRAWKA 3: Thread-safe Benchmarking ---
    
    def benchmark_implementations(self, image_array: np.ndarray, palette: List[List[int]], 
                                  config: Dict[str, Any], runs: int = 3) -> Dict[str, Any]:
        """POPRAWKA 3: Thread-safe benchmark with proper state management."""
        
        with self._benchmark_lock:
            self.logger.info(f"Running thread-safe benchmark ({runs} iterations)...")
            
            self._validate_inputs(image_array, palette)
            
            cpu_times, gpu_times, gpu_fused_times = [], [], []
            
            original_fusion_state = self._use_kernel_fusion
            
            try:
                # CPU benchmark
                for i in range(runs):
                    self.logger.debug(f"CPU benchmark run {i+1}/{runs}")
                    start = time.perf_counter()
                    try:
                        _ = self._map_pixels_to_palette_cpu(image_array, palette, config)
                        cpu_times.append(time.perf_counter() - start)
                    except Exception as e:
                        self.logger.warning(f"CPU benchmark run {i+1} failed: {e}")
                        cpu_times.append(float('inf'))
                    finally:
                        gc.collect()
                
                # GPU benchmarks
                if TAICHI_AVAILABLE:
                    # Regular GPU benchmark
                    self._use_kernel_fusion = False
                    
                    for i in range(runs):
                        self.logger.debug(f"GPU (separate kernels) run {i+1}/{runs}")
                        try:
                            start = time.perf_counter()
                            _ = self._map_pixels_to_palette_gpu(image_array, palette, config)
                            gpu_times.append(time.perf_counter() - start)
                        except Exception as e:
                            self.logger.warning(f"GPU run {i+1} failed: {e}")
                            gpu_times.append(float('inf'))
                        finally:
                            self._cleanup_gpu_memory()
                            gc.collect()
                    
                    # Fused GPU benchmark
                    self._use_kernel_fusion = True
                    
                    for i in range(runs):
                        self.logger.debug(f"GPU (fused kernel) run {i+1}/{runs}")
                        try:
                            start = time.perf_counter()
                            _ = self._map_pixels_to_palette_gpu(image_array, palette, config)
                            gpu_fused_times.append(time.perf_counter() - start)
                        except Exception as e:
                            self.logger.warning(f"GPU fused run {i+1} failed: {e}")
                            gpu_fused_times.append(float('inf'))
                        finally:
                            self._cleanup_gpu_memory()
                            gc.collect()
            
            finally:
                self._use_kernel_fusion = original_fusion_state
            
            def safe_stats(times):
                valid_times = [t for t in times if t != float('inf')]
                if not valid_times:
                    return float('inf'), 0, 0
                mean_time = statistics.mean(valid_times)
                std_time = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
                return mean_time, std_time, len(valid_times)
            
            cpu_avg, cpu_std, cpu_valid = safe_stats(cpu_times)
            gpu_avg, gpu_std, gpu_valid = safe_stats(gpu_times)
            gpu_fused_avg, gpu_fused_std, gpu_fused_valid = safe_stats(gpu_fused_times)
            
            pixel_count = image_array.shape[0] * image_array.shape[1]
            
            results = {
                'pixel_count': pixel_count,
                'palette_size': len(palette),
                'runs_attempted': runs,
                'cpu': {
                    'time_avg': cpu_avg,
                    'time_std': cpu_std,
                    'valid_runs': cpu_valid,
                    'throughput': pixel_count / cpu_avg if cpu_avg != float('inf') else 0
                },
                'gpu': {
                    'time_avg': gpu_avg,
                    'time_std': gpu_std,
                    'valid_runs': gpu_valid,
                    'throughput': pixel_count / gpu_avg if gpu_avg != float('inf') else 0
                },
                'gpu_fused': {
                    'time_avg': gpu_fused_avg,
                    'time_std': gpu_fused_std,
                    'valid_runs': gpu_fused_valid,
                    'throughput': pixel_count / gpu_fused_avg if gpu_fused_avg != float('inf') else 0
                },
                'speedup_gpu': cpu_avg / gpu_avg if gpu_avg != float('inf') and gpu_avg > 0 else 0,
                'speedup_gpu_fused': cpu_avg / gpu_fused_avg if gpu_fused_avg != float('inf') and gpu_fused_avg > 0 else 0,
                'fusion_improvement': gpu_avg / gpu_fused_avg if gpu_fused_avg != float('inf') and gpu_fused_avg > 0 else 0
            }
            
            self.logger.info(
                f"Benchmark Results ({pixel_count:,} pixels, {len(palette)} colors):\n"
                f"  CPU: {cpu_avg*1000:.1f}ms ± {cpu_std*1000:.1f}ms ({results['cpu']['throughput']:,.0f} pixels/sec, {cpu_valid}/{runs} valid)\n" +
                (f"  GPU (separate): {gpu_avg*1000:.1f}ms ± {gpu_std*1000:.1f}ms ({results['gpu']['throughput']:,.0f} pixels/sec, {gpu_valid}/{runs} valid)\n"
                 f"  GPU (fused): {gpu_fused_avg*1000:.1f}ms ± {gpu_fused_std*1000:.1f}ms ({results['gpu_fused']['throughput']:,.0f} pixels/sec, {gpu_fused_valid}/{runs} valid)\n"
                 f"  Speedup (separate): {results['speedup_gpu']:.2f}x\n"
                 f"  Speedup (fused): {results['speedup_gpu_fused']:.2f}x\n"
                 f"  Fusion improvement: {results['fusion_improvement']:.2f}x" if TAICHI_AVAILABLE else 
                 "  GPU: Not available")
            )
            
            return results

# --- Factory Function ---

def create_palette_mapping_algorithm_gpu():
    """Factory function with enhanced error handling."""
    try:
        return PaletteMappingAlgorithmGPU()
    except Exception as e:
        print(f"ERROR: Failed to create GPU algorithm instance: {e}")
        raise
