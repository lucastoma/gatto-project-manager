# /app/algorithms/algorithm_01_palette/algorithm-gpu-kernels.py
# Moduł zawierający całą logikę związaną z Taichi, w tym inicjalizację i kernele.

import gc
import threading
from typing import Any, Dict

# --- Inicjalizacja Taichi ---
# Ten blok jest jedynym miejscem, gdzie importowany jest taichi.
TAICHI_AVAILABLE = False
GPU_BACKEND = "none"
ti, tm = None, None
_taichi_lock = threading.Lock()

def _safe_taichi_cleanup():
    """Bezpiecznie czyści kontekst Taichi, jeśli jest to konieczne."""
    global ti
    try:
        if ti is not None and hasattr(ti, 'reset'):
            ti.reset()
    except Exception as e:
        print(f"Warning: Taichi cleanup failed: {e}")

def safe_taichi_init():
    """Bezpieczna inicjalizacja Taichi z odpowiednim zarządzaniem zasobami."""
    global TAICHI_AVAILABLE, GPU_BACKEND, ti, tm
    
    with _taichi_lock:
        try:
            import taichi as ti_module
            import taichi.math as tm_module
            
            # Próba inicjalizacji GPU
            try:
                ti_module.init(arch=ti_module.gpu, log_level=ti_module.WARN)
                GPU_BACKEND = str(ti_module.lang.impl.current_cfg().arch)
                ti, tm = ti_module, tm_module
                TAICHI_AVAILABLE = True
                print(f"SUCCESS: Taichi GPU initialized with backend: {GPU_BACKEND}")
                return True
            except Exception as e_gpu:
                print(f"WARNING: GPU initialization failed: {e_gpu}. Trying CPU fallback.")
                _safe_taichi_cleanup()
                ti, tm = None, None
                
                # Próba fallbacku do CPU
                try:
                    ti_module.init(arch=ti_module.cpu, log_level=ti_module.WARN)
                    ti, tm = ti_module, tm_module
                    TAICHI_AVAILABLE = True # Taichi jest dostępne, ale na CPU
                    GPU_BACKEND = "cpu"
                    print("Taichi initialized with CPU backend as a fallback.")
                    return False # Zwraca False, by zasygnalizować brak akceleracji GPU
                except Exception as e_cpu:
                    print(f"ERROR: Complete Taichi initialization failed: {e_cpu}")
                    TAICHI_AVAILABLE = False
                    return False
        except ImportError:
            print("WARNING: Taichi not available.")
            TAICHI_AVAILABLE = False
            return False

# --- Zarządzanie pamięcią GPU ---

def setup_taichi_fields(cached_fields: Dict, num_pixels: int, num_palette_colors: int, use_64bit_indices: bool, max_gpu_memory: int, max_palette_size: int, logger: Any):
    """Zarządza alokacją pól Taichi w pamięci GPU."""
    from .algorithm-gpu-utils import GPUMemoryError

    if num_pixels <= 0 or num_palette_colors <= 0 or num_palette_colors > max_palette_size:
        raise ValueError("Invalid dimensions for Taichi fields")

    cached_64bit = cached_fields.get('use_64bit_indices', False)
    if (cached_fields.get('num_pixels') == num_pixels and
        cached_fields.get('num_palette_colors') == num_palette_colors and
        cached_64bit == use_64bit_indices):
        return cached_fields

    index_dtype = ti.i64 if use_64bit_indices else ti.i32
    index_size_bytes = 8 if use_64bit_indices else 4

    field_memory = (
        num_pixels * 4 * 3 +  # pixels_rgb
        num_pixels * 4 * 3 +  # pixels_hsv
        num_palette_colors * 4 * 3 + # palette_hsv
        num_pixels * index_size_bytes + # closest_indices
        num_palette_colors * 4 # distance_modifier
    )

    if field_memory > max_gpu_memory:
        raise GPUMemoryError(f"Dataset requires {field_memory/1024**3:.1f}GB > limit {max_gpu_memory/1024**3:.1f}GB")

    log_msg = f"Allocating GPU fields: {num_pixels:,} pixels, {num_palette_colors} colors ({field_memory/1024**2:.1f}MB)"
    if use_64bit_indices:
        log_msg += " [64-bit indices enabled]"
    logger.info(log_msg)

    return {
        'pixels_rgb': ti.Vector.field(3, dtype=ti.f32, shape=num_pixels),
        'pixels_hsv': ti.Vector.field(3, dtype=ti.f32, shape=num_pixels),
        'palette_hsv': ti.Vector.field(3, dtype=ti.f32, shape=num_palette_colors),
        'closest_indices': ti.field(dtype=index_dtype, shape=num_pixels),
        'distance_modifier': ti.field(dtype=ti.f32, shape=num_palette_colors),
        'num_pixels': num_pixels,
        'num_palette_colors': num_palette_colors,
        'use_64bit_indices': use_64bit_indices
    }

def cleanup_gpu_memory(cached_fields: Dict, logger: Any):
    """Czyści pola Taichi i wymusza zwolnienie pamięci."""
    if TAICHI_AVAILABLE and cached_fields:
        try:
            cached_fields.clear()
            gc.collect()
            ti.sync()
            logger.debug("GPU memory cleanup completed")
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")

# --- Kernele Taichi ---

@ti.func
def calculate_hsv_distance_sq_gpu(pixel_hsv: tm.vec3, palette_color_hsv: tm.vec3, hue_weight: ti.f32) -> ti.f32:
    delta_s = pixel_hsv.y - palette_color_hsv.y
    delta_v = pixel_hsv.z - palette_color_hsv.z
    delta_h_abs = tm.abs(pixel_hsv.x - palette_color_hsv.x)
    delta_h = tm.min(delta_h_abs, 1.0 - delta_h_abs)
    hue_term = hue_weight * delta_h
    return hue_term * hue_term + delta_s * delta_s + delta_v * delta_v

@ti.kernel
def rgb_to_hsv_kernel(rgb_field: ti.template(), hsv_field: ti.template()):
    for i in range(rgb_field.shape[0]):
        r, g, b = tm.clamp(rgb_field[i], 0.0, 255.0) / 255.0
        max_val, min_val = tm.max(r, g, b), tm.min(r, g, b)
        delta = max_val - min_val
        v = max_val
        s = delta / max_val if max_val > 1e-10 else 0.0
        h = 0.0
        if delta > 1e-10:
            if tm.abs(max_val - r) < 1e-10: h = ((g - b) / delta) % 6.0
            elif tm.abs(max_val - g) < 1e-10: h = (b - r) / delta + 2.0
            else: h = (r - g) / delta + 4.0
            h = h / 6.0 + (1.0 if h < 0.0 else 0.0)
        hsv_field[i] = tm.vec3(h, s, v)

@ti.kernel
def find_closest_color_kernel(pixels_hsv: ti.template(), palette_hsv: ti.template(), closest_indices: ti.template(), dist_modifier: ti.template(), hue_weight: ti.f32):
    for i in range(pixels_hsv.shape[0]):
        pixel_hsv, min_dist, best_j = pixels_hsv[i], 1e30, 0
        valid_found = False
        for j in range(palette_hsv.shape[0]):
            dist = calculate_hsv_distance_sq_gpu(pixel_hsv, palette_hsv[j], hue_weight) * dist_modifier[j]
            if tm.isfinite(dist) and (not valid_found or dist < min_dist):
                min_dist, best_j, valid_found = dist, j, True
        closest_indices[i] = best_j

@ti.kernel
def combined_conversion_and_mapping(rgb_field: ti.template(), palette_hsv: ti.template(), closest_indices: ti.template(), dist_modifier: ti.template(), hue_weight: ti.f32):
    for i in range(rgb_field.shape[0]):
        r, g, b = tm.clamp(rgb_field[i], 0.0, 255.0) / 255.0
        max_val, min_val = tm.max(r, g, b), tm.min(r, g, b)
        delta = max_val - min_val
        v = max_val
        s = delta / max_val if max_val > 1e-10 else 0.0
        h = 0.0
        if delta > 1e-10:
            if tm.abs(max_val - r) < 1e-10: h = ((g - b) / delta) % 6.0
            elif tm.abs(max_val - g) < 1e-10: h = (b - r) / delta + 2.0
            else: h = (r - g) / delta + 4.0
            h = h / 6.0 + (1.0 if h < 0.0 else 0.0)
        
        pixel_hsv, min_dist, best_j = tm.vec3(h, s, v), 1e30, 0
        valid_found = False
        for j in range(palette_hsv.shape[0]):
            dist = calculate_hsv_distance_sq_gpu(pixel_hsv, palette_hsv[j], hue_weight) * dist_modifier[j]
            if tm.isfinite(dist) and (not valid_found or dist < min_dist):
                min_dist, best_j, valid_found = dist, j, True
        closest_indices[i] = best_j
