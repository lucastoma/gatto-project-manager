# /app/algorithms/algorithm_01_palette/algorithm_gpu_kernels.py
# Moduł zawierający klasę do zarządzania pamięcią GPU oraz definicje kerneli Taichi.

import threading
from typing import Any, Dict

# Import Taichi i wyjątków z dedykowanych modułów
from .algorithm_gpu_exceptions import GPUMemoryError
from .algorithm_gpu_taichi_init import ti, tm, TAICHI_AVAILABLE

class GPUMemoryManager:
    """Zarządza alokacją i czyszczeniem pamięci GPU dla pól Taichi w sposób bezpieczny wątkowo."""

    def __init__(self, logger: Any, max_gpu_memory: int, max_palette_size: int):
        self._cached_fields: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.logger = logger
        self.max_gpu_memory = max_gpu_memory
        self.max_palette_size = max_palette_size

    def get_or_create_fields(self, num_pixels: int, num_palette_colors: int, use_64bit_indices: bool) -> Dict[str, Any]:
        """Pobiera lub tworzy (jeśli to konieczne) pola Taichi o zadanych wymiarach."""
        with self._lock:
            # Sprawdź, czy istniejące pola mogą być ponownie użyte
            cached_64bit = self._cached_fields.get('use_64bit_indices', False)
            if (self._cached_fields.get('num_pixels') == num_pixels and
                self._cached_fields.get('num_palette_colors') == num_palette_colors and
                cached_64bit == use_64bit_indices):
                return self._cached_fields

            # Jeśli nie, wyczyść stare pola przed alokacją nowych
            self._cleanup_locked()

            if not TAICHI_AVAILABLE:
                raise GPUMemoryError("Taichi is not available")

            # --- POPRAWIONA LOGIKA ALOKACJI ---
            try:
                # Poprawiony typ danych dla indeksów 64-bitowych
                index_dtype = ti.i64 if use_64bit_indices else ti.i32

                # Poprawna, nowoczesna składnia alokacji pól Taichi (bez .place())
                fields = {
                    'pixels_rgb': ti.Vector.field(3, dtype=ti.f32, shape=num_pixels),
                    'pixels_hsv': ti.Vector.field(3, dtype=ti.f32, shape=num_pixels),
                    'palette_hsv': ti.Vector.field(3, dtype=ti.f32, shape=num_palette_colors),
                    'closest_indices': ti.field(dtype=index_dtype, shape=num_pixels),
                    'distance_modifier': ti.field(dtype=ti.f32, shape=num_palette_colors),
                    'num_pixels': num_pixels,
                    'num_palette_colors': num_palette_colors,
                    'use_64bit_indices': use_64bit_indices
                }
                
                self.logger.info(f"Allocated new Taichi fields for {num_pixels} pixels.")
                self._cached_fields = fields
                return fields

            except Exception as e:
                self._cleanup_locked()
                raise GPUMemoryError(f"Failed to allocate GPU memory: {e}")

    def cleanup(self) -> None:
        """Publiczna metoda do czyszczenia zasobów GPU."""
        with self._lock:
            self._cleanup_locked()

    def _cleanup_locked(self) -> None:
        """Wewnętrzna metoda czyszcząca (musi być wywoływana z aktywną blokadą)."""
        if not self._cached_fields:
            return
            
        # W Taichi nowoczesne pola są zarządzane przez GC Pythona,
        # więc usunięcie referencji jest kluczowe.
        self._cached_fields.clear()
        
        # Dodatkowe kroki dla pewności
        import gc
        gc.collect()
        if ti is not None:
            ti.sync()
        self.logger.debug("GPU memory cleanup executed.")


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
        # Normalizacja z [0,255] do [0,1] nie jest już potrzebna, dane przychodzą znormalizowane
        r, g, b = rgb_field[i]
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
