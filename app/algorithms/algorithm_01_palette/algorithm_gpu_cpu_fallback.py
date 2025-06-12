# /app/algorithms/algorithm_01_palette/algorithm-gpu-cpu-fallback.py
# Moduł zawierający logikę awaryjną (CPU) oraz operacje zawsze wykonywane na CPU.

import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Poprawiony import wyjątków
from .algorithm_gpu_exceptions import ImageProcessingError

# Sprawdzenie dostępności zależności
try:
    from skimage import color as skimage_color
    # Eksport funkcji konwersji kolorów do użycia w innych modułach
    rgb2hsv = skimage_color.rgb2hsv
    from sklearn.cluster import KMeans
    SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_SKLEARN_AVAILABLE = False
    # Definicja "pustej" funkcji, aby uniknąć AttributeError
    def rgb2hsv(x): raise NotImplementedError("scikit-image is not installed")

# --- Operacje na obrazach i paletach (CPU) ---

def safe_image_load(image_path: str, logger: Any) -> Image.Image:
    """Bezpiecznie wczytuje obraz z pliku, konwertuje do RGB i obsługuje błędy."""
    try:
        path_obj = Path(image_path)
        if not path_obj.exists(): raise ImageProcessingError(f"Image file not found: {image_path}")
        if not path_obj.is_file(): raise ImageProcessingError(f"Path is not a file: {image_path}")
        if path_obj.stat().st_size == 0: raise ImageProcessingError(f"Image file is empty: {image_path}")

        if path_obj.stat().st_size > 500 * 1024 * 1024:
            logger.warning(f"Very large image file: {path_obj.stat().st_size / 1024**2:.1f}MB")

        image = Image.open(image_path)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    except Exception as e:
        raise ImageProcessingError(f"Failed to load image {image_path}: {e}")

def extract_palette(image: Image.Image, num_colors: int, quality: int, inject_extremes: bool, max_palette_size: int, logger: Any) -> List[List[int]]:
    """Wyodrębnia paletę kolorów z obrazu przy użyciu KMeans na CPU."""
    if not SCIPY_SKLEARN_AVAILABLE:
        logger.error("Scikit-learn is required for palette extraction.")
        return [[0,0,0], [255,255,255]]

    base_size, max_size = 100, 1000
    thumbnail_size = int(base_size + (max_size - base_size) * (quality - 1) / 9.0)
    image.thumbnail((thumbnail_size, thumbnail_size))
    
    pixels = np.array(image).reshape(-1, 3)
    if len(pixels) < num_colors:
        num_colors = len(pixels)

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    palette = kmeans.cluster_centers_.astype(int).tolist()

    if inject_extremes:
        if [0, 0, 0] not in palette and len(palette) < max_palette_size: palette.insert(0, [0, 0, 0])
        if [255, 255, 255] not in palette and len(palette) < max_palette_size: palette.append([255, 255, 255])
    
    return palette

def map_pixels_to_palette_cpu(image_array: np.ndarray, palette: List[List[int]], config: Dict[str, Any], logger: Any) -> np.ndarray:
    """Implementacja mapowania pikseli do palety na CPU."""
    if not SCIPY_SKLEARN_AVAILABLE:
        logger.error("Scikit-image is required for CPU color conversion.")
        return image_array 

    start_time = time.perf_counter()
    pixel_count = image_array.shape[0] * image_array.shape[1]
    
    palette_np = np.array(palette, dtype=np.float32)
    pixels_flat = image_array.reshape(-1, 3).astype(np.float32)

    pixels_hsv = color.rgb2hsv(pixels_flat / 255.0)
    palette_hsv = color.rgb2hsv(palette_np / 255.0)

    delta_h = np.abs(pixels_hsv[:, np.newaxis, 0] - palette_hsv[np.newaxis, :, 0])
    delta_h = np.minimum(delta_h, 1.0 - delta_h)
    
    distances_sq = (config['hue_weight'] * delta_h)**2 + (pixels_hsv[:, np.newaxis, 1] - palette_hsv[np.newaxis, :, 1])**2 + (pixels_hsv[:, np.newaxis, 2] - palette_hsv[np.newaxis, :, 2])**2
    
    closest_indices = np.argmin(distances_sq, axis=1)
    result = palette_np[closest_indices].reshape(image_array.shape)

    logger.info(f"CPU processed {pixel_count:,} pixels in {(time.perf_counter() - start_time) * 1000:.1f}ms")
    return np.clip(result, 0, 255)
