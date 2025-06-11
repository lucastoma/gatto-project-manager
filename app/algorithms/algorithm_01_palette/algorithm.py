import logging
import numpy as np
from PIL import Image, ImageFilter
import time
import json
from skimage import color
from sklearn.cluster import KMeans
from typing import TYPE_CHECKING, Any, Dict, List

# --- Lepsza obsługa opcjonalnych zależności ---
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

try:
    from scipy.spatial import KDTree
    from scipy import ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    KDTree, ndimage = None, None
    SCIPY_AVAILABLE = False
# --- Koniec obsługi zależności ---

try:
    from ...core.development_logger import get_logger
    from ...core.performance_profiler import get_profiler

    if TYPE_CHECKING:
        from ...core.development_logger import DevelopmentLogger
        from ...core.performance_profiler import PerformanceProfiler
except ImportError:

    def get_logger() -> Any:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    class DummyProfiler:
        def start(self, name):
            pass

        def stop(self, name):
            pass

        def get_report(self):
            return "Profiler not available."

        def profile_operation(self, *args, **kwargs):
            import contextlib

            return contextlib.nullcontext()

    def get_profiler() -> Any:
        return DummyProfiler()


class PaletteMappingAlgorithm:
    def __init__(
        self, config_path: str = None, algorithm_id: str = "algorithm_01_palette"
    ):
        self.algorithm_id = algorithm_id
        if TYPE_CHECKING:
            self.logger: "DevelopmentLogger" = get_logger()
            self.profiler: "PerformanceProfiler" = get_profiler()
        else:
            self.logger = get_logger()
            self.profiler = get_profiler()

        self.logger.info(f"Initialized algorithm: {self.algorithm_id}")
        self.name = "Palette Mapping Refactored"
        self.version = "2.5-ColorFocus"
        self.default_config_values = self._get_default_config()
        self.config = (
            self.load_config(config_path)
            if config_path
            else self.default_config_values.copy()
        )
        if not SCIPY_AVAILABLE:
            self.logger.warning(
                "Scipy not installed. Advanced features (KDTree, Edge Blending) are disabled."
            )

        self.bayer_matrix_8x8 = np.array(
            [
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21],
            ]
        )

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "num_colors": 16,
            "palette_method": "kmeans",
            "quality": 5,
            "distance_metric": "weighted_hsv",
            "hue_weight": 3.0,
            "use_color_focus": False,
            "focus_ranges": [],  # Lista obiektów definiujących zakresy
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
        }

    # ... (metody load_config, validate_palette, extract_palette pozostają bez zmian) ...
    def load_config(self, config_path: str) -> Dict[str, Any]:
        config = self.default_config_values.copy()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            config.update(user_config)
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}, using default.")
            return config

    def validate_palette(self, palette: List[List[int]]):
        if not palette:
            raise ValueError("Palette cannot be empty")
        for i, color_val in enumerate(palette):
            if len(color_val) != 3:
                raise ValueError(f"Color {i} must have 3 components")
            if not all(0 <= c <= 255 for c in color_val):
                raise ValueError(f"Color {i} has values outside 0-255")

    def extract_palette(
        self,
        image_path: str,
        num_colors: int,
        method: str,
        quality: int,
        inject_extremes: bool,
    ) -> List[List[int]]:
        with self.profiler.profile_operation(
            "extract_palette", algorithm_id=self.algorithm_id
        ):
            try:
                image = Image.open(image_path)
                if image.mode == "RGBA":
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")

                base_size, max_size = 100, 1000
                thumbnail_size_val = int(
                    base_size + (max_size - base_size) * (quality - 1) / 9.0
                )
                image.thumbnail((thumbnail_size_val, thumbnail_size_val))
                self.logger.info(
                    f"Analyzing palette (quality: {quality}/10, size: {thumbnail_size_val}px, method: '{method}')"
                )

                if method == "median_cut":
                    quantized_image = image.quantize(
                        colors=num_colors, method=Image.MEDIANCUT, dither=Image.NONE
                    )
                    palette_raw = quantized_image.getpalette()
                    num_actual_colors = len(palette_raw) // 3
                    palette = [
                        list(palette_raw[i * 3 : i * 3 + 3])
                        for i in range(num_actual_colors)
                    ]
                else:
                    img_array = np.array(image)
                    pixels = img_array.reshape(-1, 3)
                    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
                    kmeans.fit(pixels)
                    palette = kmeans.cluster_centers_.astype(int).tolist()

                palette = [
                    [max(0, min(255, c)) for c in color_val] for color_val in palette
                ]
                if inject_extremes:
                    if [0, 0, 0] not in palette:
                        palette.insert(0, [0, 0, 0])
                    if [255, 255, 255] not in palette:
                        palette.append([255, 255, 255])
                self.validate_palette(palette)
                self.logger.info(f"Extracted {len(palette)} colors.")
                return palette
            except Exception as e:
                self.logger.error(f"Error extracting palette: {e}", exc_info=True)
                return [[0, 0, 0], [128, 128, 128], [255, 255, 255]]

    def _calculate_hsv_distance_sq(self, pixels_hsv, palette_hsv, weights):
        """Oblicza kwadrat ważonej odległości w HSV, używając tablicy wag."""
        delta_sv = pixels_hsv[:, np.newaxis, 1:] - palette_hsv[np.newaxis, :, 1:]
        delta_h_abs = np.abs(
            pixels_hsv[:, np.newaxis, 0] - palette_hsv[np.newaxis, :, 0]
        )
        delta_h = np.minimum(delta_h_abs, 1.0 - delta_h_abs)

        # Tworzenie pełnej macierzy delty
        delta_hsv = np.concatenate((delta_h[..., np.newaxis], delta_sv), axis=2)

        # Zastosowanie wag (broadcasting)
        # weights mają kształt (N, 3), delta_hsv ma (N, M, 3) -> weights[:, np.newaxis, :] ma (N, 1, 3)
        weighted_delta_hsv = delta_hsv * weights[:, np.newaxis, :]

        return np.sum(weighted_delta_hsv**2, axis=2)

    def _map_pixels_to_palette(
        self, image_array: np.ndarray, palette: List[List[int]], config: Dict[str, Any]
    ) -> np.ndarray:
        with self.profiler.profile_operation(
            "map_pixels_to_palette", algorithm_id=self.algorithm_id
        ):
            metric = config.get("distance_metric")
            palette_np = np.array(palette, dtype=np.float32)
            pixels_flat = image_array.reshape(-1, 3).astype(np.float32)

            if "hsv" in metric:
                pixels_hsv = color.rgb2hsv(pixels_flat / 255.0)
                palette_hsv = color.rgb2hsv(palette_np / 255.0)

                # Ustawienie domyślnych wag
                weights = np.full(
                    (pixels_hsv.shape[0], 3), [config.get("hue_weight", 3.0), 1.0, 1.0]
                )

                distances_sq = self._calculate_hsv_distance_sq(
                    pixels_hsv, palette_hsv, weights
                )                # POPRAWIONA LOGIKA "Color Focus"
                self.logger.info(f"COLOR FOCUS DEBUG: use_color_focus = {config.get('use_color_focus', False)}")
                self.logger.info(f"COLOR FOCUS DEBUG: focus_ranges = {config.get('focus_ranges', [])}")
                if config.get("use_color_focus", False) and config.get("focus_ranges"):
                    self.logger.info(
                        f"Using Color Focus with {len(config['focus_ranges'])} range(s)."
                    )

                    # Dla każdego focus range
                    for i, focus in enumerate(config["focus_ranges"]):
                        target_h = focus["target_hsv"][0] / 360.0
                        target_s = focus["target_hsv"][1] / 100.0
                        target_v = focus["target_hsv"][2] / 100.0

                        range_h = focus["range_h"] / 360.0
                        range_s = focus["range_s"] / 100.0
                        range_v = focus["range_v"] / 100.0

                        # Sprawdź które KOLORY Z PALETY pasują do focus range
                        palette_h_dist = np.abs(palette_hsv[:, 0] - target_h)
                        palette_hue_mask = np.minimum(
                            palette_h_dist, 1.0 - palette_h_dist
                        ) <= (range_h / 2.0)
                        palette_sat_mask = np.abs(palette_hsv[:, 1] - target_s) <= (
                            range_s / 2.0
                        )
                        palette_val_mask = np.abs(palette_hsv[:, 2] - target_v) <= (
                            range_v / 2.0
                        )

                        palette_final_mask = (
                            palette_hue_mask & palette_sat_mask & palette_val_mask
                        )

                        if np.sum(palette_final_mask) > 0:
                            # APLIKUJ COLOR FOCUS: zmniejsz odległości do preferowanych kolorów palety
                            boost = focus.get("boost_factor", 1.0)
                            distances_sq[:, palette_final_mask] /= boost
                            self.logger.info(
                                f"Color Focus applied: boosted {np.sum(palette_final_mask)} palette colors by factor {boost}"
                            )
                        else:
                            self.logger.warning(
                                f"Color Focus range {i+1}: no palette colors matched the specified range"
                            )

                closest_indices = np.argmin(distances_sq, axis=1)

            elif metric == "lab" and SCIPY_AVAILABLE:
                palette_lab = color.rgb2lab(palette_np / 255.0)
                kdtree = KDTree(palette_lab)
                pixels_lab = color.rgb2lab(pixels_flat / 255.0)
                _, closest_indices = kdtree.query(pixels_lab)
            else:
                if metric == "lab":
                    self.logger.warning(
                        "LAB metric used without Scipy. Falling back to slow calculation."
                    )
                weights = (
                    np.array([0.2126, 0.7152, 0.0722])
                    if metric == "weighted_rgb"
                    else np.array([1.0, 1.0, 1.0])
                )
                distances = np.sqrt(
                    np.sum(
                        (
                            (
                                pixels_flat[:, np.newaxis, :]
                                - palette_np[np.newaxis, :, :]
                            )
                            * weights
                        )
                        ** 2,
                        axis=2,
                    )
                )
                closest_indices = np.argmin(distances, axis=1)

            return palette_np[closest_indices].reshape(image_array.shape)

    def _apply_ordered_dithering(
        self, image_array: np.ndarray, strength: float
    ) -> np.ndarray:
        with self.profiler.profile_operation(
            "apply_ordered_dithering", algorithm_id=self.algorithm_id
        ):
            self.logger.info(
                f"Applying fast ordered dithering with strength {strength}."
            )
            h, w, _ = image_array.shape
            bayer_norm = self.bayer_matrix_8x8 / 64.0 - 0.5
            tiled_bayer = np.tile(bayer_norm, (h // 8 + 1, w // 8 + 1))[:h, :w]
            dither_pattern = tiled_bayer[:, :, np.newaxis] * strength
            dithered_image = np.clip(
                image_array.astype(np.float32) + dither_pattern, 0, 255
            )
            return dithered_image

    def _apply_edge_blending(
        self, mapped_image: Image.Image, config: Dict[str, Any]
    ) -> Image.Image:
        # ... (bez zmian)
        with self.profiler.profile_operation(
            "apply_edge_blending", algorithm_id=self.algorithm_id
        ):
            if not SCIPY_AVAILABLE:
                self.logger.warning(
                    "Scipy not installed. Falling back to simple Gaussian blur for edge blending."
                )
                return mapped_image.filter(
                    ImageFilter.GaussianBlur(radius=config["edge_blur_radius"])
                )

            self.logger.info("Applying advanced edge blending.")
            mapped_array = np.array(mapped_image, dtype=np.float64)
            gray = np.dot(mapped_array[..., :3], [0.2989, 0.5870, 0.1140])
            grad_x = ndimage.sobel(gray, axis=1)
            grad_y = ndimage.sobel(gray, axis=0)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            edge_mask = magnitude > config["edge_detection_threshold"]

            radius = int(config["edge_blur_radius"])
            if radius > 0:
                edge_mask = ndimage.binary_dilation(edge_mask, iterations=radius)

            blurred_array = mapped_array.copy()
            for channel in range(3):
                blurred_array[:, :, channel] = ndimage.gaussian_filter(
                    mapped_array[:, :, channel], sigma=config["edge_blur_radius"]
                )

            blend_factor = (edge_mask * config["edge_blur_strength"])[:, :, np.newaxis]
            result_array = (
                mapped_array * (1 - blend_factor) + blurred_array * blend_factor
            )

            return Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))

    def _preserve_extremes(
        self, mapped_array: np.ndarray, original_array: np.ndarray, threshold: int
    ) -> np.ndarray:
        with self.profiler.profile_operation(
            "preserve_extremes", algorithm_id=self.algorithm_id
        ):
            self.logger.info("Preserving extreme light and shadow areas.")
            luminance = np.dot(original_array[..., :3], [0.2989, 0.5870, 0.1140])
            black_mask = luminance <= threshold
            white_mask = luminance >= (255 - threshold)
            mapped_array[black_mask] = [0, 0, 0]
            mapped_array[white_mask] = [255, 255, 255]
            return mapped_array

    def process_images(
        self, master_path: str, target_path: str, output_path: str, **kwargs
    ) -> bool:
        with self.profiler.profile_operation(
            "process_images_full", algorithm_id=self.algorithm_id
        ):
            run_config = self.default_config_values.copy()
            run_config.update(kwargs)

            self.logger.info(f"Processing with effective config: {run_config}")

            try:
                target_image = Image.open(target_path).convert("RGB")
                target_array = np.array(target_image)
                self.logger.info(f"Target: {target_image.size}")

                palette = self.extract_palette(
                    master_path,
                    num_colors=run_config["num_colors"],
                    method=run_config["palette_method"],
                    quality=run_config["quality"],
                    inject_extremes=run_config["inject_extremes"],
                )

                array_to_map = target_array
                if run_config["dithering_method"] == "ordered":
                    array_to_map = self._apply_ordered_dithering(
                        target_array, run_config["dithering_strength"]
                    )

                mapped_array = self._map_pixels_to_palette(
                    array_to_map, palette, run_config
                )

                if run_config["preserve_extremes"]:
                    mapped_array = self._preserve_extremes(
                        mapped_array, target_array, run_config["extremes_threshold"]
                    )

                mapped_image = Image.fromarray(
                    np.clip(mapped_array, 0, 255).astype(np.uint8), "RGB"
                )

                if run_config["edge_blur_enabled"]:
                    mapped_image = self._apply_edge_blending(mapped_image, run_config)

                if run_config["postprocess_median_filter"] and SCIPY_AVAILABLE:
                    self.logger.info(
                        "Applying post-process median filter to reduce noise."
                    )
                    filtered_array = ndimage.median_filter(
                        np.array(mapped_image), size=3
                    )
                    mapped_image = Image.fromarray(filtered_array)

                mapped_image.save(output_path)
                self.logger.success(
                    f"Successfully processed and saved image to {output_path}"
                )
                return True

            except FileNotFoundError as e:
                self.logger.error(f"File not found: {e}", exc_info=True)
                return False
            except Exception as e:
                self.logger.error(
                    f"An unexpected error occurred during image processing: {e}",
                    exc_info=True,
                )
                return False


def create_palette_mapping_algorithm():
    return PaletteMappingAlgorithm()
