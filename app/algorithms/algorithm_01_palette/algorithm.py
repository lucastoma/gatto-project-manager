import numpy as np
from PIL import Image, ImageFilter, PngImagePlugin
import time
import os
from tqdm import tqdm
import json
from skimage import color # For LAB color space conversion
from sklearn.cluster import KMeans # For K-means clustering
from typing import TYPE_CHECKING, Any

try:
    import scipy.ndimage
except ImportError:
    scipy = None

try:
    from ...core.development_logger import get_logger
    from ...core.performance_profiler import get_profiler
    if TYPE_CHECKING:
        from ...core.development_logger import DevelopmentLogger
        from ...core.performance_profiler import PerformanceProfiler
except ImportError:
    import logging
    def get_logger() -> Any:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    class DummyProfiler:
        def start(self, name): pass
        def stop(self, name): pass
        def get_report(self): return "Profiler not available."
    def get_profiler() -> Any:
        return DummyProfiler()

class PaletteMappingAlgorithm:
    def __init__(self, config_path=None, algorithm_id: str = "algorithm_01_palette"):
        self.algorithm_id = algorithm_id
        if TYPE_CHECKING:
            self.logger: 'DevelopmentLogger' = get_logger()
            self.profiler: 'PerformanceProfiler' = get_profiler()
        else:
            self.logger = get_logger()
            self.profiler = get_profiler()
        self.logger.info(f"Initialized algorithm: {self.algorithm_id}")
        self.name = "Simple Palette Mapping"
        ## >> NEW: Zwiększamy wersję po dodaniu nowych funkcji
        self.version = "1.3"
        self.config = self.load_config(config_path) if config_path else self.default_config()
        self.distance_cache = {}
        
    def default_config(self):
        """Zwraca domyślną konfigurację z nowymi opcjami."""
        return {
            'num_colors': 16,
            'distance_metric': 'weighted_rgb',
            'use_cache': True,
            'preprocess': False,
            'thumbnail_size': (100, 100),
            'use_vectorized': True,
            'cache_max_size': 10000,
            'exclude_colors': [],
            'preview_mode': False,
            'preview_thumbnail_size': (500, 500),
              ## >> NEW: Nowe parametry zaawansowane
            'inject_extremes': False,           # Czy dodawać czarny i biały do palety
            'preserve_extremes': False,         # Czy chronić cienie i światła w obrazie docelowym
            'extremes_threshold': 10,           # Próg dla cieni i świateł (0-255)
            'dithering_method': 'none',         # Metoda ditheringu: 'none' lub 'floyd_steinberg'
            
            ## >> NEW: Edge Blending Parameters
            'edge_blur_enabled': False,         # Włącz/wyłącz rozmycie krawędzi
            'edge_blur_radius': 1.5,            # Promień rozmycia (px)
            'edge_blur_strength': 0.3,          # Siła rozmycia (0.0-1.0)
            'edge_detection_threshold': 25,     # Próg detekcji krawędzi między kolorami
            'edge_blur_method': 'gaussian'      # 'gaussian' | 'motion' | 'selective'
        }
    
    def load_config(self, config_path):
        """Ładuje konfigurację z pliku JSON."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}, using default.")
            return self.default_config()
    
    def clear_cache(self):
        self.distance_cache.clear()
        
    def validate_palette(self, palette):
        if not palette or len(palette) == 0:
            raise ValueError("Palette cannot be empty")
        for i, color_val in enumerate(palette):
            if len(color_val) != 3:
                raise ValueError(f"Color {i} must have 3 RGB components, has {len(color_val)}")
            if not all(0 <= c <= 255 for c in color_val):
                raise ValueError(f"Color {i} has values outside the 0-255 range: {color_val}")
                
    def extract_palette(self, image_path, num_colors=None):
        if num_colors is None:
            num_colors = self.config['num_colors']
        try:
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            original_size = image.size
            image.thumbnail(self.config['thumbnail_size']) # Still use thumbnail for performance
            
            # Convert image to numpy array for K-means
            img_array = np.array(image.convert('RGB'))
            
            # Reshape the image to be a list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Apply K-means clustering to find dominant colors
            # Ensure n_init is set to 'auto' or an integer for KMeans
            kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init='auto') 
            kmeans.fit(pixels)
            
            # Get the cluster centers (the dominant colors)
            palette = kmeans.cluster_centers_.astype(int).tolist()
            
            # Ensure colors are within 0-255 range after conversion
            palette = [[max(0, min(255, c)) for c in color_val] for color_val in palette]
            
            if self.config['exclude_colors']:
                excluded_set = set(tuple(c) for c in self.config['exclude_colors'])
                palette = [color for color in palette if tuple(color) not in excluded_set]

            ## >> NEW: Logika wstrzykiwania ekstremów
            if self.config.get('inject_extremes', False):
                self.logger.info("Injecting pure black and white into the palette.")
                pure_black, pure_white = [0, 0, 0], [255, 255, 255]
                # Sprawdź czy już istnieją, aby uniknąć duplikatów
                has_black = any(c == pure_black for c in palette)
                has_white = any(c == pure_white for c in palette)
                if not has_black:
                    palette.insert(0, pure_black)
                if not has_white:
                    palette.insert(0, pure_white)

            self.validate_palette(palette)
            self.logger.info(f"Extracted {len(palette)} colors from image {original_size} -> {image.size}")
            return palette
        except Exception as e:
            self.logger.error(f"Error extracting palette from {image_path}: {e}")
            return [[0,0,0], [255,255,255], [128,128,128]]

    def calculate_rgb_distance(self, c1, c2):
        key = None
        if self.config['use_cache']:
            key = (tuple(c1), tuple(c2))
            if key in self.distance_cache: return self.distance_cache[key]
        if self.config['distance_metric'] == 'lab':
            dist = self.calculate_lab_distance(c1, c2)
        else: # 'rgb' or 'weighted_rgb'
            dr, dg, db = float(c1[0]) - float(c2[0]), float(c1[1]) - float(c2[1]), float(c1[2]) - float(c2[2])
            if self.config['distance_metric'] == 'weighted_rgb':
                dist = np.sqrt((dr*0.2126)**2 + (dg*0.7152)**2 + (db*0.0722)**2)
            else:
                dist = np.sqrt(dr*dr + dg*dg + db*db)
        if self.config['use_cache'] and key is not None:
            self.distance_cache[key] = dist
        return dist
    def calculate_lab_distance(self, c1, c2):
        lab1 = color.rgb2lab(np.array([[c1]], dtype=np.uint8) / 255.0)[0][0]
        lab2 = color.rgb2lab(np.array([[c2]], dtype=np.uint8) / 255.0)[0][0]
        return np.sqrt(np.sum((lab1 - lab2)**2))
    def find_closest_color(self, target_color, master_palette):
        return min(master_palette, key=lambda color: self.calculate_rgb_distance(target_color, color))

    def apply_mapping(self, target_image_path, master_palette):
        start_time = time.time()
        try:
            target_image = Image.open(target_image_path)
            if target_image.mode != 'RGB':
                target_image = target_image.convert('RGB')
            if self.config['preprocess']:
                target_image = target_image.filter(ImageFilter.SMOOTH_MORE)
            if self.config['use_cache']: self.clear_cache()

            ## >> NEW: Wybór metody mapowania (Dithering vs Wektoryzacja)
            dithering_method = self.config.get('dithering_method', 'none')
            if dithering_method == 'floyd_steinberg':
                self.logger.info("Applying mapping with Floyd-Steinberg dithering (slower, high quality).")
                result_image = self.apply_mapping_dithered(target_image, master_palette, start_time)
            elif self.config['use_vectorized']:
                self.logger.info("Applying mapping with Numpy vectorization (fast).")
                result_image = self.apply_mapping_vectorized(target_image, master_palette, start_time)
            else:
                self.logger.info("Applying mapping with naive pixel-by-pixel method (slow).")
                result_image = self.apply_mapping_naive(target_image, master_palette, start_time)
            
            # Apply preservation of extremes after mapping
            result_array = np.array(result_image)
            result_array = self._apply_extremes_preservation(result_array, target_image)
            result_image = Image.fromarray(result_array.astype(np.uint8))

            ## >> NEW: Rozmycie krawędzi po mapowaniu
            result_image = self.apply_edge_blending(result_image, target_image)

            return result_image
        except Exception as e:
            self.logger.error(f"Error during image mapping for {target_image_path}: {e}")
            return None
    
    ## >> NEW: Nowa funkcja do obsługi ditheringu
    def apply_mapping_dithered(self, target_image, master_palette, start_time):
        img_array = np.array(target_image, dtype=np.float64)
        height, width, _ = img_array.shape

        for y in tqdm(range(height), desc="Dithering", unit="row"):
            for x in range(width):
                old_pixel = img_array[y, x].copy()
                new_pixel = np.array(self.find_closest_color(old_pixel, master_palette))
                img_array[y, x] = new_pixel
                
                quant_error = old_pixel - new_pixel
                
                # Rozpraszanie błędu na sąsiednie piksele
                if x + 1 < width:
                    img_array[y, x + 1] += quant_error * 7 / 16
                if y + 1 < height:
                    if x > 0:
                        img_array[y + 1, x - 1] += quant_error * 3 / 16
                    img_array[y + 1, x] += quant_error * 5 / 16
                    if x + 1 < width:
                        img_array[y + 1, x + 1] += quant_error * 1 / 16
        
        # Przytnij wartości do prawidłowego zakresu i konwertuj na obraz
        result_array = np.clip(img_array, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(result_array)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Dithered processing finished in {processing_time:.2f} seconds")
        return result_image

    def apply_mapping_vectorized(self, target_image, master_palette, start_time):
        target_array = np.array(target_image)
        pixels = target_array.reshape(-1, 3).astype(np.float64)
        palette_array = np.array(master_palette).astype(np.float64)
        
        self.logger.info(f"Calculating distances vectorized for {len(pixels)} pixels and {len(palette_array)} palette colors...")
        
        distances = np.sqrt(np.sum((pixels[:, np.newaxis] - palette_array)**2, axis=2))
        closest_indices = np.argmin(distances, axis=1)
        result_pixels = palette_array[closest_indices]
        
        result_array = result_pixels.reshape(target_array.shape)

        result_image = Image.fromarray(result_array.astype(np.uint8))
        
        processing_time = time.time() - start_time
        self.logger.info(f"Vectorized processing finished in {processing_time:.2f} seconds")
        return result_image
    
    def apply_mapping_naive(self, target_image, master_palette, start_time):
        width, height = target_image.size; target_array = np.array(target_image); result_array = np.zeros_like(target_array)
        self.logger.info(f"Naive mapping for image {width}x{height}...")
        for y in tqdm(range(height), desc="Mapping colors", unit="row"):
            for x in range(width):
                result_array[y, x] = self.find_closest_color(target_array[y, x], master_palette)
        result_image = Image.fromarray(result_array.astype(np.uint8))
        self.logger.info(f"Naive processing finished in {time.time() - start_time:.2f} seconds")
        return result_image

    ## >> NEW: Logika ochrony cieni i świateł - przeniesiona do apply_mapping
    def _apply_extremes_preservation(self, result_array, original_target_image):
        if self.config.get('preserve_extremes', False):
            self.logger.info("Preserving extreme light and shadow areas.")
            threshold = self.config.get('extremes_threshold', 10)
            # Użyj prostej luminancji do znalezienia masek
            original_target_array = np.array(original_target_image)
            luminance = np.dot(original_target_array[...,:3], [0.2989, 0.5870, 0.1140])
            black_mask = luminance <= threshold
            white_mask = luminance >= (255 - threshold)
            
            # Zastosuj maski, aby przywrócić oryginalne piksele (lub ustawić czysty czarny/biały)
            result_array[black_mask] = [0, 0, 0]
            result_array[white_mask] = [255, 255, 255]
        return result_array
    
    ## >> NEW: Edge Blending Methods
    def apply_edge_blending(self, result_image, original_target_image):
        """Rozmycie krawędzi między obszarami palety kolorów"""
        if not self.config.get('edge_blur_enabled', False):
            return result_image
            
        self.logger.info("Applying edge blending to palette boundaries...")
        
        # Convert to numpy arrays for processing
        result_array = np.array(result_image, dtype=np.float64)
        original_array = np.array(original_target_image, dtype=np.float64)
        
        # 1. Detect edges between different palette colors
        edge_mask = self._detect_palette_edges(result_array)
        
        # 2. Apply selective blur based on configuration
        blurred_result = self._apply_selective_blur(result_array, edge_mask, original_array)
        
        # Convert back to PIL Image
        return Image.fromarray(np.clip(blurred_result, 0, 255).astype(np.uint8))
    
    def _detect_palette_edges(self, image_array):
        """Wykrywa krawędzie między obszarami różnych kolorów palety"""
        from scipy import ndimage
        
        # Convert to grayscale for edge detection
        gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Detect edges using gradient
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold to create edge mask
        threshold = self.config.get('edge_detection_threshold', 25)
        edge_mask = magnitude > threshold
        
        # Dilate the mask to include surrounding pixels
        radius = int(self.config.get('edge_blur_radius', 1.5))
        if radius > 0:
            from scipy.ndimage import binary_dilation
            edge_mask = binary_dilation(edge_mask, iterations=radius)
        
        return edge_mask
    
    def _apply_selective_blur(self, image_array, edge_mask, original_array):
        """Zastosuj rozmycie tylko w obszarach określonych przez maskę"""
        blur_method = self.config.get('edge_blur_method', 'gaussian')
        blur_radius = self.config.get('edge_blur_radius', 1.5)
        blur_strength = self.config.get('edge_blur_strength', 0.3)
        
        # Create blurred version
        if blur_method == 'gaussian':
            from scipy.ndimage import gaussian_filter
            blurred = np.zeros_like(image_array)
            for channel in range(3):
                blurred[:,:,channel] = gaussian_filter(image_array[:,:,channel], sigma=blur_radius)
        else:
            # Default to simple averaging for unsupported methods
            from scipy.ndimage import uniform_filter
            blurred = np.zeros_like(image_array)
            for channel in range(3):
                blurred[:,:,channel] = uniform_filter(image_array[:,:,channel], size=int(blur_radius*2+1))
        
        # Blend original and blurred based on edge mask and strength
        result = image_array.copy()
        
        # Apply blending only where edges are detected
        for channel in range(3):
            blend_factor = edge_mask * blur_strength
            result[:,:,channel] = (
                image_array[:,:,channel] * (1 - blend_factor) + 
                blurred[:,:,channel] * blend_factor
            )
        
        return result

    def process_images(self, master_path, target_path, output_path, **kwargs):
        current_config = self.config.copy()
        for key, value in kwargs.items():
            if key in current_config:
                ## >> NEW: Konwersja stringów 'true'/'false' na boolean dla parametrów z JSX
                if isinstance(value, str) and value.lower() in ['true', 'false']:
                    current_config[key] = value.lower() == 'true'
                else:
                    current_config[key] = value
        
        # Przypisz zaktualizowaną konfigurację do instancji na czas tego uruchomienia
        self.config = current_config
        
        self.logger.info(f"Starting {self.name} v{self.version}")
        self.logger.info(f"Master (palette): {os.path.basename(master_path)}")
        self.logger.info(f"Target (destination): {os.path.basename(target_path)}")
        
        try:
            self.logger.info("Extracting color palette from MASTER image...")
            master_palette = self.extract_palette(master_path) 
            self.logger.info(f"Extracted {len(master_palette)} colors from the master palette")
            
            self.logger.info("Applying color mapping to TARGET image...")
            result = self.apply_mapping(target_path, master_palette)
            
            if result:
                try:
                    result.save(output_path, compression='none')
                    self.logger.info(f"Result saved: {output_path}")
                    return True
                except Exception as e:
                    self.logger.error(f"Error during saving: {e}")
                    return False
            else:
                self.logger.error("Error during processing")
                return False
        finally:
            # Przywróć domyślną konfigurację po zakończeniu
            self.config = self.default_config()

    def analyze_mapping_quality(self, original_path, mapped_image):
        try:
            original = Image.open(original_path).convert('RGB')
            if not isinstance(mapped_image, Image.Image): raise TypeError("mapped_image must be a PIL Image object")
            original_array = np.array(original); mapped_array = np.array(mapped_image.convert('RGB'))
            stats = {
                'unique_colors_before': len(np.unique(original_array.reshape(-1, 3), axis=0)),
                'unique_colors_after': len(np.unique(mapped_array.reshape(-1, 3), axis=0)),
                'mean_rgb_difference': np.mean(np.abs(original_array.astype(float) - mapped_array.astype(float))),
                'max_rgb_difference': np.max(np.abs(original_array.astype(float) - mapped_array.astype(float)))
            }
            return stats
        except Exception as e:
            self.logger.error(f"Quality analysis error: {e}")
            return None

def create_palette_mapping_algorithm():
    return PaletteMappingAlgorithm()
