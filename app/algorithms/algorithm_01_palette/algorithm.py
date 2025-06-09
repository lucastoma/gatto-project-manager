import numpy as np
from PIL import Image, ImageFilter, PngImagePlugin
import time
import os
from tqdm import tqdm
import json
from skimage import color # For LAB color space conversion
from skimage import exposure # For dithering (match_histograms)
from typing import TYPE_CHECKING, Any
try:
    from app.core.development_logger import get_logger
    from app.core.performance_profiler import get_profiler
    if TYPE_CHECKING:
        from app.core.development_logger import DevelopmentLogger
        from app.core.performance_profiler import PerformanceProfiler
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
        self.version = "1.2"
        self.config = self.load_config(config_path) if config_path else self.default_config()
        self.distance_cache = {}
        
    def default_config(self):
        """Returns the default configuration dictionary."""
        return {
            'num_colors': 16,
            'distance_metric': 'weighted_rgb', # 'rgb', 'weighted_rgb', 'lab'
            'use_cache': True,
            'preprocess': False,
            'thumbnail_size': (100, 100),
            'use_vectorized': True,
            'cache_max_size': 10000,
            'use_dithering': False,
            'preserve_luminance': False,
            'exclude_colors': [], # List of RGB tuples to exclude
            'preview_mode': False,
            'preview_thumbnail_size': (500, 500) # Max size for preview
        }
    
    def load_config(self, config_path):
        """Loads configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}, using default.")
            return self.default_config()
    
    def clear_cache(self):
        """Clears the distance cache."""
        self.distance_cache.clear()
        
    def validate_palette(self, palette):
        """Validates the color palette."""
        if not palette or len(palette) == 0:
            raise ValueError("Palette cannot be empty")
        
        for i, color_val in enumerate(palette):
            if len(color_val) != 3:
                raise ValueError(f"Color {i} must have 3 RGB components, has {len(color_val)}")
            if not all(0 <= c <= 255 for c in color_val):
                raise ValueError(f"Color {i} has values outside the 0-255 range: {color_val}")
                
    def extract_palette(self, image_path, num_colors=None):
        """
        Extracts a color palette from the master image using proper quantization.
        """
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
            image.thumbnail(self.config['thumbnail_size'])
            
            quantized = image.quantize(colors=num_colors)
            palette_data = quantized.getpalette()
            
            if not palette_data:
                raise ValueError("Could not extract palette data from image.")

            palette_data = palette_data[:num_colors*3]
            
            palette = [[palette_data[i], palette_data[i+1], palette_data[i+2]] 
                       for i in range(0, len(palette_data), 3)]
            
            if self.config['exclude_colors']:
                initial_len = len(palette)
                excluded_set = set(tuple(c) for c in self.config['exclude_colors'])
                palette = [color_val for color_val in palette if tuple(color_val) not in excluded_set]
                self.logger.info(f"Excluded {initial_len - len(palette)} colors from the palette.")

            self.validate_palette(palette)
            self.logger.info(f"Extracted {len(palette)} colors from image {original_size} -> {image.size}")
            return palette
            
        except Exception as e:
            self.logger.error(f"Error extracting palette from {image_path}: {e}")
            default_palette = [
                [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0],
                [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
                [128, 128, 128], [192, 192, 192], [128, 0, 0], [0, 128, 0],
                [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128]
            ]
            return default_palette[:num_colors if num_colors else 16]
    
    def calculate_rgb_distance(self, color1, color2):
        key = None
        if self.config['use_cache']:
            if len(self.distance_cache) > self.config['cache_max_size']:
                self.clear_cache()
            key = (tuple(color1), tuple(color2))
            if key in self.distance_cache:
                return self.distance_cache[key]
        
        if self.config['distance_metric'] == 'weighted_rgb':
            distance = self.calculate_weighted_rgb_distance(color1, color2)
        elif self.config['distance_metric'] == 'lab':
            distance = self.calculate_lab_distance(color1, color2)
        else:
            dr = float(color1[0]) - float(color2[0])
            dg = float(color1[1]) - float(color2[1])
            db = float(color1[2]) - float(color2[2])
            distance = np.sqrt(dr*dr + dg*dg + db*db)
        
        if self.config['use_cache'] and key is not None:
            self.distance_cache[key] = distance
        return distance
    
    def calculate_weighted_rgb_distance(self, color1, color2):
        r_weight, g_weight, b_weight = 0.2126, 0.7152, 0.0722
        dr = (float(color1[0]) - float(color2[0])) * r_weight
        dg = (float(color1[1]) - float(color2[1])) * g_weight
        db = (float(color1[2]) - float(color2[2])) * b_weight
        return np.sqrt(dr*dr + dg*dg + db*db)

    def calculate_lab_distance(self, color1, color2):
        lab1 = color.rgb2lab(np.array([[color1]], dtype=np.uint8) / 255.0)[0][0]
        lab2 = color.rgb2lab(np.array([[color2]], dtype=np.uint8) / 255.0)[0][0]
        return np.sqrt(np.sum((lab1 - lab2)**2))
    
    def find_closest_color(self, target_color, master_palette):
        min_distance = float('inf')
        best_color = master_palette[0]
        for color_val in master_palette:
            distance = self.calculate_rgb_distance(target_color, color_val)
            if distance < min_distance:
                min_distance = distance
                best_color = color_val
        return best_color
    
    def apply_mapping(self, target_image_path, master_palette):
        start_time = time.time()
        try:
            target_image = Image.open(target_image_path)
            if target_image.mode == 'RGBA':
                background = Image.new('RGB', target_image.size, (255, 255, 255))
                background.paste(target_image, mask=target_image.split()[-1])
                target_image = background
            elif target_image.mode != 'RGB':
                target_image = target_image.convert('RGB')
            
            if self.config['preprocess']:
                target_image = target_image.filter(ImageFilter.SMOOTH_MORE)
                self.logger.info("Applied smoothing to the target image.")
            
            if self.config['use_cache']:
                self.clear_cache()
            
            if self.config['use_vectorized']:
                return self.apply_mapping_vectorized(target_image, master_palette, start_time)
            else:
                return self.apply_mapping_naive(target_image, master_palette, start_time)
        except Exception as e:
            self.logger.error(f"Error during image mapping for {target_image_path}: {e}")
            return None
    
    def apply_mapping_vectorized(self, target_image, master_palette, start_time):
        target_array = np.array(target_image)
        pixels = target_array.reshape(-1, 3).astype(np.float64)
        palette_array = np.array(master_palette).astype(np.float64)
        
        self.logger.info(f"Calculating distances vectorized for {len(pixels)} pixels and {len(master_palette)} palette colors...")
        
        if self.config['distance_metric'] == 'weighted_rgb':
            weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
            distances = np.sqrt(np.sum(((pixels[:, np.newaxis] - palette_array) * weights)**2, axis=2))
        else:
            distances = np.sqrt(np.sum((pixels[:, np.newaxis] - palette_array)**2, axis=2))
        
        closest_indices = np.argmin(distances, axis=1)
        result_pixels = palette_array[closest_indices]
        
        if self.config['preserve_luminance']:
            original_lab = color.rgb2lab(target_array / 255.0)
            result_lab = color.rgb2lab(result_pixels.reshape(target_array.shape).astype(np.uint8) / 255.0)
            result_lab[:, :, 0] = original_lab[:, :, 0]
            result_pixels = (color.lab2rgb(result_lab) * 255).astype(np.uint8).reshape(-1, 3)

        if self.config['use_dithering'] and not self.config['preserve_luminance']:
            dithered_image = Image.fromarray(result_pixels.reshape(target_array.shape).astype(np.uint8))
            dithered_image = dithered_image.convert("L").convert("RGB")
            result_pixels = np.array(dithered_image).reshape(-1, 3)

        result_array = result_pixels.reshape(target_array.shape)
        result_image = Image.fromarray(result_array.astype(np.uint8))
        
        processing_time = time.time() - start_time
        self.logger.info(f"Vectorized processing finished in {processing_time:.2f} seconds")
        return result_image
    
    def apply_mapping_naive(self, target_image, master_palette, start_time):
        width, height = target_image.size
        target_array = np.array(target_image)
        result_array = np.zeros_like(target_array)
        
        original_lab_array = None
        if self.config['preserve_luminance']:
            original_lab_array = color.rgb2lab(target_array / 255.0)
        
        self.logger.info(f"Naive mapping for image {width}x{height}...")
        
        for y in tqdm(range(height), desc="Mapping colors", unit="row"):
            for x in range(width):
                target_color = target_array[y, x]
                mapped_color = self.find_closest_color(target_color, master_palette)
                
                if self.config['preserve_luminance'] and original_lab_array is not None:
                    mapped_lab = color.rgb2lab(np.array([[mapped_color]], dtype=np.uint8) / 255.0)[0][0]
                    mapped_lab[0] = original_lab_array[y, x, 0]
                    mapped_color = (color.lab2rgb(np.array([mapped_lab])) * 255).astype(np.uint8)[0]

                result_array[y, x] = mapped_color
        
        if self.config['use_dithering'] and not self.config['preserve_luminance']:
            dithered_image = Image.fromarray(result_array.astype(np.uint8))
            dithered_image = dithered_image.convert("L").convert("RGB")
            result_array = np.array(dithered_image)

        result_image = Image.fromarray(result_array.astype(np.uint8))
        
        processing_time = time.time() - start_time
        self.logger.info(f"Naive processing finished in {processing_time:.2f} seconds")
        return result_image
    
    def process_images(self, master_path, target_path, output_path, **kwargs):
        current_config = self.config.copy()
        for key, value in kwargs.items():
            if key in current_config:
                current_config[key] = value
        
        self.logger.info(f"Starting {self.name} v{self.version}")
        self.logger.info(f"Master (palette): {os.path.basename(master_path)}")
        self.logger.info(f"Target (destination): {os.path.basename(target_path)}")
        
        original_target_path = target_path
        original_master_path = master_path
        
        temp_master_path, temp_target_path = None, None
        
        try:
            if current_config['preview_mode']:
                self.logger.info("Preview mode: scaling images to a smaller resolution.")
                
                temp_dir = os.path.dirname(output_path)
                temp_master_path = os.path.join(temp_dir, f"temp_master_{os.path.basename(master_path)}")
                temp_target_path = os.path.join(temp_dir, f"temp_target_{os.path.basename(target_path)}")

                with Image.open(master_path) as master_image_pil:
                    master_image_pil.thumbnail(current_config['preview_thumbnail_size'])
                    master_image_pil.save(temp_master_path)
                    master_path = temp_master_path
                    self.logger.info(f"Resized master for preview: {master_image_pil.size}")

                with Image.open(target_path) as target_image_pil:
                    target_image_pil.thumbnail(current_config['preview_thumbnail_size'])
                    target_image_pil.save(temp_target_path)
                    target_path = temp_target_path
                    self.logger.info(f"Resized target for preview: {target_image_pil.size}")

            self.logger.info("Extracting color palette from MASTER image...")
            master_palette = self.extract_palette(master_path) 
            self.logger.info(f"Extracted {len(master_palette)} colors from the master palette")
            
            self.logger.info("Sample colors from master palette:")
            for i, color_val in enumerate(master_palette[:5]):
                self.logger.info(f"   Color {i+1}: RGB({color_val[0]}, {color_val[1]}, {color_val[2]})")
            if len(master_palette) > 5:
                self.logger.info(f"   ... and {len(master_palette)-5} more")
            
            self.logger.info("Applying color mapping to TARGET image...")
            result = self.apply_mapping(target_path, master_palette)
            
            if result:
                try:
                    # BEZPOŚREDNI ZAPIS JAKO TIFF BEZ KOMPRESJI
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
            if temp_master_path and os.path.exists(temp_master_path):
                os.remove(temp_master_path)
            if temp_target_path and os.path.exists(temp_target_path):
                os.remove(temp_target_path)
    
    def analyze_mapping_quality(self, original_path, mapped_image):
        try:
            original = Image.open(original_path).convert('RGB')
            if not isinstance(mapped_image, Image.Image):
                 raise TypeError("mapped_image must be a PIL Image object")

            original_array = np.array(original)
            mapped_array = np.array(mapped_image.convert('RGB'))

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