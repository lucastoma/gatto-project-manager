# Weighted Histogram Matching - Część 2: Zaawansowana Implementacja

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 4-6 godzin | **Złożoność**: O(n log n)

---

## Zaawansowane Techniki Implementacji

### 1. Optymalizacja Wydajności

#### Batch Processing dla Dużych Obrazów

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from numba import jit, prange

class OptimizedWeightedHistogramMatching(WeightedHistogramMatching):
    def __init__(self, use_numba=True, n_workers=None):
        super().__init__()
        self.use_numba = use_numba
        self.n_workers = n_workers or mp.cpu_count()
        
        # Kompiluj funkcje Numba przy inicjalizacji
        if self.use_numba:
            self._compile_numba_functions()
    
    def _compile_numba_functions(self):
        """
        Kompiluje funkcje Numba dla lepszej wydajności
        """
        # Dummy call to compile
        dummy_array = np.array([1, 2, 3], dtype=np.uint8)
        dummy_weights = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._numba_apply_weights(dummy_array, dummy_array, dummy_weights)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _numba_apply_weights(source_flat, transformed_flat, weights):
        """
        Numba-optimized weight application
        """
        result = np.empty_like(source_flat, dtype=np.float32)
        
        for i in prange(len(source_flat)):
            pixel_val = source_flat[i]
            weight = weights[pixel_val]
            result[i] = (1.0 - weight) * pixel_val + weight * transformed_flat[i]
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _numba_histogram_cdf(image_flat, bins=256):
        """
        Numba-optimized histogram and CDF calculation
        """
        hist = np.zeros(bins, dtype=np.float32)
        
        # Calculate histogram
        for pixel in image_flat:
            hist[pixel] += 1.0
        
        # Normalize
        total_pixels = len(image_flat)
        hist = hist / total_pixels
        
        # Calculate CDF
        cdf = np.zeros(bins, dtype=np.float32)
        cdf[0] = hist[0]
        for i in range(1, bins):
            cdf[i] = cdf[i-1] + hist[i]
        
        return hist, cdf
    
    def process_large_image_tiled(self, source_image, target_image, 
                                 tile_size=512, overlap=64, **weight_config):
        """
        Przetwarza duże obrazy używając techniki tiling
        """
        height, width = source_image.shape[:2]
        result = np.zeros_like(source_image)
        
        # Oblicz statystyki globalne dla całego obrazu
        global_stats = self._calculate_global_stats(source_image, target_image)
        
        # Przetwarzaj kafelki
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Wyznacz granice kafelka
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                
                # Wyciągnij kafelki
                source_tile = source_image[y:y_end, x:x_end]
                target_tile = target_image[y:y_end, x:x_end]
                
                # Przetwórz kafelek
                processed_tile = self._process_tile_with_global_context(
                    source_tile, target_tile, global_stats, **weight_config
                )
                
                # Blend z overlap
                if overlap > 0 and (y > 0 or x > 0):
                    processed_tile = self._blend_tile_overlap(
                        result[y:y_end, x:x_end], processed_tile, 
                        y, x, overlap, tile_size
                    )
                
                result[y:y_end, x:x_end] = processed_tile
        
        return result
    
    def _calculate_global_stats(self, source_image, target_image):
        """
        Oblicza globalne statystyki dla całego obrazu
        """
        global_stats = {}
        
        for channel in range(source_image.shape[2]):
            source_channel = source_image[:, :, channel]
            target_channel = target_image[:, :, channel]
            
            # Globalne histogramy i CDF
            source_stats = self.calculate_histogram_stats(source_channel)
            target_stats = self.calculate_histogram_stats(target_channel)
            
            global_stats[f'channel_{channel}'] = {
                'source': source_stats,
                'target': target_stats
            }
        
        return global_stats
    
    def _process_tile_with_global_context(self, source_tile, target_tile, 
                                        global_stats, **weight_config):
        """
        Przetwarza kafelek używając globalnego kontekstu
        """
        result_tile = source_tile.copy()
        
        for channel in range(source_tile.shape[2]):
            source_channel = source_tile[:, :, channel]
            
            # Użyj globalnych statystyk
            channel_stats = global_stats[f'channel_{channel}']
            source_cdf = channel_stats['source']['cdf']
            target_cdf = channel_stats['target']['cdf']
            
            # Utwórz lookup table
            levels = np.arange(256)
            lookup_table = np.interp(source_cdf, target_cdf, levels)
            
            # Zastosuj transformację
            source_flat = source_channel.flatten()
            transformed_flat = lookup_table[source_flat]
            
            # Zastosuj wagi
            weight_function = self.create_weight_function(**weight_config)
            
            if self.use_numba:
                result_flat = self._numba_apply_weights(
                    source_flat.astype(np.uint8), 
                    transformed_flat.astype(np.uint8),
                    weight_function.astype(np.float32)
                )
            else:
                pixel_weights = weight_function[source_flat]
                result_flat = ((1 - pixel_weights) * source_flat + 
                             pixel_weights * transformed_flat)
            
            result_channel = result_flat.reshape(source_channel.shape)
            result_tile[:, :, channel] = np.clip(result_channel, 0, 255).astype(np.uint8)
        
        return result_tile
    
    def _blend_tile_overlap(self, existing_tile, new_tile, y, x, overlap, tile_size):
        """
        Blenduje kafelki w obszarze overlap
        """
        result = new_tile.copy()
        
        # Vertical overlap
        if y > 0:
            overlap_height = min(overlap, new_tile.shape[0])
            for i in range(overlap_height):
                alpha = i / overlap_height
                result[i] = (1 - alpha) * existing_tile[i] + alpha * new_tile[i]
        
        # Horizontal overlap
        if x > 0:
            overlap_width = min(overlap, new_tile.shape[1])
            for j in range(overlap_width):
                alpha = j / overlap_width
                result[:, j] = (1 - alpha) * existing_tile[:, j] + alpha * new_tile[:, j]
        
        return result
```

### 2. Adaptacyjne Funkcje Wag

#### Automatyczne Dostosowanie Wag

```python
class AdaptiveWeightedHistogramMatching(OptimizedWeightedHistogramMatching):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adaptation_methods = {
            'contrast_based': self._adapt_weights_by_contrast,
            'histogram_based': self._adapt_weights_by_histogram,
            'gradient_based': self._adapt_weights_by_gradient,
            'content_aware': self._adapt_weights_content_aware
        }
    
    def adaptive_weight_matching(self, source_image, target_image, 
                               adaptation_method='contrast_based', **params):
        """
        Wykonuje weighted histogram matching z adaptacyjnymi wagami
        """
        # Wybierz metodę adaptacji
        adapt_func = self.adaptation_methods.get(adaptation_method)
        if not adapt_func:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
        
        # Oblicz adaptacyjne wagi
        adaptive_weights = adapt_func(source_image, target_image, **params)
        
        # Zastosuj weighted histogram matching
        result = self._apply_adaptive_matching(source_image, target_image, adaptive_weights)
        
        return result, adaptive_weights
    
    def _adapt_weights_by_contrast(self, source_image, target_image, **params):
        """
        Adaptuje wagi na podstawie lokalnego kontrastu
        """
        # Parametry
        window_size = params.get('window_size', 15)
        contrast_threshold = params.get('contrast_threshold', 0.1)
        base_weight = params.get('base_weight', 0.5)
        max_weight = params.get('max_weight', 1.0)
        
        # Konwertuj do grayscale dla analizy kontrastu
        source_gray = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
        
        # Oblicz lokalny kontrast (standard deviation w oknie)
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        mean = cv2.filter2D(source_gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((source_gray.astype(np.float32))**2, -1, kernel)
        local_std = np.sqrt(sqr_mean - mean**2)
        
        # Normalizuj kontrast do [0, 1]
        local_std_norm = local_std / (local_std.max() + 1e-8)
        
        # Utwórz mapę wag na podstawie kontrastu
        contrast_weights = np.where(
            local_std_norm > contrast_threshold,
            base_weight + (max_weight - base_weight) * local_std_norm,
            base_weight * 0.5  # Niższe wagi dla obszarów o niskim kontraście
        )
        
        # Rozszerz do 3 kanałów
        adaptive_weights = np.stack([contrast_weights] * 3, axis=2)
        
        return adaptive_weights
    
    def _adapt_weights_by_histogram(self, source_image, target_image, **params):
        """
        Adaptuje wagi na podstawie lokalnych histogramów
        """
        # Parametry
        block_size = params.get('block_size', 64)
        similarity_threshold = params.get('similarity_threshold', 0.7)
        base_weight = params.get('base_weight', 0.5)
        max_weight = params.get('max_weight', 1.0)
        
        height, width = source_image.shape[:2]
        adaptive_weights = np.full((height, width, 3), base_weight, dtype=np.float32)
        
        # Przetwarzaj bloki
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                y_end = min(y + block_size, height)
                x_end = min(x + block_size, width)
                
                # Wyciągnij bloki
                source_block = source_image[y:y_end, x:x_end]
                target_block = target_image[y:y_end, x:x_end]
                
                # Oblicz podobieństwo histogramów
                similarity = self._calculate_histogram_similarity(source_block, target_block)
                
                # Adaptuj wagi - większe różnice = większe wagi
                block_weight = base_weight + (max_weight - base_weight) * (1 - similarity)
                adaptive_weights[y:y_end, x:x_end] = block_weight
        
        return adaptive_weights
    
    def _adapt_weights_by_gradient(self, source_image, target_image, **params):
        """
        Adaptuje wagi na podstawie gradientów obrazu
        """
        # Parametry
        gradient_threshold = params.get('gradient_threshold', 50)
        base_weight = params.get('base_weight', 0.3)
        edge_weight = params.get('edge_weight', 0.8)
        
        # Konwertuj do grayscale
        source_gray = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
        
        # Oblicz gradienty
        grad_x = cv2.Sobel(source_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(source_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Utwórz mapę wag
        edge_mask = gradient_magnitude > gradient_threshold
        gradient_weights = np.where(edge_mask, edge_weight, base_weight)
        
        # Wygładź przejścia
        gradient_weights = cv2.GaussianBlur(gradient_weights, (5, 5), 1.0)
        
        # Rozszerz do 3 kanałów
        adaptive_weights = np.stack([gradient_weights] * 3, axis=2)
        
        return adaptive_weights
    
    def _adapt_weights_content_aware(self, source_image, target_image, **params):
        """
        Adaptuje wagi na podstawie analizy zawartości obrazu
        """
        # Parametry
        skin_detection = params.get('skin_detection', True)
        sky_detection = params.get('sky_detection', True)
        vegetation_detection = params.get('vegetation_detection', True)
        
        height, width = source_image.shape[:2]
        adaptive_weights = np.full((height, width, 3), 0.5, dtype=np.float32)
        
        # Detekcja skóry
        if skin_detection:
            skin_mask = self._detect_skin_regions(source_image)
            adaptive_weights[skin_mask] = [0.8, 0.9, 0.7]  # Różne wagi dla RGB
        
        # Detekcja nieba
        if sky_detection:
            sky_mask = self._detect_sky_regions(source_image)
            adaptive_weights[sky_mask] = [0.6, 0.7, 1.0]  # Większa waga dla niebieskiego
        
        # Detekcja roślinności
        if vegetation_detection:
            vegetation_mask = self._detect_vegetation_regions(source_image)
            adaptive_weights[vegetation_mask] = [0.7, 1.0, 0.6]  # Większa waga dla zielonego
        
        return adaptive_weights
    
    def _detect_skin_regions(self, image):
        """
        Prosta detekcja regionów skóry w przestrzeni HSV
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Zakresy HSV dla skóry
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morfologia dla wygładzenia
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask > 0
    
    def _detect_sky_regions(self, image):
        """
        Prosta detekcja regionów nieba
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Zakresy HSV dla nieba
        lower_sky = np.array([100, 50, 50], dtype=np.uint8)
        upper_sky = np.array([130, 255, 255], dtype=np.uint8)
        
        sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)
        
        # Dodatkowo sprawdź czy region jest w górnej części obrazu
        height = image.shape[0]
        upper_region_mask = np.zeros_like(sky_mask)
        upper_region_mask[:height//2, :] = 255
        
        sky_mask = cv2.bitwise_and(sky_mask, upper_region_mask)
        
        return sky_mask > 0
    
    def _detect_vegetation_regions(self, image):
        """
        Prosta detekcja regionów roślinności
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Zakresy HSV dla roślinności
        lower_vegetation = np.array([40, 40, 40], dtype=np.uint8)
        upper_vegetation = np.array([80, 255, 255], dtype=np.uint8)
        
        vegetation_mask = cv2.inRange(hsv, lower_vegetation, upper_vegetation)
        
        # Morfologia
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
        
        return vegetation_mask > 0
    
    def _calculate_histogram_similarity(self, block1, block2):
        """
        Oblicza podobieństwo histogramów między dwoma blokami
        """
        similarities = []
        
        for channel in range(3):
            hist1 = cv2.calcHist([block1[:,:,channel]], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([block2[:,:,channel]], [0], None, [256], [0, 256])
            
            # Normalizuj histogramy
            hist1 = hist1 / (hist1.sum() + 1e-8)
            hist2 = hist2 / (hist2.sum() + 1e-8)
            
            # Oblicz korelację
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            similarities.append(max(0, correlation))
        
        return np.mean(similarities)
    
    def _apply_adaptive_matching(self, source_image, target_image, adaptive_weights):
        """
        Stosuje weighted histogram matching z adaptacyjnymi wagami
        """
        result = source_image.copy().astype(np.float32)
        
        for channel in range(3):
            source_channel = source_image[:, :, channel]
            target_channel = target_image[:, :, channel]
            
            # Standardowy histogram matching
            transformed = self.standard_histogram_matching(source_channel, target_channel)
            
            # Zastosuj adaptacyjne wagi
            channel_weights = adaptive_weights[:, :, channel]
            
            result_channel = ((1 - channel_weights) * source_channel.astype(np.float32) + 
                            channel_weights * transformed.astype(np.float32))
            
            result[:, :, channel] = result_channel
        
        return np.clip(result, 0, 255).astype(np.uint8)
```

### 3. Lokalne Histogram Matching

#### Implementacja CLAHE-inspired Approach

```python
class LocalWeightedHistogramMatching(AdaptiveWeightedHistogramMatching):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def local_weighted_matching(self, source_image, target_image, 
                              grid_size=(8, 8), clip_limit=2.0, **weight_config):
        """
        Lokalne weighted histogram matching inspirowane CLAHE
        """
        height, width = source_image.shape[:2]
        grid_h, grid_w = grid_size
        
        # Rozmiary bloków
        block_h = height // grid_h
        block_w = width // grid_w
        
        result = np.zeros_like(source_image, dtype=np.float32)
        
        # Utwórz lookup tables dla każdego bloku
        lookup_tables = self._create_local_lookup_tables(
            source_image, target_image, grid_size, clip_limit, **weight_config
        )
        
        # Interpoluj między blokami
        for y in range(height):
            for x in range(width):
                # Znajdź pozycję w siatce
                grid_y = min(y / block_h, grid_h - 1)
                grid_x = min(x / block_w, grid_w - 1)
                
                # Indeksy sąsiadujących bloków
                y0, y1 = int(grid_y), min(int(grid_y) + 1, grid_h - 1)
                x0, x1 = int(grid_x), min(int(grid_x) + 1, grid_w - 1)
                
                # Wagi interpolacji
                wy = grid_y - y0
                wx = grid_x - x0
                
                # Interpolacja bilinearna lookup tables
                for channel in range(3):
                    pixel_val = source_image[y, x, channel]
                    
                    # Pobierz wartości z 4 sąsiadujących lookup tables
                    v00 = lookup_tables[y0, x0, channel][pixel_val]
                    v01 = lookup_tables[y0, x1, channel][pixel_val]
                    v10 = lookup_tables[y1, x0, channel][pixel_val]
                    v11 = lookup_tables[y1, x1, channel][pixel_val]
                    
                    # Interpolacja bilinearna
                    v0 = v00 * (1 - wx) + v01 * wx
                    v1 = v10 * (1 - wx) + v11 * wx
                    interpolated_val = v0 * (1 - wy) + v1 * wy
                    
                    result[y, x, channel] = interpolated_val
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _create_local_lookup_tables(self, source_image, target_image, 
                                  grid_size, clip_limit, **weight_config):
        """
        Tworzy lokalne lookup tables dla każdego bloku w siatce
        """
        height, width = source_image.shape[:2]
        grid_h, grid_w = grid_size
        
        block_h = height // grid_h
        block_w = width // grid_w
        
        # Tablica lookup tables [grid_h, grid_w, channels, 256]
        lookup_tables = np.zeros((grid_h, grid_w, 3, 256), dtype=np.float32)
        
        for gy in range(grid_h):
            for gx in range(grid_w):
                # Granice bloku z rozszerzeniem
                y_start = max(0, gy * block_h - block_h // 4)
                y_end = min(height, (gy + 1) * block_h + block_h // 4)
                x_start = max(0, gx * block_w - block_w // 4)
                x_end = min(width, (gx + 1) * block_w + block_w // 4)
                
                # Wyciągnij rozszerzony blok
                source_block = source_image[y_start:y_end, x_start:x_end]
                target_block = target_image[y_start:y_end, x_start:x_end]
                
                # Utwórz lookup table dla tego bloku
                for channel in range(3):
                    source_channel = source_block[:, :, channel]
                    target_channel = target_block[:, :, channel]
                    
                    # Oblicz histogramy z clipping
                    source_hist_clipped = self._calculate_clipped_histogram(
                        source_channel, clip_limit
                    )
                    target_hist_clipped = self._calculate_clipped_histogram(
                        target_channel, clip_limit
                    )
                    
                    # Oblicz CDF
                    source_cdf = np.cumsum(source_hist_clipped)
                    target_cdf = np.cumsum(target_hist_clipped)
                    
                    # Normalizuj CDF
                    source_cdf = source_cdf / (source_cdf[-1] + 1e-8)
                    target_cdf = target_cdf / (target_cdf[-1] + 1e-8)
                    
                    # Utwórz lookup table
                    levels = np.arange(256)
                    standard_lut = np.interp(source_cdf, target_cdf, levels)
                    
                    # Zastosuj wagi
                    weight_function = self.create_weight_function(**weight_config)
                    weighted_lut = ((1 - weight_function) * levels + 
                                  weight_function * standard_lut)
                    
                    lookup_tables[gy, gx, channel] = weighted_lut
        
        return lookup_tables
    
    def _calculate_clipped_histogram(self, image_channel, clip_limit):
        """
        Oblicza histogram z clipping (podobnie jak w CLAHE)
        """
        hist, _ = np.histogram(image_channel.flatten(), bins=256, range=(0, 255))
        
        # Clip histogram
        total_pixels = len(image_channel.flatten())
        clip_threshold = clip_limit * total_pixels / 256
        
        # Znajdź bins do clippingu
        excess = np.maximum(hist - clip_threshold, 0)
        hist = np.minimum(hist, clip_threshold)
        
        # Redystrybuuj excess równomiernie
        total_excess = np.sum(excess)
        redistribution = total_excess / 256
        hist = hist + redistribution
        
        # Normalizuj
        hist = hist / np.sum(hist)
        
        return hist
```

### 4. Integracja z Maskami i ROI

#### Selektywne Przetwarzanie Regionów

```python
class MaskedWeightedHistogramMatching(LocalWeightedHistogramMatching):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def masked_weighted_matching(self, source_image, target_image, 
                               mask=None, roi=None, **weight_config):
        """
        Weighted histogram matching z maską lub ROI
        """
        result = source_image.copy()
        
        # Utwórz maskę jeśli podano ROI
        if roi is not None and mask is None:
            mask = self._create_roi_mask(source_image.shape[:2], roi)
        
        # Jeśli nie ma maski, przetwórz cały obraz
        if mask is None:
            return self.process_rgb_image(source_image, target_image, **weight_config)
        
        # Przetwórz tylko obszary objęte maską
        for channel in range(3):
            source_channel = source_image[:, :, channel]
            target_channel = target_image[:, :, channel]
            
            # Wyciągnij piksele z maski
            masked_source = source_channel[mask]
            masked_target = target_channel[mask]
            
            if len(masked_source) == 0:
                continue
            
            # Oblicz statystyki dla zamaskowanych obszarów
            source_stats = self._calculate_masked_stats(masked_source)
            target_stats = self._calculate_masked_stats(masked_target)
            
            # Utwórz lookup table
            lookup_table = np.interp(source_stats['cdf'], target_stats['cdf'], 
                                   np.arange(256))
            
            # Zastosuj transformację tylko w masce
            transformed_channel = source_channel.copy().astype(np.float32)
            transformed_values = lookup_table[masked_source]
            
            # Zastosuj wagi
            weight_function = self.create_weight_function(**weight_config)
            masked_weights = weight_function[masked_source]
            
            # Weighted blending
            final_values = ((1 - masked_weights) * masked_source.astype(np.float32) + 
                          masked_weights * transformed_values)
            
            # Wstaw z powrotem do obrazu
            transformed_channel[mask] = final_values
            result[:, :, channel] = np.clip(transformed_channel, 0, 255).astype(np.uint8)
        
        return result
    
    def _create_roi_mask(self, image_shape, roi):
        """
        Tworzy maskę z ROI (x, y, width, height)
        """
        mask = np.zeros(image_shape, dtype=bool)
        x, y, w, h = roi
        mask[y:y+h, x:x+w] = True
        return mask
    
    def _calculate_masked_stats(self, masked_pixels):
        """
        Oblicza statystyki dla zamaskowanych pikseli
        """
        hist, _ = np.histogram(masked_pixels, bins=256, range=(0, 255), density=True)
        cdf = np.cumsum(hist)
        cdf = cdf / (cdf[-1] + 1e-8)
        
        return {
            'histogram': hist,
            'cdf': cdf,
            'mean': np.mean(masked_pixels),
            'std': np.std(masked_pixels)
        }
    
    def multi_region_matching(self, source_image, target_image, regions_config):
        """
        Weighted histogram matching dla wielu regionów z różnymi konfiguracjami
        
        regions_config: lista słowników z kluczami:
        - 'mask' lub 'roi': definicja regionu
        - 'weight_config': konfiguracja wag dla tego regionu
        - 'target_region': opcjonalnie inny region z target_image
        """
        result = source_image.copy()
        
        for region_config in regions_config:
            # Pobierz maskę regionu
            if 'mask' in region_config:
                mask = region_config['mask']
            elif 'roi' in region_config:
                mask = self._create_roi_mask(source_image.shape[:2], region_config['roi'])
            else:
                continue
            
            # Pobierz target region (domyślnie ten sam co source)
            if 'target_region' in region_config:
                target_mask = region_config['target_region']
            else:
                target_mask = mask
            
            # Pobierz konfigurację wag
            weight_config = region_config.get('weight_config', {})
            
            # Przetwórz region
            region_result = self._process_single_region(
                source_image, target_image, mask, target_mask, weight_config
            )
            
            # Wstaw wynik do obrazu wynikowego
            result[mask] = region_result[mask]
        
        return result
    
    def _process_single_region(self, source_image, target_image, 
                             source_mask, target_mask, weight_config):
        """
        Przetwarza pojedynczy region
        """
        result = source_image.copy()
        
        for channel in range(3):
            source_channel = source_image[:, :, channel]
            target_channel = target_image[:, :, channel]
            
            # Wyciągnij piksele z masek
            source_pixels = source_channel[source_mask]
            target_pixels = target_channel[target_mask]
            
            if len(source_pixels) == 0 or len(target_pixels) == 0:
                continue
            
            # Oblicz statystyki
            source_stats = self._calculate_masked_stats(source_pixels)
            target_stats = self._calculate_masked_stats(target_pixels)
            
            # Histogram matching
            lookup_table = np.interp(source_stats['cdf'], target_stats['cdf'], 
                                   np.arange(256))
            
            transformed_pixels = lookup_table[source_pixels]
            
            # Zastosuj wagi
            weight_function = self.create_weight_function(**weight_config)
            pixel_weights = weight_function[source_pixels]
            
            final_pixels = ((1 - pixel_weights) * source_pixels.astype(np.float32) + 
                          pixel_weights * transformed_pixels)
            
            # Wstaw z powrotem
            result_channel = result[:, :, channel].astype(np.float32)
            result_channel[source_mask] = final_pixels
            result[:, :, channel] = np.clip(result_channel, 0, 255).astype(np.uint8)
        
        return result
```

---

## Konfiguracja i Parametry

### System Konfiguracji

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import json

@dataclass
class WeightConfig:
    """Konfiguracja funkcji wag"""
    weight_type: str = 'segmented'
    shadow_weight: float = 0.8
    midtone_weight: float = 1.0
    highlight_weight: float = 0.6
    shadow_threshold: int = 85
    highlight_threshold: int = 170
    transition_width: int = 10
    
    # Parametry dla innych typów wag
    min_weight: float = 0.0
    max_weight: float = 1.0
    direction: str = 'ascending'
    center: int = 128
    sigma: float = 50.0
    amplitude: float = 1.0
    baseline: float = 0.0
    control_points: List[Tuple[int, float]] = field(default_factory=lambda: [(0, 0.5), (128, 1.0), (255, 0.5)])
    interpolation: str = 'linear'

@dataclass
class ProcessingConfig:
    """Konfiguracja przetwarzania"""
    use_numba: bool = True
    n_workers: int = 4
    tile_size: int = 512
    overlap: int = 64
    process_channels: str = 'RGB'
    
@dataclass
class AdaptiveConfig:
    """Konfiguracja adaptacyjnych wag"""
    adaptation_method: str = 'contrast_based'
    window_size: int = 15
    contrast_threshold: float = 0.1
    base_weight: float = 0.5
    max_weight: float = 1.0
    block_size: int = 64
    similarity_threshold: float = 0.7
    gradient_threshold: float = 50
    edge_weight: float = 0.8
    skin_detection: bool = True
    sky_detection: bool = True
    vegetation_detection: bool = True

@dataclass
class LocalConfig:
    """Konfiguracja lokalnego przetwarzania"""
    grid_size: Tuple[int, int] = (8, 8)
    clip_limit: float = 2.0
    use_local_matching: bool = False

class WeightedHistogramConfig:
    """Główna klasa konfiguracji"""
    
    def __init__(self):
        self.weight_config = WeightConfig()
        self.processing_config = ProcessingConfig()
        self.adaptive_config = AdaptiveConfig()
        self.local_config = LocalConfig()
    
    def save_to_file(self, filepath: str):
        """Zapisuje konfigurację do pliku JSON"""
        config_dict = {
            'weight_config': self.weight_config.__dict__,
            'processing_config': self.processing_config.__dict__,
            'adaptive_config': self.adaptive_config.__dict__,
            'local_config': self.local_config.__dict__
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Wczytuje konfigurację z pliku JSON"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        if 'weight_config' in config_dict:
            self.weight_config = WeightConfig(**config_dict['weight_config'])
        if 'processing_config' in config_dict:
            self.processing_config = ProcessingConfig(**config_dict['processing_config'])
        if 'adaptive_config' in config_dict:
            self.adaptive_config = AdaptiveConfig(**config_dict['adaptive_config'])
        if 'local_config' in config_dict:
            self.local_config = LocalConfig(**config_dict['local_config'])
    
    def get_preset(self, preset_name: str):
        """Zwraca predefiniowaną konfigurację"""
        presets = {
            'portrait': self._get_portrait_preset(),
            'landscape': self._get_landscape_preset(),
            'low_light': self._get_low_light_preset(),
            'high_contrast': self._get_high_contrast_preset(),
            'subtle': self._get_subtle_preset()
        }
        
        if preset_name in presets:
            return presets[preset_name]
        else:
            raise ValueError(f"Unknown preset: {preset_name}")
    
    def _get_portrait_preset(self):
        """Preset dla portretów"""
        config = WeightedHistogramConfig()
        config.weight_config.weight_type = 'segmented'
        config.weight_config.shadow_weight = 0.9
        config.weight_config.midtone_weight = 1.0
        config.weight_config.highlight_weight = 0.7
        config.adaptive_config.skin_detection = True
        return config
    
    def _get_landscape_preset(self):
        """Preset dla krajobrazów"""
        config = WeightedHistogramConfig()
        config.weight_config.weight_type = 'segmented'
        config.weight_config.shadow_weight = 0.8
        config.weight_config.midtone_weight = 0.9
        config.weight_config.highlight_weight = 1.0
        config.adaptive_config.sky_detection = True
        config.adaptive_config.vegetation_detection = True
        return config
    
    def _get_low_light_preset(self):
        """Preset dla słabo oświetlonych zdjęć"""
        config = WeightedHistogramConfig()
        config.weight_config.weight_type = 'linear'
        config.weight_config.direction = 'ascending'
        config.weight_config.min_weight = 0.9
        config.weight_config.max_weight = 0.3
        return config
    
    def _get_high_contrast_preset(self):
        """Preset dla wysokiego kontrastu"""
        config = WeightedHistogramConfig()
        config.weight_config.weight_type = 'gaussian'
        config.weight_config.center = 128
        config.weight_config.sigma = 60
        config.weight_config.amplitude = 1.0
        config.weight_config.baseline = 0.2
        return config
    
    def _get_subtle_preset(self):
        """Preset dla subtelnych zmian"""
        config = WeightedHistogramConfig()
        config.weight_config.weight_type = 'segmented'
        config.weight_config.shadow_weight = 0.4
        config.weight_config.midtone_weight = 0.5
        config.weight_config.highlight_weight = 0.3
        return config
```

---

## Podsumowanie Części 2

W tej części omówiliśmy:

1. **Optymalizację wydajności** z Numba i przetwarzaniem kafelkowym
2. **Adaptacyjne funkcje wag** dostosowujące się do zawartości obrazu
3. **Lokalne histogram matching** inspirowane CLAHE
4. **Integrację z maskami i ROI** dla selektywnego przetwarzania
5. **System konfiguracji** z presetami i zapisem do plików

### Co dalej?

**Część 3** będzie zawierać:
- Testy jednostkowe i benchmarki wydajności
- Przykłady praktycznego zastosowania
- Rozwiązywanie problemów i debugowanie
- Integrację z głównym systemem Flask
- Dokumentację API

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Część 2 - Zaawansowana implementacja