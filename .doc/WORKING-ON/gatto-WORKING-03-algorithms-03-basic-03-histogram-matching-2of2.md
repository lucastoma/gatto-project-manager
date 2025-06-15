# Simple Histogram Matching - Proste Dopasowanie Histogramu [2/2]

## quality control

Quality tester A: Problems found and correction applied to code snippets
->
Quality tester B: Problems found and correction applied
Quality tester B: Final review passed 2025-06-08 16:06 CEST


## 🟡 Poziom: Intermediate
**Trudność**: Wysoka | **Czas implementacji**: 4-8 godzin | **Złożoność**: O(n log n)

**📋 Część 2/2**: Zaawansowane funkcje, optymalizacje, batch processing, testy

---

## Zaawansowane Implementacje

### LAB Color Space Matching

```python
class LABHistogramMatching(SimpleHistogramMatching):
    def __init__(self):
        super().__init__()
        self.name = "LAB Histogram Matching"
        
    def lab_histogram_matching(self, source_array, target_array, match_luminance=True, match_color=True):
        """
        Dopasowanie w przestrzeni LAB dla lepszej kontroli
        """
        from skimage import color
        
        # Konwersja do LAB
        source_lab = color.rgb2lab(source_array / 255.0)
        target_lab = color.rgb2lab(target_array / 255.0)
        result_lab = source_lab.copy()
        
        # Dopasuj kanały
        if match_luminance:
            # L channel (0-100)
            L_matched = self._match_channel_float(
                source_lab[:,:,0], target_lab[:,:,0], value_range=(0, 100)
            )
            result_lab[:,:,0] = L_matched
        
        if match_color:
            # a,b channels (-128 to 127)
            for i in [1, 2]:
                matched = self._match_channel_float(
                    source_lab[:,:,i], target_lab[:,:,i], value_range=(-128, 127)
                )
                result_lab[:,:,i] = matched
        
        # Konwersja z powrotem do RGB
        result_rgb = color.lab2rgb(result_lab)
        return np.clip(result_rgb * 255, 0, 255).astype(np.uint8)
    
    def _match_channel_float(self, source_channel, target_channel, value_range, bins=1000):
        """
        Dopasowanie kanału z wartościami float
        """
        # Normalizuj do 0-1
        source_norm = (source_channel - value_range[0]) / (value_range[1] - value_range[0])
        target_norm = (target_channel - value_range[0]) / (value_range[1] - value_range[0])
        
        # Oblicz histogramy
        source_hist, bins_s = np.histogram(source_norm.flatten(), bins=bins, range=(0, 1))
        target_hist, bins_t = np.histogram(target_norm.flatten(), bins=bins, range=(0, 1))
        
        # Oblicz CDF
        source_cdf = np.cumsum(source_hist) / np.sum(source_hist)
        target_cdf = np.cumsum(target_hist) / np.sum(target_hist)
        
        # Utwórz mapowanie
        bin_centers = (bins_s[:-1] + bins_s[1:]) / 2
        
        # Interpolacja dla dokładnego mapowania
        from scipy import interpolate
        
        # Inwersja target CDF
        valid_mask = target_cdf > 0
        if np.sum(valid_mask) > 1:
            inverse_target = interpolate.interp1d(
                target_cdf[valid_mask], bin_centers[valid_mask],
                bounds_error=False, fill_value=(0, 1)
            )
            
            # Mapowanie source przez target inverse
            matched_norm = inverse_target(source_cdf[
                np.digitize(source_norm.flatten(), bins_s) - 1
            ])
            matched_norm = np.clip(matched_norm, 0, 1)
        else:
            matched_norm = source_norm.flatten()
        
        # Przywróć oryginalny zakres
        matched = matched_norm * (value_range[1] - value_range[0]) + value_range[0]
        return matched.reshape(source_channel.shape)
```

### Advanced Histogram Matching with Caching

```python
class CachedHistogramMatcher(SimpleHistogramMatching):
    def __init__(self, cache_size=100):
        super().__init__()
        self._lut_cache = {}
        self._histogram_cache = {}
        self.max_cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _get_image_hash(self, image_array):
        """Szybki hash obrazu"""
        # Używamy hash z sample punktów dla szybkości
        sample = image_array[::10, ::10].flatten()
        return hash(sample.tobytes())
    
    def _get_cache_key(self, source_array, target_array, use_interpolation):
        """Klucz cache'a"""
        return (
            self._get_image_hash(source_array),
            self._get_image_hash(target_array),
            use_interpolation
        )
    
    def apply_histogram_matching(self, source_array, target_array, use_interpolation=True):
        """Cached version of histogram matching"""
        cache_key = self._get_cache_key(source_array, target_array, use_interpolation)
        
        if cache_key in self._lut_cache:
            self.cache_hits += 1
            luts = self._lut_cache[cache_key]
            
            # Zastosuj cached LUTs
            result = np.zeros_like(source_array)
            for ch in range(3):
                result[:,:,ch] = luts[ch][source_array[:,:,ch]]
            return result
        
        self.cache_misses += 1
        
        # Oblicz normalnie
        result = super().apply_histogram_matching(source_array, target_array, use_interpolation)
        
        # Cache LUTs
        if len(self._lut_cache) < self.max_cache_size:
            luts = []
            for ch in range(3):
                source_hist = self.calculate_histogram(source_array, ch)
                target_hist = self.calculate_histogram(target_array, ch)
                source_cdf = self.calculate_cdf(source_hist)
                target_cdf = self.calculate_cdf(target_hist)
                
                if use_interpolation:
                    lut = self.create_lookup_table_interpolated(source_cdf, target_cdf)
                else:
                    lut = self.create_lookup_table(source_cdf, target_cdf)
                
                luts.append(lut)
            
            self._lut_cache[cache_key] = luts
        
        return result
    
    def get_cache_stats(self):
        """Statystyki cache'a"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._lut_cache)
        }
```

### Batch Processing with Progress

```python
class BatchHistogramMatcher(CachedHistogramMatcher):
    def __init__(self):
        super().__init__()
          def batch_process_with_progress(self, file_list, target_path, output_dir, 
                                  use_interpolation=True, max_workers=4):
        """
        Przetwarzanie wsadowe z progress bar i multiprocessing
        UWAGA: W trybie wieloprocesowym cachowanie nie działa - każdy proces ma własną instancję
        """
        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        failed = []
        
        # Przygotuj argumenty dla procesów
        args_list = []
        for i, source_path in enumerate(file_list):
            output_path = os.path.join(output_dir, f"matched_{i:04d}_{os.path.basename(source_path)}")
            args_list.append((source_path, target_path, output_path, use_interpolation))
        
        # Przetwarzanie równoległe (bez cachingu - każdy proces ma własną instancję)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Prześlij zadania
            future_to_args = {
                executor.submit(self._process_single_image, args): args 
                for args in args_list
            }
            
            # Zbieraj wyniki z progress bar
            with tqdm(total=len(file_list), desc="Processing images") as pbar:
                for future in as_completed(future_to_args):
                    args = future_to_args[future]
                    source_path, _, output_path, _ = args
                    
                    try:
                        success = future.result()
                        if success:
                            results.append(output_path)
                        else:
                            failed.append(source_path)
                    except Exception as e:
                        print(f"\nError processing {source_path}: {e}")
                        failed.append(source_path)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': len(results), 
                        'Failed': len(failed)
                        # Uwaga: Cache Hit Rate nie jest dostępny w trybie wieloprocesowym
                    })
        
        return results, failed
    
    @staticmethod
    def _process_single_image(args):
        """Helper function for multiprocessing"""
        source_path, target_path, output_path, use_interpolation = args
        
        try:
            matcher = SimpleHistogramMatching()
            return matcher.process_images(source_path, target_path, output_path, use_interpolation)
        except Exception:
            return False
    
    def batch_analyze_quality(self, original_paths, target_path, result_paths):
        """
        Analiza jakości dla całej serii obrazów
        """
        from collections import defaultdict
        
        quality_stats = defaultdict(list)
        
        target_array = self.extract_target_histogram(target_path)
        
        for orig_path, result_path in zip(original_paths, result_paths):
            try:
                source_array = np.array(Image.open(orig_path).convert('RGB'))
                result_array = np.array(Image.open(result_path).convert('RGB'))
                
                # Oblicz metryki dla każdego kanału
                for ch, ch_name in enumerate(['R', 'G', 'B']):
                    metrics = self._calculate_channel_metrics(
                        source_array[:,:,ch], 
                        target_array[:,:,ch], 
                        result_array[:,:,ch]
                    )
                    
                    for metric_name, value in metrics.items():
                        quality_stats[f"{ch_name}_{metric_name}"].append(value)
                        
            except Exception as e:
                print(f"Error analyzing {orig_path}: {e}")
        
        # Statystyki podsumowujące
        summary = {}
        for key, values in quality_stats.items():
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return summary
      def _calculate_channel_metrics(self, source_ch, target_ch, result_ch):
        """Oblicz metryki dla pojedynczego kanału"""
        # Podstawowe metryki
        metrics = {
            'mse_to_target': np.mean((result_ch.astype(float) - target_ch.astype(float))**2),
            'mse_to_source': np.mean((result_ch.astype(float) - source_ch.astype(float))**2),
            'mean_diff_target': abs(np.mean(result_ch) - np.mean(target_ch)),
            'std_ratio_target': np.std(result_ch) / (np.std(target_ch) + 1e-10),
        }
        
        # Histogram correlation
        hist_source = np.histogram(source_ch, bins=256, range=(0, 255))[0]
        hist_target = np.histogram(target_ch, bins=256, range=(0, 255))[0] 
        hist_result = np.histogram(result_ch, bins=256, range=(0, 255))[0]
        
        # Korelacja histogramów
        if np.std(hist_target) > 1e-10 and np.std(hist_result) > 1e-10:
            metrics['hist_corr_target'] = np.corrcoef(hist_result, hist_target)[0,1]
        else:
            metrics['hist_corr_target'] = 0.0
            
        return metrics
```

### Memory-Efficient Processing

```python
class MemoryEfficientMatcher(SimpleHistogramMatching):
    def __init__(self, tile_size=2048, overlap=128):
        super().__init__()
        self.tile_size = tile_size
        self.overlap = overlap
        
    def process_large_image(self, source_path, target_path, output_path, use_interpolation=True):
        """
        Przetwarzanie dużych obrazów po kawałkach z blendingiem
        """
        import gc
        from PIL import Image
        
        print(f"Processing large image: {source_path}")
        
        # Wczytaj target histogram raz
        target_array = self.extract_target_histogram(target_path)
        print(f"Target loaded: {target_array.shape}")
        
        # Otwórz source image bez ładowania do pamięci
        with Image.open(source_path) as source_img:
            width, height = source_img.size
            print(f"Source size: {width}x{height}")
            
            # Przygotuj output image
            output_img = Image.new('RGB', (width, height))
            
            # Oblicz tiles z overlap
            tiles_x = range(0, width, self.tile_size - self.overlap)
            tiles_y = range(0, height, self.tile_size - self.overlap)
            
            total_tiles = len(list(tiles_x)) * len(list(tiles_y))
            processed_tiles = 0
            
            print(f"Processing {total_tiles} tiles...")
            
            for y in tiles_y:
                for x in tiles_x:
                    # Oblicz granice tile'a
                    x_end = min(x + self.tile_size, width)
                    y_end = min(y + self.tile_size, height)
                    
                    # Wytnij tile
                    tile_box = (x, y, x_end, y_end)
                    tile = source_img.crop(tile_box).convert('RGB')
                    tile_array = np.array(tile)
                    
                    # Przetwórz tile
                    try:
                        result_tile_array = self.apply_histogram_matching(
                            tile_array, target_array, use_interpolation
                        )
                        result_tile = Image.fromarray(result_tile_array)
                    except Exception as e:
                        print(f"Error processing tile at ({x},{y}): {e}")
                        result_tile = tile  # Fallback do oryginału
                    
                    # Obsługa overlap - blend edges
                    if self.overlap > 0 and (x > 0 or y > 0):
                        result_tile = self._blend_tile_edges(
                            output_img, result_tile, tile_box, x > 0, y > 0
                        )
                    
                    # Wklej tile do output
                    output_img.paste(result_tile, (x, y))
                    
                    processed_tiles += 1
                    if processed_tiles % 10 == 0:
                        print(f"Processed {processed_tiles}/{total_tiles} tiles")
                    
                    # Cleanup
                    del tile, tile_array, result_tile_array, result_tile
                    gc.collect()
            
            # Zapisz wynik
            output_img.save(output_path, quality=95, optimize=True)
            print(f"Saved: {output_path}")
            
        return True
    
    def _blend_tile_edges(self, base_img, new_tile, tile_box, blend_left, blend_top):
        """
        Płynne łączenie krawędzi tiles
        """
        x, y, x_end, y_end = tile_box
        tile_w = x_end - x
        tile_h = y_end - y
        
        # Konwertuj do numpy dla łatwiejszego blendingu
        new_tile_array = np.array(new_tile)
        
        # Blend left edge
        if blend_left and x > 0:
            blend_width = min(self.overlap // 2, tile_w // 4)
            
            # Pobierz część z base image
            base_region = base_img.crop((x, y, x + blend_width, y_end))
            base_array = np.array(base_region)
            
            # Utwórz gradient alpha
            alpha = np.linspace(0, 1, blend_width).reshape(1, -1, 1)
            
            # Blend
            if base_array.shape == new_tile_array[:, :blend_width].shape:
                blended = (1 - alpha) * base_array + alpha * new_tile_array[:, :blend_width]
                new_tile_array[:, :blend_width] = blended.astype(np.uint8)
        
        # Blend top edge
        if blend_top and y > 0:
            blend_height = min(self.overlap // 2, tile_h // 4)
            
            # Pobierz część z base image
            base_region = base_img.crop((x, y, x_end, y + blend_height))
            base_array = np.array(base_region)
            
            # Utwórz gradient alpha
            alpha = np.linspace(0, 1, blend_height).reshape(-1, 1, 1)
            
            # Blend
            if base_array.shape == new_tile_array[:blend_height].shape:
                blended = (1 - alpha) * base_array + alpha * new_tile_array[:blend_height]
                new_tile_array[:blend_height] = blended.astype(np.uint8)
        
        return Image.fromarray(new_tile_array)
    
    def estimate_memory_usage(self, width, height, tile_size):
        """
        Estymacja zużycia pamięci
        """
        # Bytes per pixel dla RGB uint8
        bytes_per_pixel = 3
        
        # Pamięć dla tile
        tile_memory = tile_size * tile_size * bytes_per_pixel
        
        # Dodatkowa pamięć dla obliczeń (histogramy, LUTs, etc.)
        overhead_memory = tile_memory * 2
        
        total_mb = (tile_memory + overhead_memory) / (1024 * 1024)
        
        return {
            'tile_memory_mb': tile_memory / (1024 * 1024),
            'overhead_mb': overhead_memory / (1024 * 1024),
            'total_mb': total_mb,
            'tiles_count': ((width // tile_size) + 1) * ((height // tile_size) + 1)
        }
```

### Numba Acceleration

```python
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

class AcceleratedHistogramMatcher(SimpleHistogramMatching):
    def __init__(self):
        super().__init__()
        self.numba_available = NUMBA_AVAILABLE
        
    @jit(nopython=True, parallel=True) if NUMBA_AVAILABLE else lambda x: x
    def _fast_lut_apply(self, image, lut):
        """
        Szybkie stosowanie LUT z Numba
        """
        height, width = image.shape
        result = np.empty_like(image)
        
        for i in prange(height):
            for j in range(width):
                result[i, j] = lut[image[i, j]]
        
        return result
    
    @jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
    def _fast_histogram(self, data):
        """
        Szybkie obliczanie histogramu z Numba
        """
        hist = np.zeros(256, dtype=np.int32)
        
        for i in range(data.size):
            hist[data.flat[i]] += 1
            
        return hist
    
    def apply_histogram_matching(self, source_array, target_array, use_interpolation=True):
        """
        Accelerated version using Numba where available
        """
        if not self.numba_available:
            return super().apply_histogram_matching(source_array, target_array, use_interpolation)
        
        # Walidacja wejściowa
        if source_array.size == 0 or target_array.size == 0:
            raise ValueError("Empty arrays provided")
        
        if source_array.dtype != np.uint8:
            source_array = np.clip(source_array, 0, 255).astype(np.uint8)
        
        if target_array.dtype != np.uint8:
            target_array = np.clip(target_array, 0, 255).astype(np.uint8)
        
        result_array = np.zeros_like(source_array)
        
        for channel in range(3):
            # Sprawdź czy kanał nie jest monochromatyczny
            if np.std(source_array[:, :, channel]) < 1e-6:
                result_array[:, :, channel] = source_array[:, :, channel]
                continue
            
            # Użyj fast histogram jeśli dostępne
            if self.numba_available:
                source_hist = self._fast_histogram(source_array[:, :, channel])
                target_hist = self._fast_histogram(target_array[:, :, channel])
            else:
                source_hist = self.calculate_histogram(source_array, channel)
                target_hist = self.calculate_histogram(target_array, channel)
            
            # Oblicz CDF
            source_cdf = self.calculate_cdf(source_hist)
            target_cdf = self.calculate_cdf(target_hist)
            
            # Utwórz lookup table
            try:
                if use_interpolation:
                    lut = self.create_lookup_table_interpolated(source_cdf, target_cdf)
                else:
                    lut = self.create_lookup_table(source_cdf, target_cdf)
            except Exception as e:
                print(f"Warning: Interpolation failed for channel {channel}, using simple LUT: {e}")
                lut = self.create_lookup_table(source_cdf, target_cdf)
            
            # Zastosuj transformację - użyj fast version jeśli dostępne
            if self.numba_available:
                result_array[:, :, channel] = self._fast_lut_apply(
                    source_array[:, :, channel], lut
                )
            else:
                result_array[:, :, channel] = lut[source_array[:, :, channel]]
        
        return result_array
```

### Comprehensive Quality Metrics

```python
class QualityAnalyzer:
    def __init__(self):
        self.name = "Histogram Matching Quality Analyzer"
    
    def comprehensive_quality_metrics(self, source_array, target_array, result_array):
        """
        Kompleksowa analiza jakości z wieloma metrykami
        """
        try:
            from scipy.stats import wasserstein_distance, ks_2samp
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import peak_signal_noise_ratio as psnr
            scipy_available = True
        except ImportError:
            print("Warning: scipy/skimage not available, using basic metrics only")
            scipy_available = False
        
        metrics = {}
        
        for ch, ch_name in enumerate(['Red', 'Green', 'Blue']):
            ch_metrics = {}
            
            source_ch = source_array[:, :, ch].astype(float)
            target_ch = target_array[:, :, ch].astype(float)
            result_ch = result_array[:, :, ch].astype(float)
            
            # Podstawowe metryki
            ch_metrics['mse_to_target'] = np.mean((result_ch - target_ch)**2)
            ch_metrics['mse_to_source'] = np.mean((result_ch - source_ch)**2)
            ch_metrics['mae_to_target'] = np.mean(np.abs(result_ch - target_ch))
            ch_metrics['mean_diff_target'] = abs(np.mean(result_ch) - np.mean(target_ch))
            ch_metrics['std_diff_target'] = abs(np.std(result_ch) - np.std(target_ch))
            ch_metrics['std_ratio_target'] = np.std(result_ch) / (np.std(target_ch) + 1e-10)
            
            # Metryki histogramów
            hist_source = np.histogram(source_ch, bins=256, range=(0, 255))[0]
            hist_target = np.histogram(target_ch, bins=256, range=(0, 255))[0]
            hist_result = np.histogram(result_ch, bins=256, range=(0, 255))[0]
            
            # Normalizuj histogramy
            hist_source_norm = hist_source / np.sum(hist_source)
            hist_target_norm = hist_target / np.sum(hist_target)
            hist_result_norm = hist_result / np.sum(hist_result)
              # Korelacja histogramów
            if np.std(hist_target_norm) > 1e-10 and np.std(hist_result_norm) > 1e-10:
                ch_metrics['hist_corr_target'] = np.corrcoef(hist_result_norm, hist_target_norm)[0,1]
            else:
                ch_metrics['hist_corr_target'] = 0.0
            
            # Chi-square distance
            ch_metrics['chi2_target'] = np.sum((hist_result_norm - hist_target_norm)**2 / 
                                             (hist_target_norm + 1e-10))
            
            # KL divergence
            ch_metrics['kl_divergence'] = np.sum(hist_result_norm * np.log(
                (hist_result_norm + 1e-10) / (hist_target_norm + 1e-10)
            ))
            
            if scipy_available:
                # Earth Mover's Distance (Wasserstein)
                ch_metrics['emd_to_target'] = wasserstein_distance(
                    np.arange(256), np.arange(256), 
                    hist_result_norm, hist_target_norm
                )
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = ks_2samp(result_ch.flatten(), target_ch.flatten())
                ch_metrics['ks_statistic'] = ks_stat
                ch_metrics['ks_p_value'] = ks_p
                
                # PSNR
                if np.max(target_ch) > np.min(target_ch):
                    ch_metrics['psnr'] = psnr(target_ch, result_ch, data_range=255)
                else:
                    ch_metrics['psnr'] = float('inf')
                
                # SSIM
                if target_ch.shape[0] > 6 and target_ch.shape[1] > 6:  # SSIM requires minimum size
                    ch_metrics['ssim'] = ssim(target_ch, result_ch, data_range=255)
                else:
                    ch_metrics['ssim'] = 1.0
            
            # Perceptual hash similarity (simplified)
            ch_metrics['perceptual_sim'] = self._perceptual_similarity(result_ch, target_ch)
            
            metrics[ch_name] = ch_metrics
        
        # Overall metrics
        overall_metrics = self._calculate_overall_metrics(metrics)
        metrics['Overall'] = overall_metrics
        
        return metrics
    
    def _perceptual_similarity(self, img1, img2, hash_size=8):
        """
        Uproszczona miara podobieństwa perceptualnego
        """
        # Resize to hash_size x hash_size
        from PIL import Image
        
        img1_pil = Image.fromarray(img1.astype(np.uint8)).resize((hash_size, hash_size))
        img2_pil = Image.fromarray(img2.astype(np.uint8)).resize((hash_size, hash_size))
        
        # Convert to grayscale for simplicity
        img1_gray = np.array(img1_pil.convert('L')).flatten()
        img2_gray = np.array(img2_pil.convert('L')).flatten()
        
        # Calculate hash similarity
        mean1, mean2 = np.mean(img1_gray), np.mean(img2_gray)
        hash1 = img1_gray > mean1
        hash2 = img2_gray > mean2
        
        # Hamming distance
        hamming_dist = np.sum(hash1 != hash2)
        similarity = 1.0 - (hamming_dist / len(hash1))
        
        return similarity
    
    def _calculate_overall_metrics(self, channel_metrics):
        """
        Oblicz metryki ogólne ze wszystkich kanałów
        """
        overall = {}
        
        # Średnie ze wszystkich kanałów
        metric_keys = channel_metrics['Red'].keys()
        
        for key in metric_keys:
            values = [channel_metrics[ch][key] for ch in ['Red', 'Green', 'Blue'] 
                     if not np.isnan(channel_metrics[ch][key])]
            
            if values:
                overall[f'{key}_mean'] = np.mean(values)
                overall[f'{key}_std'] = np.std(values)
                overall[f'{key}_min'] = np.min(values)
                overall[f'{key}_max'] = np.max(values)
        
        # Composite quality score (0-100)
        quality_components = []
        
        # Histogram correlation (weight: 30%)
        hist_corr = overall.get('hist_corr_target_mean', 0)
        quality_components.append(max(0, hist_corr) * 30)
        
        # SSIM (weight: 25%)
        if 'ssim_mean' in overall:
            quality_components.append(overall['ssim_mean'] * 25)
        
        # Inverse MSE (weight: 20%)
        mse = overall.get('mse_to_target_mean', 1000)
        mse_score = max(0, 1 - mse / 1000) * 20  # Normalize to 0-20
        quality_components.append(mse_score)
        
        # Perceptual similarity (weight: 25%)
        perc_sim = overall.get('perceptual_sim_mean', 0)
        quality_components.append(perc_sim * 25)
        
        overall['quality_score'] = sum(quality_components)
        
        return overall
    
    def print_quality_report(self, metrics, title="Quality Analysis Report"):
        """
        Drukuj sformatowany raport jakości
        """
        print("\n" + "="*60)
        print(f"📊 {title}")
        print("="*60)
        
        # Wyniki ogólne
        if 'Overall' in metrics:
            overall = metrics['Overall']
            quality_score = overall.get('quality_score', 0)
            
            print(f"\n🎯 OVERALL QUALITY SCORE: {quality_score:.1f}/100")
            
            if quality_score >= 85:
                grade = "A (Excellent)"
            elif quality_score >= 75:
                grade = "B (Good)"
            elif quality_score >= 65:
                grade = "C (Fair)" 
            elif quality_score >= 50:
                grade = "D (Poor)"
            else:
                grade = "F (Failed)"
                
            print(f"🏆 GRADE: {grade}")
        
        # Szczegóły dla każdego kanału
        for ch_name in ['Red', 'Green', 'Blue']:
            if ch_name in metrics:
                ch_metrics = metrics[ch_name]
                print(f"\n📈 {ch_name} Channel:")
                print("-" * 30)
                
                # Kluczowe metryki
                print(f"  Histogram Correlation: {ch_metrics.get('hist_corr_target', 0):.4f}")
                print(f"  MSE to Target:        {ch_metrics.get('mse_to_target', 0):.2f}")
                print(f"  Mean Difference:      {ch_metrics.get('mean_diff_target', 0):.2f}")
                
                if 'ssim' in ch_metrics:
                    print(f"  SSIM:                 {ch_metrics['ssim']:.4f}")
                if 'psnr' in ch_metrics:
                    psnr_val = ch_metrics['psnr']
                    if np.isfinite(psnr_val):
                        print(f"  PSNR:                 {psnr_val:.2f} dB")
                    else:
                        print(f"  PSNR:                 ∞ dB (perfect match)")
                
                print(f"  Perceptual Similarity: {ch_metrics.get('perceptual_sim', 0):.4f}")
        
        print("\n" + "="*60)
```

### Comprehensive Testing Suite

```python
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

class TestHistogramMatchingComprehensive(unittest.TestCase):
    def setUp(self):
        self.matcher = SimpleHistogramMatching()
        self.advanced_matcher = CachedHistogramMatcher()
        self.quality_analyzer = QualityAnalyzer()
        
        # Testowe obrazy
        np.random.seed(42)  # For reproducible tests
        self.test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.uniform_image = np.full((50, 50, 3), 128, dtype=np.uint8)
        self.gradient_image = self._create_gradient_image()
        
        # Temporary directory for file tests
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Cleanup temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_gradient_image(self, size=(100, 100)):
        """Create a gradient image for testing"""
        image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        for y in range(size[0]):
            for x in range(size[1]):
                image[y, x, 0] = int((x / size[1]) * 255)  # Red gradient
                image[y, x, 1] = int((y / size[0]) * 255)  # Green gradient
                image[y, x, 2] = 128  # Constant blue
                
        return image
    
    def test_histogram_calculation_edge_cases(self):
        """Test histogram calculation with edge cases"""
        # Test z obrazem czarnym
        black_image = np.zeros((50, 50, 3), dtype=np.uint8)
        hist = self.matcher.calculate_histogram(black_image, 0)
        
        self.assertEqual(hist[0], 50*50)  # All pixels at value 0
        self.assertEqual(np.sum(hist[1:]), 0)  # No pixels at other values
        
        # Test z obrazem białym
        white_image = np.full((50, 50, 3), 255, dtype=np.uint8)
        hist = self.matcher.calculate_histogram(white_image, 0)
        
        self.assertEqual(hist[255], 50*50)  # All pixels at value 255
        self.assertEqual(np.sum(hist[:255]), 0)  # No pixels at other values
    
    def test_cdf_properties(self):
        """Test mathematical properties of CDF"""
        hist = self.matcher.calculate_histogram(self.test_image, 0)
        cdf = self.matcher.calculate_cdf(hist)
        
        # CDF should be monotonically non-decreasing
        self.assertTrue(np.all(np.diff(cdf) >= -1e-10))  # Allow small floating point errors
        
        # CDF should start at or above 0
        self.assertGreaterEqual(cdf[0], 0)
        
        # CDF should end at 1
        self.assertAlmostEqual(cdf[-1], 1.0, places=10)
    
    def test_lookup_table_bounds(self):
        """Test that LUT values are within valid bounds"""
        source_cdf = np.linspace(0, 1, 256)
        target_cdf = np.linspace(0, 1, 256)
        
        lut = self.matcher.create_lookup_table(source_cdf, target_cdf)
        
        # All LUT values should be in [0, 255]
        self.assertGreaterEqual(np.min(lut), 0)
        self.assertLessEqual(np.max(lut), 255)
        
        # LUT should be monotonically non-decreasing for monotonic CDFs
        # (with some tolerance for floating point errors)
        diff = np.diff(lut.astype(int))
        self.assertTrue(np.all(diff >= -1))  # Allow small decreases due to quantization
    
    def test_interpolated_lut_robustness(self):
        """Test interpolated LUT with problematic CDFs"""
        # CDF with many duplicates
        duplicate_cdf = np.array([0.0] * 100 + [0.5] * 100 + [1.0] * 56)
        source_cdf = np.linspace(0, 1, 256)
        
        # Should not raise exception
        lut = self.matcher.create_lookup_table_interpolated(source_cdf, duplicate_cdf)
        
        self.assertEqual(len(lut), 256)
        self.assertGreaterEqual(np.min(lut), 0)
        self.assertLessEqual(np.max(lut), 255)
    
    def test_identity_matching_precision(self):
        """Test that matching image to itself gives minimal changes"""
        result = self.matcher.apply_histogram_matching(
            self.test_image, self.test_image
        )
        
        # Changes should be minimal (due to discretization only)
        max_diff = np.max(np.abs(result.astype(int) - self.test_image.astype(int)))
        self.assertLessEqual(max_diff, 2)  # Allow small quantization errors
    
    def test_extreme_histogram_cases(self):
        """Test with extreme histogram cases"""
        # Single value images
        black_image = np.zeros((50, 50, 3), dtype=np.uint8)
        white_image = np.full((50, 50, 3), 255, dtype=np.uint8)
        
        # Should not crash
        result1 = self.matcher.apply_histogram_matching(black_image, white_image)
        result2 = self.matcher.apply_histogram_matching(white_image, black_image)
        
        self.assertEqual(result1.shape, black_image.shape)
        self.assertEqual(result2.shape, white_image.shape)
    
    def test_cached_matcher_functionality(self):
        """Test caching functionality"""
        # First call - should be cache miss
        result1 = self.advanced_matcher.apply_histogram_matching(
            self.test_image, self.gradient_image
        )
        
        stats1 = self.advanced_matcher.get_cache_stats()
        self.assertEqual(stats1['cache_size'], 1)
        self.assertEqual(stats1['misses'], 1)
        
        # Second call with same images - should be cache hit
        result2 = self.advanced_matcher.apply_histogram_matching(
            self.test_image, self.gradient_image
        )
        
        stats2 = self.advanced_matcher.get_cache_stats()
        self.assertEqual(stats2['hits'], 1)
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
    
    def test_quality_metrics_comprehensive(self):
        """Test comprehensive quality metrics"""
        # Create a simple test case
        source = self.test_image
        target = self.gradient_image
        result = self.matcher.apply_histogram_matching(source, target)
        
        metrics = self.quality_analyzer.comprehensive_quality_metrics(
            source, target, result
        )
        
        # Check that all expected metrics are present
        expected_channels = ['Red', 'Green', 'Blue', 'Overall']
        for channel in expected_channels:
            self.assertIn(channel, metrics)
        
        # Check some specific metrics
        for ch in ['Red', 'Green', 'Blue']:
            ch_metrics = metrics[ch]
            
            # All metrics should be finite
            for key, value in ch_metrics.items():
                self.assertTrue(np.isfinite(value), f"{ch}.{key} is not finite: {value}")
            
            # Histogram correlation should be between -1 and 1
            hist_corr = ch_metrics.get('hist_corr_target', 0)
            self.assertGreaterEqual(hist_corr, -1.0)
            self.assertLessEqual(hist_corr, 1.0)
            
            # MSE should be non-negative
            mse = ch_metrics.get('mse_to_target', 0)
            self.assertGreaterEqual(mse, 0)
        
        # Overall quality score should be between 0 and 100
        quality_score = metrics['Overall'].get('quality_score', 0)
        self.assertGreaterEqual(quality_score, 0)
        self.assertLessEqual(quality_score, 100)
    
    def test_memory_efficient_processing(self):
        """Test memory-efficient processing estimation"""
        mem_matcher = MemoryEfficientMatcher(tile_size=512)
        
        # Test memory estimation
        memory_info = mem_matcher.estimate_memory_usage(2048, 2048, 512)
        
        self.assertIn('total_mb', memory_info)
        self.assertIn('tiles_count', memory_info)
        self.assertGreater(memory_info['total_mb'], 0)
        self.assertGreater(memory_info['tiles_count'], 0)
    
    @unittest.mock.patch('PIL.Image.open')
    def test_file_processing_error_handling(self, mock_open):
        """Test error handling in file processing"""
        # Mock file that raises exception
        mock_open.side_effect = FileNotFoundError("Test file not found")
        
        # Should handle error gracefully
        success = self.matcher.process_images(
            "nonexistent_source.jpg",
            "nonexistent_target.jpg", 
            "output.jpg"
        )
        
        self.assertFalse(success)
    
    def test_batch_processing_with_mock(self):
        """Test batch processing with mocked components"""
        batch_matcher = BatchHistogramMatcher()
        
        # Create some test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"test_{i}.png")
            Image.fromarray(self.test_image).save(file_path)
            test_files.append(file_path)
        
        target_path = os.path.join(self.temp_dir, "target.png")
        Image.fromarray(self.gradient_image).save(target_path)
        
        output_dir = os.path.join(self.temp_dir, "output")
        
        # Mock the heavy processing to avoid actual computation
        with patch.object(batch_matcher, '_process_single_image', return_value=True):
            results, failed = batch_matcher.batch_process_with_progress(
                test_files, target_path, output_dir, max_workers=1
            )
        
        self.assertEqual(len(results), 3)
        self.assertEqual(len(failed), 0)
    
    def test_data_validation(self):
        """Test input data validation"""
        # Empty arrays
        empty_array = np.array([]).reshape(0, 0, 3).astype(np.uint8)
        
        with self.assertRaises(ValueError):
            self.matcher.apply_histogram_matching(empty_array, self.test_image)
        
        # Wrong data types
        float_image = self.test_image.astype(np.float32) / 255.0
        
        # Should handle automatic conversion
        result = self.matcher.apply_histogram_matching(float_image, self.test_image)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_preservation_features(self):
        """Test histogram matching with preservation"""
        original_brightness = np.mean(self.test_image)
        
        result = self.matcher.histogram_matching_with_preservation(
            self.test_image, self.gradient_image, 
            preserve_brightness=True
        )
        
        result_brightness = np.mean(result)
        
        # Brightness should be approximately preserved
        brightness_diff = abs(original_brightness - result_brightness)
        self.assertLess(brightness_diff, 5.0)  # Allow some tolerance
    
    def test_exact_histogram_specification(self):
        """Test exact histogram specification"""
        # Create specific histograms
        target_hists = [
            np.ones(256) * 100,  # Uniform for red
            np.zeros(256),       # Empty for green (should handle gracefully)
            np.zeros(256)        # Empty for blue
        ]
        target_hists[1][128] = 5000  # Single spike for green
        target_hists[2][64] = 2500   # Single spike for blue at different position
        target_hists[2][192] = 2500
        
        result = self.matcher.exact_histogram_specification(
            self.test_image, target_hists
        )
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, self.test_image.dtype)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
```

---

## Performance Benchmarks and Optimization Guide

### Benchmark Results

```python
class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
    
    def run_comprehensive_benchmark(self):
        """
        Uruchom kompletny benchmark różnych implementacji
        """
        import time
        
        # Test images of different sizes
        test_sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
        implementations = {
            'Basic': SimpleHistogramMatching(),
            'Cached': CachedHistogramMatcher(),
            'Accelerated': AcceleratedHistogramMatcher() if NUMBA_AVAILABLE else None
        }
        
        print("🚀 Performance Benchmark Results")
        print("=" * 60)
        
        for size in test_sizes:
            print(f"\n📏 Image Size: {size[0]}x{size[1]}")
            print("-" * 40)
            
            # Generate test images
            source = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            target = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            
            for name, impl in implementations.items():
                if impl is None:
                    continue
                    
                # Warm up
                _ = impl.apply_histogram_matching(source[:100, :100], target[:100, :100])
                
                # Benchmark
                times = []
                for _ in range(3):  # 3 runs for average
                    start_time = time.time()
                    result = impl.apply_histogram_matching(source, target)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                # Memory usage estimate
                memory_mb = (source.nbytes + target.nbytes + result.nbytes) / (1024**2)
                
                print(f"  {name:12s}: {avg_time:.3f}±{std_time:.3f}s  "
                      f"({memory_mb:.1f}MB, {source.size/avg_time/1e6:.2f}MP/s)")
                
                # Store results
                if size not in self.results:
                    self.results[size] = {}
                self.results[size][name] = {
                    'time': avg_time,
                    'std': std_time,
                    'memory_mb': memory_mb,
                    'throughput_mps': source.size / avg_time / 1e6
                }
        
        return self.results
    
    def print_optimization_recommendations(self):
        """
        Wydrukuj rekomendacje optymalizacji na podstawie wyników
        """
        print("\n🎯 Optimization Recommendations")
        print("=" * 60)
        
        recommendations = [
            "1. 🚀 **Use Numba**: Install numba for 2-5x speedup",
            "2. 💾 **Enable Caching**: For repeated processing of similar images", 
            "3. 🔧 **Tile Processing**: For images >2048x2048 to reduce memory usage",
            "4. ⚡ **Interpolation**: Disable for faster processing if quality allows",
            "5. 🔄 **Batch Processing**: Use multiprocessing for multiple images",
            "6. 🎛️ **Adaptive Matching**: Use alpha<1.0 for gentler, faster processing"
        ]
        
        for rec in recommendations:
            print(rec)
        
        print(f"\n📊 **Memory Usage Guidelines**:")
        print(f"  • Images <1024x1024: Use basic implementation")
        print(f"  • Images 1024-2048: Consider caching if processing multiple")
        print(f"  • Images >2048x2048: Use tiled processing")
        print(f"  • Batch processing: Use 2-4 workers depending on CPU cores")
```

---

## Integration Example

```python
class PhotoshopHistogramMatcher:
    """
    Przykład integracji z większym systemem edycji zdjęć
    """
    def __init__(self, quality_preset='balanced'):
        self.quality_presets = {
            'fast': {
                'use_interpolation': False,
                'tile_size': 1024,
                'use_cache': True,
                'alpha': 0.8
            },
            'balanced': {
                'use_interpolation': True,
                'tile_size': 2048,
                'use_cache': True,
                'alpha': 1.0
            },
            'quality': {
                'use_interpolation': True,
                'tile_size': 4096,
                'use_cache': False,
                'alpha': 1.0,
                'use_lab': True
            }
        }
        
        self.current_preset = self.quality_presets[quality_preset]
        self.matcher = self._create_matcher()
        
    def _create_matcher(self):
        """Create appropriate matcher based on preset"""
        if self.current_preset.get('use_cache', False):
            return CachedHistogramMatcher()
        else:
            return AcceleratedHistogramMatcher()
    
    def match_histogram(self, source_path, target_path, output_path, 
                       progress_callback=None):
        """
        Main interface for histogram matching
        """
        try:
            if progress_callback:
                progress_callback(0, "Loading images...")
            
            # Load images
            source_img = Image.open(source_path).convert('RGB')
            source_array = np.array(source_img)
            
            # Check if we need tiled processing
            needs_tiling = (source_array.shape[0] * source_array.shape[1] > 
                          self.current_preset['tile_size']**2)
            
            if needs_tiling:
                if progress_callback:
                    progress_callback(20, "Processing large image in tiles...")
                
                mem_matcher = MemoryEfficientMatcher(
                    tile_size=self.current_preset['tile_size']
                )
                success = mem_matcher.process_large_image(
                    source_path, target_path, output_path
                )
            else:
                if progress_callback:
                    progress_callback(20, "Processing histogram matching...")
                
                # Use LAB color space if requested
                if self.current_preset.get('use_lab', False):
                    lab_matcher = LABHistogramMatching()
                    target_array = lab_matcher.extract_target_histogram(target_path)
                    result_array = lab_matcher.lab_histogram_matching(
                        source_array, target_array
                    )
                else:
                    target_array = self.matcher.extract_target_histogram(target_path)
                    
                    # Apply adaptive matching if alpha < 1
                    alpha = self.current_preset.get('alpha', 1.0)
                    if alpha < 1.0:
                        result_array = self.matcher.adaptive_histogram_matching(
                            source_array, target_array, alpha=alpha
                        )
                    else:
                        result_array = self.matcher.apply_histogram_matching(
                            source_array, target_array,
                            use_interpolation=self.current_preset['use_interpolation']
                        )
                
                # Save result
                result_img = Image.fromarray(result_array)
                result_img.save(output_path, quality=95)
                success = True
            
            if progress_callback:
                progress_callback(100, "Complete!")
            
            return success
            
        except Exception as e:
            if progress_callback:
                progress_callback(-1, f"Error: {str(e)}")
            return False
    
    def batch_process(self, file_pairs, output_dir, progress_callback=None):
        """
        Process multiple image pairs
        """
        batch_matcher = BatchHistogramMatcher()
        
        # Wrapper for progress callback
        def batch_progress(processed, total, current_file):
            if progress_callback:
                percent = int((processed / total) * 100)
                progress_callback(percent, f"Processing {current_file}")
        
        source_files = [pair[0] for pair in file_pairs]
        target_files = [pair[1] for pair in file_pairs]
        
        # For simplicity, use first target for all (extend as needed)
        target_path = target_files[0]
        
        results, failed = batch_matcher.batch_process_with_progress(
            source_files, target_path, output_dir
        )
        
        return len(results), len(failed)

# Usage example
if __name__ == "__main__":
    # Create photoshop-style matcher
    ps_matcher = PhotoshopHistogramMatcher(quality_preset='balanced')
    
    # Simple progress callback
    def progress_update(percent, message):
        if percent >= 0:
            print(f"[{percent:3d}%] {message}")
        else:
            print(f"[ERROR] {message}")
    
    # Process single image
    success = ps_matcher.match_histogram(
        "input_photo.jpg",
        "reference_style.jpg", 
        "output_matched.jpg",
        progress_callback=progress_update
    )
    
    if success:
        print("✅ Histogram matching completed successfully!")
    else:
        print("❌ Histogram matching failed!")
```

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Gotowy do implementacji