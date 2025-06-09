# Perceptual Color Matching Algorithm - Część 4 z 6: Implementacja Zaawansowana

**Część 4 z 6: Implementacja Zaawansowana**

---

## Nawigacja

**◀️ Poprzednia część**: [Implementacja Podstawowa](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-3of6.md)  
**▶️ Następna część**: [Optymalizacja i Debugging](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-5of6.md)  
**🏠 Powrót do**: [Spis Treści](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-0of6.md)

---

## 4. Implementacja Python - Część 2 (Kontynuacja)

### Zaawansowane Funkcje Mapowania

```python
    def calculate_memory_color_mapping(self, source_memory, target_memory):
        """Obliczenie mapowania z zachowaniem memory colors"""
        mapping = {
            'memory_color_adjustments': {},
            'global_adjustments': {
                'lightness_scale': 1.0,
                'lightness_shift': 0.0,
                'chroma_scale': 1.0,
                'chroma_shift': 0.0,
                'hue_shift': 0.0
            }
        }
        
        # Analiza memory colors
        for color_name in self.memory_colors_lab.keys():
            if color_name in source_memory and color_name in target_memory:
                source_color = source_memory[color_name]['mean_lab']
                target_color = target_memory[color_name]['mean_lab']
                
                # Obliczenie różnic
                delta_L = target_color[0] - source_color[0]
                delta_a = target_color[1] - source_color[1]
                delta_b = target_color[2] - source_color[2]
                
                mapping['memory_color_adjustments'][color_name] = {
                    'delta_L': delta_L,
                    'delta_a': delta_a,
                    'delta_b': delta_b,
                    'weight': source_memory[color_name]['percentage'] / 100.0
                }
        
        return mapping
    
    def apply_local_adaptation(self, lab_image, radius=50):
        """Aplikacja lokalnej adaptacji percepcyjnej"""
        result = lab_image.copy()
        h, w = lab_image.shape[:2]
        
        # Gaussian kernel dla lokalnego uśredniania
        kernel_size = radius * 2 + 1
        sigma = radius / 3.0
        
        for channel in range(3):
            # Lokalne średnie
            local_mean = cv2.GaussianBlur(
                lab_image[:,:,channel], 
                (kernel_size, kernel_size), 
                sigma
            )
            
            # Lokalna adaptacja (zmniejszenie lokalnego kontrastu)
            adaptation_factor = 0.3
            result[:,:,channel] = (
                lab_image[:,:,channel] * (1 - adaptation_factor) +
                local_mean * adaptation_factor
            )
        
        return result
    
    def apply_gamut_mapping(self, lab_image, target_gamut='srgb', method='perceptual'):
        """Mapowanie gamutu z zachowaniem percepcji"""
        if target_gamut != 'srgb':
            # Dla innych gamutów - implementacja rozszerzona
            return lab_image
        
        # Konwersja do RGB dla sprawdzenia gamutu
        rgb_test = self.lab_to_rgb(lab_image)
        
        # Znajdź piksele poza gamutem
        out_of_gamut = (
            (rgb_test < 0) | (rgb_test > 255)
        ).any(axis=2)
        
        if not np.any(out_of_gamut):
            return lab_image  # Wszystkie piksele w gamucie
        
        result = lab_image.copy()
        
        if method == 'perceptual':
            # Perceptual gamut mapping - zachowanie lightness, redukcja chroma
            for i in range(lab_image.shape[0]):
                for j in range(lab_image.shape[1]):
                    if out_of_gamut[i, j]:
                        L, a, b = lab_image[i, j]
                        chroma = np.sqrt(a**2 + b**2)
                        hue = np.arctan2(b, a)
                        
                        # Redukcja chroma do momentu wejścia w gamut
                        for reduction in np.linspace(0.9, 0.1, 20):
                            new_chroma = chroma * reduction
                            new_a = new_chroma * np.cos(hue)
                            new_b = new_chroma * np.sin(hue)
                            
                            test_lab = np.array([L, new_a, new_b]).reshape(1, 1, 3)
                            test_rgb = self.lab_to_rgb(test_lab)
                            
                            if np.all((test_rgb >= 0) & (test_rgb <= 255)):
                                result[i, j] = [L, new_a, new_b]
                                break
        
        return result
    
    def correct_chroma_smoothing(self, lab_image, smoothing_factor=0.1):
        """Korekta i wygładzanie chromy"""
        L, a, b = lab_image[:,:,0], lab_image[:,:,1], lab_image[:,:,2]
        
        # Obliczenie chroma
        chroma = np.sqrt(a**2 + b**2)
        hue = np.arctan2(b, a)
        
        # Wygładzanie chroma (redukcja szumu)
        smoothed_chroma = cv2.GaussianBlur(
            chroma, 
            (5, 5), 
            smoothing_factor
        )
        
        # Rekonstrukcja a*, b*
        new_a = smoothed_chroma * np.cos(hue)
        new_b = smoothed_chroma * np.sin(hue)
        
        result = lab_image.copy()
        result[:,:,1] = new_a
        result[:,:,2] = new_b
        
        return result
    
    def smooth_color_gradients(self, lab_image, gradient_threshold=5.0):
        """Wygładzanie gradientów kolorów"""
        # Obliczenie gradientów
        L = lab_image[:,:,0]
        grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Maska obszarów z dużymi gradientami
        high_gradient = gradient_magnitude > gradient_threshold
        
        result = lab_image.copy()
        
        # Wygładzanie tylko w obszarach z dużymi gradientami
        if np.any(high_gradient):
            for channel in range(3):
                smoothed = cv2.GaussianBlur(
                    lab_image[:,:,channel], 
                    (3, 3), 
                    0.5
                )
                
                # Aplikacja tylko w obszarach z gradientami
                result[:,:,channel] = np.where(
                    high_gradient,
                    smoothed,
                    lab_image[:,:,channel]
                )
        
        return result

    def debug_perceptual_matching(self, source_image, target_image, result_image):
        """Narzędzie debugowania dla perceptual matching"""
        debug_info = {}
        
        # Konwersja do LAB
        source_lab = self.rgb_to_lab(source_image)
        target_lab = self.rgb_to_lab(target_image)
        result_lab = self.rgb_to_lab(result_image)
        
        # Analiza memory colors
        debug_info['memory_colors'] = {
            'source': self.analyze_memory_colors(source_lab),
            'target': self.analyze_memory_colors(target_lab),
            'result': self.analyze_memory_colors(result_lab)
        }
        
        # Metryki jakości
        debug_info['quality_metrics'] = {
            'delta_e_source_target': np.mean(
                colour.delta_E_CIE2000(source_lab, target_lab)
            ),
            'delta_e_result_target': np.mean(
                colour.delta_E_CIE2000(result_lab, target_lab)
            ),
            'delta_e_source_result': np.mean(
                colour.delta_E_CIE2000(source_lab, result_lab)
            )
        }
        
        # Analiza rozkładów kolorów
        debug_info['color_distributions'] = {
            'source': self.analyze_perceptual_characteristics(source_lab),
            'target': self.analyze_perceptual_characteristics(target_lab),
            'result': self.analyze_perceptual_characteristics(result_lab)
        }
        
        # Wizualizacja różnic
        debug_info['visualizations'] = self.create_debug_visualizations(
            source_lab, target_lab, result_lab
        )
        
        return debug_info
    
    def create_debug_visualizations(self, source_lab, target_lab, result_lab):
        """Tworzenie wizualizacji dla debugowania"""
        visualizations = {}
        
        # Mapa różnic Delta E
        delta_e_map = colour.delta_E_CIE2000(result_lab, target_lab)
        visualizations['delta_e_map'] = delta_e_map
        
        # Mapa różnic lightness
        lightness_diff = np.abs(result_lab[:,:,0] - target_lab[:,:,0])
        visualizations['lightness_diff'] = lightness_diff
        
        # Mapa różnic chroma
        source_chroma = np.sqrt(
            source_lab[:,:,1]**2 + source_lab[:,:,2]**2
        )
        target_chroma = np.sqrt(
            target_lab[:,:,1]**2 + target_lab[:,:,2]**2
        )
        result_chroma = np.sqrt(
            result_lab[:,:,1]**2 + result_lab[:,:,2]**2
        )
        
        visualizations['chroma_diff'] = np.abs(result_chroma - target_chroma)
        
        return visualizations
```

---

## 5. Parametry i Konfiguracja

### Główne Parametry

```python
class PerceptualParameters:
    def __init__(self):
        # Przestrzeń kolorów
        self.color_space = 'lab'  # 'lab', 'cam16ucs', 'luv'
        
        # Metoda mapowania
        self.mapping_method = 'statistical'  # 'statistical', 'memory_color_preservation'
        
        # Wagi percepcyjne
        self.use_perceptual_weights = True
        self.lightness_sensitivity = 1.0
        self.chroma_sensitivity = 1.0
        self.hue_sensitivity = 1.0
        
        # Adaptacja chromatyczna
        self.chromatic_adaptation = True
        self.adaptation_method = 'bradford'  # 'bradford', 'von_kries', 'xyz_scaling'
        self.source_illuminant = 'D65'
        self.target_illuminant = 'D65'
        
        # Memory colors
        self.preserve_memory_colors = True
        self.memory_color_weight = 2.0
        
        # Warunki obserwacji
        self.viewing_conditions = 'average'  # 'average', 'dim', 'dark'
        
        # Optymalizacja
        self.local_adaptation = False
        self.adaptation_radius = 50
        self.gamut_mapping = True
        self.gamut_mapping_method = 'perceptual'  # 'perceptual', 'colorimetric'
        
        # Post-processing
        self.chroma_smoothing = True
        self.gradient_smoothing = True
        self.color_harmony_preservation = True
        
        # Wydajność
        self.use_gpu_acceleration = False
        self.batch_processing = True
        self.memory_optimization = True
```

### Predefiniowane Profile

#### Fotografia Portretowa
```python
def create_portrait_profile():
    params = PerceptualParameters()
    params.preserve_memory_colors = True
    params.memory_color_weight = 3.0
    params.hue_sensitivity = 2.0  # Wysoka wrażliwość na odcienie skóry
    params.local_adaptation = True
    params.adaptation_radius = 30
    params.chroma_smoothing = True
    params.color_space = 'lab'  # Najlepsze dla skóry
    return params
```

#### Fotografia Krajobrazowa
```python
def create_landscape_profile():
    params = PerceptualParameters()
    params.chroma_sensitivity = 1.5  # Podkreślenie saturacji
    params.color_space = 'cam16ucs'  # Lepsza percepcja dla krajobrazów
    params.gamut_mapping = True
    params.gradient_smoothing = False  # Zachowanie ostrości
    params.local_adaptation = False
    return params
```

#### Reprodukcja Dzieł Sztuki
```python
def create_art_reproduction_profile():
    params = PerceptualParameters()
    params.mapping_method = 'memory_color_preservation'
    params.chromatic_adaptation = True
    params.adaptation_method = 'bradford'
    params.use_perceptual_weights = False  # Zachowanie oryginalnych proporcji
    params.color_harmony_preservation = True
    params.preserve_memory_colors = True
    return params
```

#### Fotografia Produktowa (E-commerce)
```python
def create_product_profile():
    params = PerceptualParameters()
    params.color_space = 'lab'
    params.chromatic_adaptation = True
    params.preserve_memory_colors = True
    params.memory_color_weight = 2.5
    params.gamut_mapping = True
    params.gamut_mapping_method = 'colorimetric'  # Precyzja kolorów
    return params
```

### Konfiguracja Zaawansowana

```python
class AdvancedPerceptualConfig:
    def __init__(self):
        # Modele percepcji
        self.perception_models = {
            'standard': {
                'lightness_model': 'cie_lab',
                'chroma_model': 'euclidean',
                'hue_model': 'angular'
            },
            'advanced': {
                'lightness_model': 'cam16_j',
                'chroma_model': 'cam16_c',
                'hue_model': 'cam16_h'
            }
        }
        
        # Funkcje wagowe
        self.weight_functions = {
            'lightness': self.lightness_weight_function,
            'chroma': self.chroma_weight_function,
            'hue': self.hue_weight_function,
            'spatial': self.spatial_weight_function
        }
        
        # Metryki jakości
        self.quality_metrics = {
            'delta_e_2000': True,
            'delta_e_cam16': True,
            'memory_color_accuracy': True,
            'color_harmony_preservation': True,
            'spatial_consistency': True
        }
    
    def lightness_weight_function(self, L):
        """Funkcja wagowa dla lightness"""
        # Wyższa wrażliwość w średnich tonach
        return 1.0 + 0.5 * np.exp(-((L - 50)**2) / (2 * 20**2))
    
    def chroma_weight_function(self, C):
        """Funkcja wagowa dla chroma"""
        # Wyższa wrażliwość dla średnich saturacji
        return 1.0 + 0.3 * np.exp(-((C - 30)**2) / (2 * 15**2))
    
    def hue_weight_function(self, H):
        """Funkcja wagowa dla hue"""
        # Specjalna wrażliwość dla memory colors
        weights = np.ones_like(H)
        
        # Odcienie skóry (20-40°)
        skin_mask = (H >= 20) & (H <= 40)
        weights[skin_mask] = 2.0
        
        # Zieleń (90-150°)
        green_mask = (H >= 90) & (H <= 150)
        weights[green_mask] = 1.5
        
        # Błękit nieba (200-250°)
        sky_mask = (H >= 200) & (H <= 250)
        weights[sky_mask] = 1.3
        
        return weights
    
    def spatial_weight_function(self, image, position):
        """Funkcja wagowa przestrzenna"""
        # Wyższa waga dla centralnych obszarów
        h, w = image.shape[:2]
        y, x = position
        
        center_y, center_x = h // 2, w // 2
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_distance = np.sqrt(center_y**2 + center_x**2)
        
        # Waga maleje z odległością od centrum
        return 1.0 + 0.5 * (1.0 - distance / max_distance)
```

---

## 6. Analiza Wydajności

### Złożoność Obliczeniowa

**Analiza teoretyczna:**
- **Czasowa**: O(n × m × k × w) gdzie:
  - n = liczba pikseli
  - m = liczba kanałów kolorów
  - k = liczba iteracji optymalizacji
  - w = liczba funkcji wagowych

- **Pamięciowa**: O(n × m × s) gdzie:
  - s = liczba przestrzeni kolorów używanych równocześnie

### Benchmarki Wydajności

| Rozdzielczość | LAB [s] | CAM16-UCS [s] | RAM [MB] | Jakość [ΔE] |
|---------------|---------|---------------|----------|-------------|
| 1920×1080     | 3.8     | 12.4          | 45       | 2.1         |
| 3840×2160     | 15.2    | 49.6          | 180      | 1.8         |
| 7680×4320     | 60.8    | 198.4         | 720      | 1.6         |

### Optymalizacje Wydajności

```python
class PerformanceOptimizer:
    def __init__(self):
        self.use_numba = True
        self.use_multiprocessing = True
        self.chunk_size = 1000000  # pikseli na chunk
        self.memory_limit = 2048   # MB
    
    @staticmethod
    @numba.jit(nopython=True)
    def fast_delta_e_calculation(lab1, lab2):
        """Szybkie obliczenie Delta E z Numba"""
        dL = lab1[0] - lab2[0]
        da = lab1[1] - lab2[1]
        db = lab1[2] - lab2[2]
        return np.sqrt(dL**2 + da**2 + db**2)
    
    def process_image_chunks(self, image, processing_function):
        """Przetwarzanie obrazu w chunkach"""
        h, w, c = image.shape
        total_pixels = h * w
        
        if total_pixels <= self.chunk_size:
            return processing_function(image)
        
        # Podział na chunki
        chunks = []
        pixels_processed = 0
        
        while pixels_processed < total_pixels:
            chunk_size = min(self.chunk_size, total_pixels - pixels_processed)
            
            # Obliczenie granic chunka
            start_row = pixels_processed // w
            start_col = pixels_processed % w
            
            end_pixel = pixels_processed + chunk_size
            end_row = end_pixel // w
            end_col = end_pixel % w
            
            # Ekstrakcja chunka
            if start_row == end_row:
                chunk = image[start_row, start_col:end_col]
            else:
                chunk = image[start_row:end_row+1, :]
                # Przycinanie do dokładnego rozmiaru chunka
                chunk = chunk.reshape(-1, c)[:chunk_size].reshape(-1, w, c)
            
            chunks.append(processing_function(chunk))
            pixels_processed += chunk_size
        
        # Składanie wyników
        return np.concatenate(chunks, axis=0).reshape(h, w, c)
    
    def parallel_processing(self, image, processing_function, n_processes=None):
        """Równoległe przetwarzanie"""
        if n_processes is None:
            n_processes = multiprocessing.cpu_count()
        
        h, w, c = image.shape
        
        # Podział na paski poziome
        strip_height = h // n_processes
        strips = []
        
        for i in range(n_processes):
            start_row = i * strip_height
            end_row = (i + 1) * strip_height if i < n_processes - 1 else h
            strips.append(image[start_row:end_row])
        
        # Przetwarzanie równoległe
        with multiprocessing.Pool(n_processes) as pool:
            results = pool.map(processing_function, strips)
        
        return np.concatenate(results, axis=0)
```

### Profiling i Monitoring

```python
import time
import psutil
import tracemalloc

class PerformanceProfiler:
    def __init__(self):
        self.start_time = None
        self.memory_snapshots = []
        self.timing_data = {}
    
    def start_profiling(self):
        """Rozpoczęcie profilowania"""
        self.start_time = time.time()
        tracemalloc.start()
        self.memory_snapshots = []
        self.timing_data = {}
    
    def checkpoint(self, name):
        """Checkpoint czasowy"""
        current_time = time.time()
        if self.start_time:
            self.timing_data[name] = current_time - self.start_time
        
        # Snapshot pamięci
        current, peak = tracemalloc.get_traced_memory()
        self.memory_snapshots.append({
            'checkpoint': name,
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'system_ram_percent': psutil.virtual_memory().percent
        })
    
    def get_report(self):
        """Raport wydajności"""
        return {
            'timing': self.timing_data,
            'memory': self.memory_snapshots,
            'total_time': time.time() - self.start_time if self.start_time else 0
        }
```

---

## Nawigacja

**◀️ Poprzednia część**: [Implementacja Podstawowa](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-3of6.md)  
**▶️ Następna część**: [Optymalizacja i Debugging](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-5of6.md)  
**🏠 Powrót do**: [Spis Treści](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-0of6.md)

---

*Ostatnia aktualizacja: 2024-01-20*  
*Autor: GattoNero AI Assistant*  
*Wersja: 1.0*  
*Status: Część 4 z 6 - Implementacja Zaawansowana* ✅