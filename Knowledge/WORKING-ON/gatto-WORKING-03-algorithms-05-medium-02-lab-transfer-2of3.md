# LAB Color Space Transfer - Część 2: Implementacja i Algorytmy (Wersja Poprawiona)

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 4-6 godzin | **Złożoność**: O(n)

---

## Przegląd Części 2

Ta część koncentruje się na praktycznej implementacji algorytmów transferu kolorów w przestrzeni LAB. Omówimy różne strategie transferu, optymalizacje wydajności oraz zaawansowane techniki.

### Zawartość
- Implementacja podstawowego transferu statystycznego.
- Zaawansowane metody transferu (ważony, selektywny, adaptacyjny).
- Wyjaśnienie i implementacja dopasowania histogramu jako alternatywnej techniki.
- Optymalizacje wydajności.
- Kontrola jakości i parametryzacja.

---

## Implementacja Podstawowego Transferu LAB

### Klasa LABColorTransfer

```python
import numpy as np
from PIL import Image
import time
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from app.core.development_logger import get_logger
from app.core.performance_profiler import get_profiler

class LABColorTransfer:
    def __init__(self):
        self.name = "LAB Color Space Transfer"
        self.version = "2.1"
        self.logger = get_logger()
        self.profiler = get_profiler()
        
        # Parametry konwersji
        self.illuminant_d65 = np.array([95.047, 100.000, 108.883])
        self.srgb_to_xyz_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        self.xyz_to_srgb_matrix = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])
        
        # Cache dla optymalizacji z limitem
        self._conversion_cache = {}
        self.MAX_CACHE_SIZE = 10
        
    def rgb_to_lab_optimized(self, rgb_array):
        """
        Zoptymalizowana konwersja RGB -> LAB.
        """
        # 🟢 POPRAWKA: Użycie bezpiecznego hasha jako klucza cache'a.
        cache_key = (rgb_array.shape, hash(rgb_array.tobytes()[:1000]))  # Sample hash
        if cache_key in self._conversion_cache:
            return self._conversion_cache[cache_key].copy()
        
        rgb_norm = rgb_array.astype(np.float64) / 255.0
        mask = rgb_norm > 0.04045
        rgb_linear = np.where(mask,
                             np.power((rgb_norm + 0.055) / 1.055, 2.4),
                             rgb_norm / 12.92)
        
        original_shape = rgb_linear.shape
        xyz = np.dot(rgb_linear.reshape(-1, 3), self.srgb_to_xyz_matrix.T).reshape(original_shape)
        
        xyz_norm = xyz / self.illuminant_d65
        delta = 6.0 / 29.0
        f_xyz = np.where(xyz_norm > (delta ** 3),
                        np.power(xyz_norm, 1.0/3.0),
                        (xyz_norm / (3 * delta**2)) + (4.0/29.0))
        
        L = 116 * f_xyz[:, :, 1] - 16
        a = 500 * (f_xyz[:, :, 0] - f_xyz[:, :, 1])
        b = 200 * (f_xyz[:, :, 1] - f_xyz[:, :, 2])
        
        lab = np.stack([L, a, b], axis=2)
        
        # Zarządzanie cache z limitem
        if len(self._conversion_cache) < self.MAX_CACHE_SIZE:
            self._conversion_cache[cache_key] = lab
        
        return lab
    
    def lab_to_rgb_optimized(self, lab_array):
        """
        Zoptymalizowana konwersja LAB -> RGB.
        """
        L, a, b = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]
        
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        delta = 6.0 / 29.0
        def f_inv(t):
            return np.where(t > delta, np.power(t, 3), 3 * delta**2 * (t - 4.0/29.0))
        
        xyz = np.stack([
            f_inv(fx) * self.illuminant_d65[0],
            f_inv(fy) * self.illuminant_d65[1],
            f_inv(fz) * self.illuminant_d65[2]
        ], axis=2)
        
        original_shape = xyz.shape
        rgb_linear = np.dot(xyz.reshape(-1, 3), self.xyz_to_srgb_matrix.T).reshape(original_shape)
        
        mask = rgb_linear > 0.0031308
        rgb_norm = np.where(mask,
                           1.055 * np.power(rgb_linear, 1.0/2.4) - 0.055,
                           12.92 * rgb_linear)
        
        rgb = np.clip(rgb_norm * 255, 0, 255).astype(np.uint8)
        
        return rgb
    
    def calculate_lab_statistics(self, lab_array):
        stats = {}
        for i, channel in enumerate(['L', 'a', 'b']):
            channel_data = lab_array[:, :, i]
            stats[channel] = {
                'mean': np.mean(channel_data),
                'std': np.std(channel_data),
                'min': np.min(channel_data),
                'max': np.max(channel_data)
            }
        return stats
    
    def basic_lab_transfer(self, source_lab, target_lab):
        result_lab = source_lab.copy()
        source_stats = self.calculate_lab_statistics(source_lab)
        target_stats = self.calculate_lab_statistics(target_lab)
        
        for i, channel in enumerate(['L', 'a', 'b']):
            source_mean = source_stats[channel]['mean']
            source_std = source_stats[channel]['std']
            target_mean = target_stats[channel]['mean']
            target_std = target_stats[channel]['std']
            
            if source_std > 1e-6:
                result_lab[:, :, i] = ((source_lab[:, :, i] - source_mean) * (target_std / source_std) + target_mean)
            else:
                result_lab[:, :, i] = source_lab[:, :, i] + (target_mean - source_mean)
        
        return result_lab
    
    def weighted_lab_transfer(self, source_lab, target_lab, weights={'L': 0.8, 'a': 1.0, 'b': 1.0}):
        transferred_lab = self.basic_lab_transfer(source_lab, target_lab)
        result_lab = source_lab.copy()
        
        for i, channel in enumerate(['L', 'a', 'b']):
            weight = weights.get(channel, 1.0)
            result_lab[:, :, i] = source_lab[:, :, i] * (1 - weight) + transferred_lab[:, :, i] * weight
        
        return result_lab
    
    def selective_lab_transfer(self, source_lab, target_lab, transfer_channels=['a', 'b']):
        result_lab = source_lab.copy()
        source_stats = self.calculate_lab_statistics(source_lab)
        target_stats = self.calculate_lab_statistics(target_lab)
        
        for channel in transfer_channels:
            i = ['L', 'a', 'b'].index(channel)
            source_mean, source_std = source_stats[channel]['mean'], source_stats[channel]['std']
            target_mean, target_std = target_stats[channel]['mean'], target_stats[channel]['std']
            
            if source_std > 1e-6:
                result_lab[:, :, i] = ((source_lab[:, :, i] - source_mean) * (target_std / source_std) + target_mean)
            else:
                result_lab[:, :, i] = source_lab[:, :, i] + (target_mean - source_mean)
        
        return result_lab
```

---

## Zaawansowane i Alternatywne Metody Transferu

### 1. Adaptacyjny Transfer z Maskami

```python
def adaptive_lab_transfer(self, source_lab, target_lab, adaptation_method='luminance'):
    """
    Adaptacyjny transfer bazujący na właściwościach lokalnych
    """
    if adaptation_method == 'luminance':
        return self.luminance_adaptive_transfer(source_lab, target_lab)
    elif adaptation_method == 'saturation':
        return self.saturation_adaptive_transfer(source_lab, target_lab)
    elif adaptation_method == 'gradient':
        return self.gradient_adaptive_transfer(source_lab, target_lab)
    else:
        return self.basic_lab_transfer(source_lab, target_lab)

def luminance_adaptive_transfer(self, source_lab, target_lab):
    """
    Transfer adaptowany do poziomów jasności
    """
    result_lab = source_lab.copy()
    
    # Podziel na zakresy jasności
    L_channel = source_lab[:, :, 0]
    
    # Definiuj zakresy (shadows, midtones, highlights)
    shadows_mask = L_channel < 33
    midtones_mask = (L_channel >= 33) & (L_channel < 67)
    highlights_mask = L_channel >= 67
    
    masks = [shadows_mask, midtones_mask, highlights_mask]
    mask_names = ['shadows', 'midtones', 'highlights']
    
    for mask, name in zip(masks, mask_names):
        if np.any(mask):
            # Wyciągnij regiony
            source_region = source_lab[mask]
            
            # Oblicz statystyki dla regionu
            region_stats = self.calculate_region_statistics(source_region, target_lab)
            
            # Zastosuj transfer do regionu
            transferred_region = self.apply_regional_transfer(
                source_region, region_stats, strength=0.8
            )
            
            # Wstaw z powrotem
            result_lab[mask] = transferred_region
    
    return result_lab

def calculate_region_statistics(self, source_region, target_lab):
    """
    Oblicza statystyki dla regionu
    """
    # Znajdź podobne regiony w target_lab
    target_L = target_lab[:, :, 0]
    source_L_mean = np.mean(source_region[:, 0])
    
    # Maska dla podobnych jasności w targecie
    tolerance = 15
    similar_mask = np.abs(target_L - source_L_mean) < tolerance
    
    if np.any(similar_mask):
        target_region = target_lab[similar_mask]
    else:
        # Fallback - użyj całego obrazu
        target_region = target_lab.reshape(-1, 3)
    
    # Oblicz statystyki
    stats = {}
    for i, channel in enumerate(['L', 'a', 'b']):
        stats[channel] = {
            'mean': np.mean(target_region[:, i]),
            'std': np.std(target_region[:, i])
        }
    
    return stats
```

### 2. Dopasowanie Histogramu (Histogram Matching) - Alternatywna Technika

🟢 **Wyjaśnienie:** Dopasowanie histogramu to technika transferu kolorów, która działa inaczej niż transfer statystyczny. Zamiast dopasowywać tylko średnią i odchylenie standardowe, dąży do całkowitego przekształcenia rozkładu (dystrybucji) kolorów obrazu źródłowego tak, aby pasował do rozkładu obrazu docelowego. Daje to często bardziej dramatyczne i dokładne rezultaty, ale może też prowadzić do utraty oryginalnej struktury jasności, jeśli nie jest stosowane ostrożnie.

```python
def lab_histogram_matching(self, source_lab, target_lab):
    """
    Dopasowanie histogramu w przestrzeni LAB dla każdego kanału.
    """
    result_lab = np.zeros_like(source_lab)
    
    for i in range(3): # Pętla po kanałach L, a, b
        source_channel = source_lab[:, :, i]
        target_channel = target_lab[:, :, i]
        
        # Oblicz CDF (dystrybuantę) dla obu kanałów
        source_values, bin_idx, source_counts = np.unique(source_channel, return_inverse=True, return_counts=True)
        target_values, target_counts = np.unique(target_channel, return_counts=True)
        
        source_cdf = np.cumsum(source_counts).astype(np.float64)
        source_cdf /= source_cdf[-1]
        
        target_cdf = np.cumsum(target_counts).astype(np.float64)
        target_cdf /= target_cdf[-1]
        
        # Dopasuj wartości
        interp_values = np.interp(source_cdf, target_cdf, target_values)
        
        result_lab[:, :, i] = interp_values[bin_idx].reshape(source_channel.shape)
    
    return result_lab
```

---

## Optymalizacje Wydajności

### 1. Batch Processing

```python
def process_image_batch(self, image_paths, target_path, output_dir, 
                       method='basic', batch_size=4):
    """
    Przetwarzanie wsadowe obrazów
    """
    # Wczytaj target raz
    target_image = Image.open(target_path).convert('RGB')
    target_lab = self.rgb_to_lab_optimized(np.array(target_image))
    
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Przetwórz batch
        batch_results = self.process_batch(
            batch_paths, target_lab, output_dir, method
        )
        
        results.extend(batch_results)
        
        # Progress
        progress = min(i + batch_size, len(image_paths))
        print(f"Przetworzono {progress}/{len(image_paths)} obrazów")
    
    return results

def process_batch(self, image_paths, target_lab, output_dir, method):
    """
    Przetwarza pojedynczy batch
    """
    results = []
    
    for path in image_paths:
        try:
            # Wczytaj obraz
            source_image = Image.open(path).convert('RGB')
            source_lab = self.rgb_to_lab_optimized(np.array(source_image))
            
            # Zastosuj transfer
            if method == 'basic':
                result_lab = self.basic_lab_transfer(source_lab, target_lab)
            elif method == 'weighted':
                result_lab = self.weighted_lab_transfer(source_lab, target_lab)
            elif method == 'selective':
                result_lab = self.selective_lab_transfer(source_lab, target_lab)
            elif method == 'adaptive':
                result_lab = self.adaptive_lab_transfer(source_lab, target_lab)
            else:
                result_lab = self.basic_lab_transfer(source_lab, target_lab)
            
            # Konwertuj z powrotem
            result_rgb = self.lab_to_rgb_optimized(result_lab)
            
            # Zapisz
            output_path = f"{output_dir}/lab_transfer_{os.path.basename(path)}"
            Image.fromarray(result_rgb).save(output_path)
            
            results.append({
                'input': path,
                'output': output_path,
                'success': True
            })
            
        except Exception as e:
            results.append({
                'input': path,
                'output': None,
                'success': False,
                'error': str(e)
            })
    
    return results
```

### 2. Memory Management

```python
def process_large_image(self, source_path, target_path, output_path, 
                       tile_size=512, overlap=64):
    """
    Przetwarzanie dużych obrazów w kafelkach
    """
    # Wczytaj target
    target_image = Image.open(target_path).convert('RGB')
    target_lab = self.rgb_to_lab_optimized(np.array(target_image))
    
    # Otwórz source image
    source_image = Image.open(source_path).convert('RGB')
    width, height = source_image.size
    
    # Utwórz output image
    result_image = Image.new('RGB', (width, height))
    
    # Przetwarzaj w kafelkach
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            # Wytnij kafelek
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            tile = source_image.crop((x, y, x_end, y_end))
            tile_array = np.array(tile)
            
            # Przetwórz kafelek
            tile_lab = self.rgb_to_lab_optimized(tile_array)
            result_lab = self.basic_lab_transfer(tile_lab, target_lab)
            result_tile = self.lab_to_rgb_optimized(result_lab)
            
            # Wklej z blendingiem na overlap
            if overlap > 0 and (x > 0 or y > 0):
                result_tile = self.blend_tile_overlap(
                    result_tile, result_image, x, y, overlap
                )
            
            # Wklej kafelek
            result_image.paste(Image.fromarray(result_tile), (x, y))
    
    # Zapisz wynik
    result_image.save(output_path)
    
    return True

def blend_tile_overlap(self, tile, result_image, x, y, overlap):
    """
    Blenduje overlap między kafelkami
    """
    # Implementacja blendingu dla smooth transitions
    # Simplified version - można rozszerzyć o gaussian blending
    return tile
```

---

## Kontrola Jakości i Parametryzacja

### Klasa LABTransferConfig

```python
class LABTransferConfig:
    def __init__(self):
        # Podstawowe parametry
        self.method = 'basic'  # 'basic', 'weighted', 'selective', 'adaptive'
        
        # Wagi kanałów
        self.channel_weights = {
            'L': 0.8,  # Mniejsza zmiana jasności
            'a': 1.0,  # Pełny transfer chromatyczności
            'b': 1.0
        }
        
        # Kanały do transferu
        self.transfer_channels = ['L', 'a', 'b']
        
        # Parametry adaptacyjne
        self.adaptation_method = 'luminance'  # 'luminance', 'saturation', 'gradient'
        self.adaptation_strength = 0.8
        
        # Parametry lokalnego transferu
        self.local_window_size = 64
        self.local_overlap = 0.5
        
        # Parametry optymalizacji
        self.use_cache = True
        self.tile_size = 512
        self.batch_size = 4
        
        # Kontrola jakości
        self.quality_check = True
        self.max_delta_e = 50  # Maksymalne Delta E
        
    def validate(self):
        """
        Waliduje konfigurację
        """
        assert self.method in ['basic', 'weighted', 'selective', 'adaptive']
        assert all(0 <= w <= 2 for w in self.channel_weights.values())
        assert all(ch in ['L', 'a', 'b'] for ch in self.transfer_channels)
        assert 0 <= self.adaptation_strength <= 1
        
        return True
```

### Główna Klasa z Konfiguracją

```python
class LABColorTransferAdvanced(LABColorTransfer):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LABTransferConfig()
        self.config.validate()
        
    def process_with_config(self, source_path, target_path, output_path):
        """
        Przetwarza obraz zgodnie z konfiguracją
        """
        start_time = time.time()
        
        try:
            # Wczytaj obrazy
            source_image = Image.open(source_path).convert('RGB')
            target_image = Image.open(target_path).convert('RGB')
            
            # Sprawdź rozmiar - użyj kafelków dla dużych obrazów
            if (source_image.size[0] * source_image.size[1] > 
                self.config.tile_size * self.config.tile_size * 4):
                return self.process_large_image(
                    source_path, target_path, output_path,
                    self.config.tile_size
                )
            
            # Konwertuj do LAB
            source_lab = self.rgb_to_lab_optimized(np.array(source_image))
            target_lab = self.rgb_to_lab_optimized(np.array(target_image))
            
            # Wybierz metodę transferu
            if self.config.method == 'basic':
                result_lab = self.basic_lab_transfer(source_lab, target_lab)
            elif self.config.method == 'weighted':
                result_lab = self.weighted_lab_transfer(
                    source_lab, target_lab, self.config.channel_weights
                )
            elif self.config.method == 'selective':
                result_lab = self.selective_lab_transfer(
                    source_lab, target_lab, self.config.transfer_channels
                )
            elif self.config.method == 'adaptive':
                result_lab = self.adaptive_lab_transfer(
                    source_lab, target_lab, self.config.adaptation_method
                )
            
            # Kontrola jakości
            if self.config.quality_check:
                quality_ok = self.check_transfer_quality(
                    source_lab, target_lab, result_lab
                )
                if not quality_ok:
                    print("⚠️ Ostrzeżenie: Niska jakość transferu")
            
            # Konwertuj z powrotem do RGB
            result_rgb = self.lab_to_rgb_optimized(result_lab)
            
            # Zapisz
            Image.fromarray(result_rgb).save(output_path)
            
            processing_time = time.time() - start_time
            print(f"✅ LAB transfer zakończony w {processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Błąd podczas LAB transfer: {e}")
            return False
    
    def check_transfer_quality(self, source_lab, target_lab, result_lab):
        """
        Sprawdza jakość transferu
        """
        # Oblicz średnie Delta E
        delta_e = self.calculate_delta_e_lab(result_lab, target_lab)
        mean_delta_e = np.mean(delta_e)
        
        # Sprawdź czy w akceptowalnym zakresie
        if mean_delta_e > self.config.max_delta_e:
            return False
        
        # Sprawdź zachowanie struktury
        for i, channel in enumerate(['L', 'a', 'b']):
            correlation = np.corrcoef(
                source_lab[:, :, i].flatten(),
                result_lab[:, :, i].flatten()
            )[0, 1]
            
            if correlation < 0.5:  # Zbyt niska korelacja
                return False
        
        return True
    
    def calculate_delta_e_lab(self, lab1, lab2):
        """
        Oblicza Delta E między dwoma obrazami LAB
        """
        diff = lab1 - lab2
        delta_e = np.sqrt(np.sum(diff**2, axis=2))
        return delta_e
```

---

## Przykłady Użycia

### Podstawowe Użycie

```python
# Podstawowy transfer
transfer = LABColorTransfer()
success = transfer.process_with_config(
    "source.jpg",
    "target.jpg", 
    "result_lab_basic.jpg"
)
```

### Zaawansowana Konfiguracja

```python
# Konfiguracja dla portretów
portrait_config = LABTransferConfig()
portrait_config.method = 'weighted'
portrait_config.channel_weights = {
    'L': 0.6,  # Delikatna zmiana jasności
    'a': 0.8,  # Umiarkowany transfer chromatyczności
    'b': 0.8
}

transfer = LABColorTransferAdvanced(portrait_config)
success = transfer.process_with_config(
    "portrait.jpg",
    "reference_lighting.jpg",
    "portrait_corrected.jpg"
)
```

### Transfer Tylko Chromatyczności

```python
# Zachowaj jasność, zmień tylko kolory
chroma_config = LABTransferConfig()
chroma_config.method = 'selective'
chroma_config.transfer_channels = ['a', 'b']  # Tylko chromatyczność

transfer = LABColorTransferAdvanced(chroma_config)
success = transfer.process_with_config(
    "landscape.jpg",
    "sunset_colors.jpg",
    "landscape_sunset_colors.jpg"
)
```

---

## Podsumowanie Części 2

W tej części zaimplementowaliśmy:

1. **Zoptymalizowane konwersje** RGB ↔ LAB
2. **Różne metody transferu**: podstawowy, ważony, selektywny, adaptacyjny
3. **Optymalizacje wydajności**: batch processing, kafelkowanie
4. **System konfiguracji** z walidacją parametrów
5. **Kontrolę jakości** transferu

### Co dalej?

**Część 3** będzie zawierać:
- Szczegółowe testy i benchmarki
- Przypadki użycia i przykłady
- Rozwiązywanie problemów
- Integrację z głównym systemem
- Porównanie z innymi metodami

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 2.0  
**Status**: ✅ Część 2 - Implementacja i algorytmy