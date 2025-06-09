# Weighted Histogram Matching - Część 1: Teoria i Podstawy

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 4-6 godzin | **Złożoność**: O(n log n)

---

## Przegląd Algorytmu

**Weighted Histogram Matching** to zaawansowana technika dopasowywania histogramów, która pozwala na selektywne przenoszenie charakterystyk kolorowych między obrazami z kontrolą nad siłą transferu w różnych zakresach tonalnych.

### Kluczowe Cechy
- 🎯 **Selektywny transfer**: Różne wagi dla różnych zakresów tonalnych
- 🔄 **Zachowanie struktury**: Lepsze zachowanie szczegółów niż standardowy histogram matching
- ⚖️ **Kontrola intensywności**: Precyzyjna kontrola siły transferu
- 🎨 **Wielokanałowość**: Niezależne przetwarzanie kanałów RGB/LAB

---

## Podstawy Teoretyczne

### 1. Standardowy Histogram Matching

Podstawowy histogram matching działa według wzoru:

\[
T(r) = G^{-1}[F(r)]
\]

Gdzie:
- \( F(r) \) - CDF (Cumulative Distribution Function) obrazu źródłowego
- \( G^{-1} \) - odwrotność CDF obrazu docelowego
- \( T(r) \) - funkcja transformacji

### 2. Weighted Histogram Matching

W wersji ważonej wprowadzamy funkcję wag \( W(r) \):

\[
T_{weighted}(r) = (1 - W(r)) \cdot r + W(r) \cdot T(r)
\]

Gdzie:
- \( W(r) \) - funkcja wag zależna od poziomu jasności
- \( r \) - oryginalny poziom piksela
- \( T(r) \) - standardowa transformacja histogram matching

### 3. Funkcje Wag

Możemy definiować różne funkcje wag:

#### Wagi Liniowe
\[
W_{linear}(r) = \alpha \cdot \frac{r}{255}
\]

#### Wagi Gaussowskie
\[
W_{gaussian}(r) = \alpha \cdot e^{-\frac{(r-\mu)^2}{2\sigma^2}}
\]

#### Wagi Segmentowe
\[
W_{segment}(r) = \begin{cases}
\alpha_{shadows} & \text{if } r < t_1 \\
\alpha_{midtones} & \text{if } t_1 \leq r < t_2 \\
\alpha_{highlights} & \text{if } r \geq t_2
\end{cases}
\]

---

## Matematyczne Podstawy

### Histogram i CDF

```python
def calculate_histogram_and_cdf(image_channel, bins=256):
    """
    Oblicza histogram i CDF dla kanału obrazu
    """
    # Histogram
    hist, bin_edges = np.histogram(image_channel.flatten(), 
                                  bins=bins, range=(0, 255), density=True)
    
    # CDF (Cumulative Distribution Function)
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]  # Normalizacja do [0, 1]
    
    return hist, cdf, bin_edges
```

### Interpolacja CDF

```python
def create_lookup_table(source_cdf, target_cdf, bins=256):
    """
    Tworzy lookup table dla histogram matching
    """
    # Wartości poziomów jasności
    levels = np.arange(bins)
    
    # Interpolacja odwrotnej CDF targetu
    lookup_table = np.interp(source_cdf, target_cdf, levels)
    
    return lookup_table.astype(np.uint8)
```

### Zastosowanie Wag

```python
def apply_weighted_transform(original, transformed, weights):
    """
    Stosuje ważoną transformację
    """
    # Weighted blending
    result = (1 - weights) * original + weights * transformed
    
    return np.clip(result, 0, 255).astype(np.uint8)
```

---

## Implementacja Podstawowa

### Klasa WeightedHistogramMatching

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage, interpolate
import cv2

class WeightedHistogramMatching:
    def __init__(self):
        self.name = "Weighted Histogram Matching"
        self.version = "1.0"
        
        # Parametry domyślne
        self.default_weights = {
            'shadows': 0.8,    # 0-85
            'midtones': 1.0,   # 85-170
            'highlights': 0.6  # 170-255
        }
        
        self.default_thresholds = {
            'shadow_threshold': 85,
            'highlight_threshold': 170
        }
    
    def calculate_histogram_stats(self, image_channel):
        """
        Oblicza statystyki histogramu dla kanału
        """
        hist, bin_edges = np.histogram(image_channel.flatten(), 
                                      bins=256, range=(0, 255), density=True)
        
        # CDF
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
        
        # Statystyki
        mean_val = np.mean(image_channel)
        std_val = np.std(image_channel)
        median_val = np.median(image_channel)
        
        # Percentyle
        p5 = np.percentile(image_channel, 5)
        p95 = np.percentile(image_channel, 95)
        
        return {
            'histogram': hist,
            'cdf': cdf,
            'bin_edges': bin_edges,
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'p5': p5,
            'p95': p95
        }
    
    def create_weight_function(self, weight_type='segmented', **params):
        """
        Tworzy funkcję wag dla poziomów jasności 0-255
        """
        levels = np.arange(256)
        
        if weight_type == 'segmented':
            return self._create_segmented_weights(levels, **params)
        elif weight_type == 'linear':
            return self._create_linear_weights(levels, **params)
        elif weight_type == 'gaussian':
            return self._create_gaussian_weights(levels, **params)
        elif weight_type == 'custom':
            return self._create_custom_weights(levels, **params)
        else:
            # Domyślne wagi jednostajne
            return np.ones(256) * params.get('strength', 1.0)
    
    def _create_segmented_weights(self, levels, **params):
        """
        Tworzy wagi segmentowe (shadows/midtones/highlights)
        """
        weights = np.zeros_like(levels, dtype=float)
        
        shadow_threshold = params.get('shadow_threshold', 85)
        highlight_threshold = params.get('highlight_threshold', 170)
        
        shadow_weight = params.get('shadow_weight', 0.8)
        midtone_weight = params.get('midtone_weight', 1.0)
        highlight_weight = params.get('highlight_weight', 0.6)
        
        # Smooth transitions
        transition_width = params.get('transition_width', 10)
        
        # Shadows
        shadow_mask = levels <= shadow_threshold
        weights[shadow_mask] = shadow_weight
        
        # Highlights
        highlight_mask = levels >= highlight_threshold
        weights[highlight_mask] = highlight_weight
        
        # Midtones
        midtone_mask = (levels > shadow_threshold) & (levels < highlight_threshold)
        weights[midtone_mask] = midtone_weight
        
        # Smooth transitions
        if transition_width > 0:
            weights = ndimage.gaussian_filter1d(weights, sigma=transition_width/3)
        
        return weights
    
    def _create_linear_weights(self, levels, **params):
        """
        Tworzy wagi liniowe
        """
        min_weight = params.get('min_weight', 0.0)
        max_weight = params.get('max_weight', 1.0)
        direction = params.get('direction', 'ascending')  # 'ascending' or 'descending'
        
        if direction == 'ascending':
            weights = min_weight + (max_weight - min_weight) * (levels / 255.0)
        else:
            weights = max_weight - (max_weight - min_weight) * (levels / 255.0)
        
        return weights
    
    def _create_gaussian_weights(self, levels, **params):
        """
        Tworzy wagi gaussowskie
        """
        center = params.get('center', 128)
        sigma = params.get('sigma', 50)
        amplitude = params.get('amplitude', 1.0)
        baseline = params.get('baseline', 0.0)
        
        weights = baseline + amplitude * np.exp(-((levels - center) ** 2) / (2 * sigma ** 2))
        
        return np.clip(weights, 0, 1)
    
    def _create_custom_weights(self, levels, **params):
        """
        Tworzy niestandardowe wagi z punktów kontrolnych
        """
        control_points = params.get('control_points', [(0, 0.5), (128, 1.0), (255, 0.5)])
        interpolation_method = params.get('interpolation', 'linear')  # 'linear', 'cubic'
        
        # Wyciągnij x i y z punktów kontrolnych
        x_points = [p[0] for p in control_points]
        y_points = [p[1] for p in control_points]
        
        # Interpolacja
        if interpolation_method == 'cubic' and len(control_points) >= 4:
            f = interpolate.interp1d(x_points, y_points, kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')
        else:
            f = interpolate.interp1d(x_points, y_points, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
        
        weights = f(levels)
        
        return np.clip(weights, 0, 1)
    
    def standard_histogram_matching(self, source_channel, target_channel):
        """
        Standardowy histogram matching dla pojedynczego kanału
        """
        # Oblicz histogramy i CDF
        source_stats = self.calculate_histogram_stats(source_channel)
        target_stats = self.calculate_histogram_stats(target_channel)
        
        source_cdf = source_stats['cdf']
        target_cdf = target_stats['cdf']
        
        # Utwórz lookup table
        levels = np.arange(256)
        lookup_table = np.interp(source_cdf, target_cdf, levels)
        
        # Zastosuj transformację
        source_flat = source_channel.flatten()
        transformed_flat = lookup_table[source_flat]
        
        return transformed_flat.reshape(source_channel.shape).astype(np.uint8)
    
    def weighted_histogram_matching(self, source_channel, target_channel, 
                                  weight_function=None, **weight_params):
        """
        Weighted histogram matching dla pojedynczego kanału
        """
        # Standardowy histogram matching
        transformed = self.standard_histogram_matching(source_channel, target_channel)
        
        # Utwórz funkcję wag jeśli nie podano
        if weight_function is None:
            weight_function = self.create_weight_function(**weight_params)
        
        # Zastosuj wagi
        source_flat = source_channel.flatten()
        transformed_flat = transformed.flatten()
        
        # Pobierz wagi dla każdego piksela
        pixel_weights = weight_function[source_flat]
        
        # Weighted blending
        result_flat = ((1 - pixel_weights) * source_flat + 
                      pixel_weights * transformed_flat)
        
        result = result_flat.reshape(source_channel.shape)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def process_rgb_image(self, source_image, target_image, 
                         weight_config=None, process_channels='RGB'):
        """
        Przetwarza obraz RGB z weighted histogram matching
        """
        # Konwertuj do numpy arrays
        if isinstance(source_image, Image.Image):
            source_array = np.array(source_image)
        else:
            source_array = source_image
        
        if isinstance(target_image, Image.Image):
            target_array = np.array(target_image)
        else:
            target_array = target_image
        
        # Domyślna konfiguracja wag
        if weight_config is None:
            weight_config = {
                'weight_type': 'segmented',
                'shadow_weight': 0.8,
                'midtone_weight': 1.0,
                'highlight_weight': 0.6
            }
        
        result_array = source_array.copy()
        
        # Przetwórz wybrane kanały
        channels_to_process = []
        if 'R' in process_channels:
            channels_to_process.append(0)
        if 'G' in process_channels:
            channels_to_process.append(1)
        if 'B' in process_channels:
            channels_to_process.append(2)
        
        for channel_idx in channels_to_process:
            source_channel = source_array[:, :, channel_idx]
            target_channel = target_array[:, :, channel_idx]
            
            # Weighted histogram matching
            result_channel = self.weighted_histogram_matching(
                source_channel, target_channel, **weight_config
            )
            
            result_array[:, :, channel_idx] = result_channel
        
        return result_array
```

---

## Analiza Funkcji Wag

### Wizualizacja Różnych Typów Wag

```python
def visualize_weight_functions():
    """
    Wizualizuje różne typy funkcji wag
    """
    matcher = WeightedHistogramMatching()
    levels = np.arange(256)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Wagi segmentowe
    plt.subplot(2, 3, 1)
    weights_seg = matcher.create_weight_function(
        weight_type='segmented',
        shadow_weight=0.8,
        midtone_weight=1.0,
        highlight_weight=0.6
    )
    plt.plot(levels, weights_seg, 'b-', linewidth=2)
    plt.title('Segmented Weights')
    plt.xlabel('Pixel Level')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    
    # 2. Wagi liniowe rosnące
    plt.subplot(2, 3, 2)
    weights_lin_asc = matcher.create_weight_function(
        weight_type='linear',
        min_weight=0.2,
        max_weight=1.0,
        direction='ascending'
    )
    plt.plot(levels, weights_lin_asc, 'g-', linewidth=2)
    plt.title('Linear Ascending Weights')
    plt.xlabel('Pixel Level')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    
    # 3. Wagi liniowe malejące
    plt.subplot(2, 3, 3)
    weights_lin_desc = matcher.create_weight_function(
        weight_type='linear',
        min_weight=0.2,
        max_weight=1.0,
        direction='descending'
    )
    plt.plot(levels, weights_lin_desc, 'r-', linewidth=2)
    plt.title('Linear Descending Weights')
    plt.xlabel('Pixel Level')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    
    # 4. Wagi gaussowskie - środek
    plt.subplot(2, 3, 4)
    weights_gauss_mid = matcher.create_weight_function(
        weight_type='gaussian',
        center=128,
        sigma=40,
        amplitude=1.0,
        baseline=0.1
    )
    plt.plot(levels, weights_gauss_mid, 'm-', linewidth=2)
    plt.title('Gaussian Weights (Midtones)')
    plt.xlabel('Pixel Level')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    
    # 5. Wagi gaussowskie - highlights
    plt.subplot(2, 3, 5)
    weights_gauss_high = matcher.create_weight_function(
        weight_type='gaussian',
        center=200,
        sigma=30,
        amplitude=1.0,
        baseline=0.2
    )
    plt.plot(levels, weights_gauss_high, 'c-', linewidth=2)
    plt.title('Gaussian Weights (Highlights)')
    plt.xlabel('Pixel Level')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    
    # 6. Wagi niestandardowe
    plt.subplot(2, 3, 6)
    weights_custom = matcher.create_weight_function(
        weight_type='custom',
        control_points=[(0, 0.3), (64, 0.8), (128, 0.5), (192, 0.9), (255, 0.4)],
        interpolation='cubic'
    )
    plt.plot(levels, weights_custom, 'orange', linewidth=2)
    plt.title('Custom Weights (Cubic)')
    plt.xlabel('Pixel Level')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weight_functions_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Uruchom wizualizację
if __name__ == '__main__':
    visualize_weight_functions()
```

---

## Porównanie z Innymi Metodami

### Tabela Porównawcza

| Metoda | Kontrola | Zachowanie Szczegółów | Złożoność | Przypadki Użycia |
|--------|----------|----------------------|-----------|------------------|
| **Standard Histogram Matching** | Niska | Średnie | O(n) | Globalna korekta |
| **Weighted Histogram Matching** | Wysoka | Bardzo dobre | O(n log n) | Selektywna korekta |
| **LAB Color Transfer** | Średnia | Dobre | O(n) | Transfer kolorów |
| **Palette Mapping** | Niska | Słabe | O(n) | Stylizacja |

### Zalety Weighted Histogram Matching

✅ **Zalety**:
- Precyzyjna kontrola nad transferem w różnych zakresach tonalnych
- Zachowanie struktury i szczegółów obrazu
- Elastyczność w definiowaniu funkcji wag
- Możliwość selektywnego przetwarzania kanałów
- Dobra wydajność dla średnich obrazów

⚠️ **Ograniczenia**:
- Wyższa złożoność obliczeniowa niż podstawowy histogram matching
- Wymaga dobrego zrozumienia funkcji wag
- Może wprowadzać artefakty przy źle dobranych wagach
- Nie zawsze zachowuje naturalne przejścia kolorów

---

## Przykłady Zastosowań

### 1. Korekcja Ekspozycji

```python
def exposure_correction_example():
    """
    Przykład korekcji ekspozycji z zachowaniem szczegółów
    """
    matcher = WeightedHistogramMatching()
    
    # Konfiguracja dla korekcji ekspozycji
    exposure_config = {
        'weight_type': 'segmented',
        'shadow_weight': 1.0,      # Pełna korekta cieni
        'midtone_weight': 0.8,     # Umiarkowana korekta średnich tonów
        'highlight_weight': 0.3,   # Delikatna korekta świateł
        'transition_width': 15     # Płynne przejścia
    }
    
    # Symulacja - w rzeczywistości wczytaj obrazy
    underexposed_image = create_test_image('underexposed')
    reference_image = create_test_image('well_exposed')
    
    # Zastosuj korekcję
    corrected_image = matcher.process_rgb_image(
        underexposed_image, reference_image, 
        weight_config=exposure_config
    )
    
    return corrected_image
```

### 2. Selektywna Korekta Kolorów

```python
def selective_color_correction():
    """
    Przykład selektywnej korekcji kolorów
    """
    matcher = WeightedHistogramMatching()
    
    # Konfiguracja dla korekcji tylko niebieskiego kanału
    blue_correction_config = {
        'weight_type': 'gaussian',
        'center': 180,           # Skupienie na jasnych niebieskich
        'sigma': 40,
        'amplitude': 1.0,
        'baseline': 0.1
    }
    
    # Symulacja obrazów
    source_image = create_test_image('landscape')
    target_image = create_test_image('blue_sky')
    
    # Korekcja tylko kanału niebieskiego
    corrected_image = matcher.process_rgb_image(
        source_image, target_image,
        weight_config=blue_correction_config,
        process_channels='B'  # Tylko niebieski kanał
    )
    
    return corrected_image
```

---

## Podsumowanie Części 1

W tej części omówiliśmy:

1. **Podstawy teoretyczne** Weighted Histogram Matching
2. **Matematyczne fundamenty** algorytmu
3. **Implementację podstawową** klasy WeightedHistogramMatching
4. **Różne typy funkcji wag** i ich zastosowania
5. **Porównanie z innymi metodami** transferu kolorów
6. **Przykłady zastosowań** praktycznych

### Co dalej?

**Część 2** będzie zawierać:
- Zaawansowane techniki optymalizacji
- Implementację dla różnych przestrzeni kolorów
- Adaptacyjne funkcje wag
- Lokalne histogram matching
- Integrację z maskami i ROI

**Część 3** będzie zawierać:
- Testy i benchmarki wydajności
- Przypadki użycia i przykłady
- Rozwiązywanie problemów
- Integrację z głównym systemem

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Część 1 - Teoria i podstawy