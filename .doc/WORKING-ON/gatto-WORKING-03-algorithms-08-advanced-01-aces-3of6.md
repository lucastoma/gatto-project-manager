# ACES Color Space Transfer - Część 3of6: Implementacja Core 🔧

> **Seria:** ACES Color Space Transfer  
> **Część:** 3 z 6 - Implementacja Core  
> **Wymagania:** [2of6 - Pseudokod i Architektura](gatto-WORKING-03-algorithms-08-advanced-01-aces-2of6.md)  
> **Następna część:** [4of6 - Parametry i Konfiguracja](gatto-WORKING-03-algorithms-08-advanced-01-aces-4of6.md)

---

## 1. Struktura Modułów

```python
# Struktura plików
aces_color_transfer/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── aces_transform.py      # Główna klasa transformacji
│   ├── color_conversion.py    # Konwersje przestrzeni kolorów
│   ├── statistics.py          # Analiza statystyk obrazu
│   └── tone_mapping.py        # Algorytmy tone mappingu
├── utils/
│   ├── __init__.py
│   ├── validation.py          # Walidacja danych wejściowych
│   ├── matrices.py            # Macierze transformacji
│   └── helpers.py             # Funkcje pomocnicze
├── analysis/
│   ├── __init__.py
│   ├── quality.py             # Ocena jakości
│   └── performance.py         # Monitoring wydajności
└── examples/
    ├── __init__.py
    ├── basic_usage.py
    └── advanced_usage.py
```

---

## 2. Główna Klasa ACESColorTransfer

### 2.1 Definicja Klasy

```python
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import cv2
from pathlib import Path

class TransformMethod(Enum):
    """Dostępne metody transformacji ACES."""
    CHROMATIC_ADAPTATION = "chromatic_adaptation"
    STATISTICAL_MATCHING = "statistical_matching"
    HISTOGRAM_MATCHING = "histogram_matching"
    PERCEPTUAL_MATCHING = "perceptual_matching"
    HYBRID = "hybrid"

@dataclass
class ACESParameters:
    """Parametry konfiguracyjne dla transformacji ACES."""
    
    # Metoda transformacji
    method: TransformMethod = TransformMethod.STATISTICAL_MATCHING
    
    # Parametry tone mappingu
    use_tone_mapping: bool = True
    tone_curve_a: float = 2.51
    tone_curve_b: float = 0.03
    tone_curve_c: float = 2.43
    tone_curve_d: float = 0.59
    tone_curve_e: float = 0.14
    
    # Parametry luminancji
    preserve_luminance: bool = True
    luminance_weight: float = 0.8
    
    # Parametry gamut
    gamut_compression: bool = True
    compression_strength: float = 0.7
    
    # Parametry jakości
    min_confidence: float = 0.7
    quality_threshold: float = 0.8
    
    # Parametry wydajności
    chunk_size: int = 1024
    use_parallel: bool = True
    num_threads: int = 4
    
    # Profil wyjściowy
    output_profile: str = "sRGB"
    output_gamma: float = 2.2

class ACESColorTransfer:
    """Główna klasa do transferu kolorów w przestrzeni ACES."""
    
    def __init__(self, parameters: Optional[ACESParameters] = None):
        """Inicjalizacja z opcjonalnymi parametrami."""
        self.params = parameters or ACESParameters()
        self._initialize_matrices()
        self._performance_stats = {}
    
    def _initialize_matrices(self) -> None:
        """Inicjalizacja macierzy transformacji."""
        # Macierz sRGB -> ACES AP0
        self.srgb_to_aces_matrix = np.array([
            [0.4397010, 0.3829780, 0.1773350],
            [0.0897923, 0.8134230, 0.0967616],
            [0.0175439, 0.1115440, 0.8707040]
        ], dtype=np.float32)
        
        # Macierz ACES AP0 -> sRGB
        self.aces_to_srgb_matrix = np.array([
            [2.52169, -1.13413, -0.38756],
            [-0.27648, 1.37272, -0.09624],
            [-0.01538, -0.15298, 1.16835]
        ], dtype=np.float32)
        
        # Macierz Bradford dla adaptacji chromatycznej
        self.bradford_matrix = np.array([
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296]
        ], dtype=np.float32)
        
        self.bradford_inverse = np.linalg.inv(self.bradford_matrix)
    
    def transfer_colors(
        self, 
        source_image: np.ndarray, 
        target_image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Główna metoda transferu kolorów.
        
        Args:
            source_image: Obraz źródłowy (H, W, 3)
            target_image: Obraz docelowy (H, W, 3)
            mask: Opcjonalna maska (H, W) dla selektywnego przetwarzania
            
        Returns:
            Dict zawierający wynik, metryki jakości i metadane
        """
        import time
        start_time = time.time()
        
        try:
            # === ETAP 1: WALIDACJA ===
            self._validate_inputs(source_image, target_image, mask)
            
            # === ETAP 2: KONWERSJA DO ACES ===
            source_aces = self._convert_to_aces(source_image)
            target_aces = self._convert_to_aces(target_image)
            
            # === ETAP 3: ANALIZA STATYSTYK ===
            source_stats = self._analyze_statistics(source_aces, mask)
            target_stats = self._analyze_statistics(target_aces)
            
            # === ETAP 4: OBLICZENIE TRANSFORMACJI ===
            transform_data = self._calculate_transformation(
                source_stats, target_stats
            )
            
            # === ETAP 5: PREDYKCJA JAKOŚCI ===
            quality_prediction = self._predict_quality(
                source_stats, target_stats, transform_data
            )
            
            if quality_prediction['confidence'] < self.params.min_confidence:
                raise ValueError(
                    f"Niska pewność predykcji: {quality_prediction['confidence']:.3f}"
                )
            
            # === ETAP 6: APLIKACJA TRANSFORMACJI ===
            result_aces = self._apply_transformation(
                source_aces, transform_data, mask
            )
            
            # === ETAP 7: POST-PROCESSING ===
            if self.params.use_tone_mapping:
                result_aces = self._apply_tone_mapping(result_aces)
            
            if self.params.preserve_luminance:
                result_aces = self._preserve_luminance(
                    source_aces, result_aces, mask
                )
            
            if self.params.gamut_compression:
                result_aces = self._compress_gamut(result_aces)
            
            # === ETAP 8: KONWERSJA WYJŚCIOWA ===
            result_image = self._convert_from_aces(result_aces)
            
            # === ETAP 9: OCENA JAKOŚCI ===
            quality_metrics = self._evaluate_quality(
                source_image, result_image, target_image
            )
            
            # === ETAP 10: GENEROWANIE RAPORTU ===
            processing_time = time.time() - start_time
            report = self._generate_report(
                source_stats, target_stats, transform_data, 
                quality_metrics, processing_time
            )
            
            return {
                'image': result_image,
                'quality': quality_metrics,
                'report': report,
                'metadata': {
                    'transform_data': transform_data,
                    'processing_time': processing_time,
                    'parameters': self.params
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
```

### 2.2 Metody Konwersji Kolorów

```python
    def _convert_to_aces(self, image: np.ndarray) -> np.ndarray:
        """Konwersja obrazu do przestrzeni ACES AP0."""
        # Normalizacja do [0, 1]
        if image.dtype == np.uint8:
            normalized = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            normalized = image.astype(np.float32) / 65535.0
        else:
            normalized = image.astype(np.float32)
        
        # Konwersja do linear RGB (usunięcie gamma)
        linear_rgb = self._remove_gamma(normalized)
        
        # Transformacja do ACES AP0
        original_shape = linear_rgb.shape
        pixels = linear_rgb.reshape(-1, 3)
        
        # Aplikacja macierzy transformacji
        aces_pixels = np.dot(pixels, self.srgb_to_aces_matrix.T)
        
        # Ograniczenie do zakresu ACES (0, 65504)
        aces_pixels = np.clip(aces_pixels, 0.0, 65504.0)
        
        return aces_pixels.reshape(original_shape)
    
    def _convert_from_aces(self, aces_image: np.ndarray) -> np.ndarray:
        """Konwersja z przestrzeni ACES AP0 do sRGB."""
        original_shape = aces_image.shape
        pixels = aces_image.reshape(-1, 3)
        
        # Transformacja ACES -> sRGB
        srgb_pixels = np.dot(pixels, self.aces_to_srgb_matrix.T)
        
        # Ograniczenie do [0, 1]
        srgb_pixels = np.clip(srgb_pixels, 0.0, 1.0)
        
        # Aplikacja gamma
        gamma_corrected = self._apply_gamma(
            srgb_pixels.reshape(original_shape)
        )
        
        # Konwersja do uint8
        result = (gamma_corrected * 255.0).astype(np.uint8)
        
        return result
    
    def _remove_gamma(self, image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Usunięcie korekcji gamma (konwersja do linear)."""
        # sRGB ma specjalną krzywą gamma
        linear = np.where(
            image <= 0.04045,
            image / 12.92,
            np.power((image + 0.055) / 1.055, 2.4)
        )
        return linear
    
    def _apply_gamma(self, image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Aplikacja korekcji gamma."""
        # sRGB gamma encoding
        gamma_corrected = np.where(
            image <= 0.0031308,
            image * 12.92,
            1.055 * np.power(image, 1.0/2.4) - 0.055
        )
        return np.clip(gamma_corrected, 0.0, 1.0)
```

### 2.3 Analiza Statystyk

```python
    def _analyze_statistics(
        self, 
        aces_image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Analiza statystyk obrazu w przestrzeni ACES."""
        
        # Przygotowanie danych
        if mask is not None:
            # Zastosowanie maski
            mask_3d = np.stack([mask] * 3, axis=-1)
            pixels = aces_image[mask_3d].reshape(-1, 3)
        else:
            pixels = aces_image.reshape(-1, 3)
        
        # Podstawowe statystyki
        stats = {
            'mean': np.mean(pixels, axis=0),
            'std': np.std(pixels, axis=0),
            'min': np.min(pixels, axis=0),
            'max': np.max(pixels, axis=0),
            'median': np.median(pixels, axis=0)
        }
        
        # Percentyle
        percentiles = {}
        for p in [1, 5, 25, 75, 95, 99]:
            percentiles[f'p{p}'] = np.percentile(pixels, p, axis=0)
        stats['percentiles'] = percentiles
        
        # Histogramy
        histograms = {}
        for i, channel in enumerate(['R', 'G', 'B']):
            hist, bins = np.histogram(
                pixels[:, i], 
                bins=1024, 
                range=(0, 65504),
                density=True
            )
            histograms[channel] = {
                'histogram': hist,
                'bins': bins
            }
        stats['histograms'] = histograms
        
        # Luminancja ACES
        luminance = (
            0.2722287168 * pixels[:, 0] + 
            0.6740817658 * pixels[:, 1] + 
            0.0536895174 * pixels[:, 2]
        )
        
        stats['luminance'] = {
            'mean': np.mean(luminance),
            'std': np.std(luminance),
            'min': np.min(luminance),
            'max': np.max(luminance),
            'dynamic_range': np.max(luminance) - np.min(luminance)
        }
        
        # Temperatura kolorów (przybliżona)
        stats['color_temperature'] = self._estimate_color_temperature(
            stats['mean']
        )
        
        # Analiza kontrastu
        stats['contrast'] = self._analyze_contrast(aces_image, luminance)
        
        # Analiza gamut
        stats['gamut'] = self._analyze_gamut(pixels)
        
        return stats
    
    def _estimate_color_temperature(self, rgb_mean: np.ndarray) -> float:
        """Oszacowanie temperatury kolorów na podstawie średnich RGB."""
        # Konwersja do XYZ
        xyz = self._rgb_to_xyz(rgb_mean)
        
        # Obliczenie chromatyczności
        x = xyz[0] / (xyz[0] + xyz[1] + xyz[2] + 1e-10)
        y = xyz[1] / (xyz[0] + xyz[1] + xyz[2] + 1e-10)
        
        # McCamy's formula dla temperatury kolorów
        n = (x - 0.3320) / (0.1858 - y)
        cct = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33
        
        return np.clip(cct, 2000, 25000)  # Ograniczenie do rozsądnego zakresu
    
    def _rgb_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """Konwersja RGB do XYZ."""
        # Macierz sRGB -> XYZ (D65)
        matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        return np.dot(matrix, rgb)
    
    def _analyze_contrast(
        self, 
        aces_image: np.ndarray, 
        luminance: np.ndarray
    ) -> Dict[str, float]:
        """Analiza kontrastu obrazu."""
        # Kontrast globalny (RMS)
        global_contrast = np.std(luminance) / (np.mean(luminance) + 1e-10)
        
        # Kontrast lokalny (średnia z gradientów)
        lum_2d = luminance.reshape(aces_image.shape[:2])
        grad_x = np.gradient(lum_2d, axis=1)
        grad_y = np.gradient(lum_2d, axis=0)
        local_contrast = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Zakres dynamiczny
        dynamic_range = np.max(luminance) - np.min(luminance)
        
        return {
            'global': global_contrast,
            'local': local_contrast,
            'dynamic_range': dynamic_range
        }
    
    def _analyze_gamut(self, pixels: np.ndarray) -> Dict[str, Any]:
        """Analiza pokrycia gamut."""
        # Procent pikseli w gamut sRGB
        srgb_pixels = np.dot(pixels, self.aces_to_srgb_matrix.T)
        in_srgb_gamut = np.all((srgb_pixels >= 0) & (srgb_pixels <= 1), axis=1)
        srgb_coverage = np.mean(in_srgb_gamut)
        
        # Maksymalne wartości w każdym kanale
        max_values = np.max(pixels, axis=0)
        
        # Objętość gamut (przybliżona)
        gamut_volume = np.prod(max_values - np.min(pixels, axis=0))
        
        return {
            'srgb_coverage': srgb_coverage,
            'max_values': max_values,
            'volume': gamut_volume
        }
```

---

## 3. Metody Transformacji

### 3.1 Obliczenie Transformacji

```python
    def _calculate_transformation(
        self, 
        source_stats: Dict[str, Any], 
        target_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Obliczenie parametrów transformacji."""
        
        method = self.params.method
        
        if method == TransformMethod.CHROMATIC_ADAPTATION:
            return self._calculate_chromatic_adaptation(
                source_stats['color_temperature'],
                target_stats['color_temperature']
            )
        
        elif method == TransformMethod.STATISTICAL_MATCHING:
            return self._calculate_statistical_transform(
                source_stats, target_stats
            )
        
        elif method == TransformMethod.HISTOGRAM_MATCHING:
            return self._calculate_histogram_transform(
                source_stats['histograms'],
                target_stats['histograms']
            )
        
        elif method == TransformMethod.PERCEPTUAL_MATCHING:
            return self._calculate_perceptual_transform(
                source_stats, target_stats
            )
        
        elif method == TransformMethod.HYBRID:
            return self._calculate_hybrid_transform(
                source_stats, target_stats
            )
        
        else:
            raise ValueError(f"Nieznana metoda transformacji: {method}")
    
    def _calculate_statistical_transform(
        self, 
        source_stats: Dict[str, Any], 
        target_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Obliczenie transformacji statystycznej."""
        
        # Korekcja średniej
        mean_shift = target_stats['mean'] - source_stats['mean']
        
        # Korekcja odchylenia standardowego
        source_std = source_stats['std']
        target_std = target_stats['std']
        
        # Zabezpieczenie przed dzieleniem przez zero
        std_ratio = np.where(
            source_std > 1e-6,
            target_std / source_std,
            1.0
        )
        
        return {
            'method': 'statistical',
            'mean_shift': mean_shift,
            'std_ratio': std_ratio,
            'source_mean': source_stats['mean'],
            'target_mean': target_stats['mean']
        }
    
    def _calculate_chromatic_adaptation(
        self, 
        source_temp: float, 
        target_temp: float
    ) -> Dict[str, Any]:
        """Obliczenie adaptacji chromatycznej Bradford."""
        
        # Konwersja temperatur na illuminanty XYZ
        source_illuminant = self._temperature_to_xyz(source_temp)
        target_illuminant = self._temperature_to_xyz(target_temp)
        
        # Transformacja Bradford
        source_bradford = np.dot(self.bradford_matrix, source_illuminant)
        target_bradford = np.dot(self.bradford_matrix, target_illuminant)
        
        # Macierz adaptacji
        adaptation_diagonal = np.diag(
            target_bradford / (source_bradford + 1e-10)
        )
        
        # Finalna macierz transformacji
        adaptation_matrix = np.dot(
            self.bradford_inverse,
            np.dot(adaptation_diagonal, self.bradford_matrix)
        )
        
        return {
            'method': 'chromatic_adaptation',
            'matrix': adaptation_matrix,
            'source_temp': source_temp,
            'target_temp': target_temp
        }
    
    def _temperature_to_xyz(self, temperature: float) -> np.ndarray:
        """Konwersja temperatury kolorów na illuminant XYZ."""
        # Planck's law approximation dla illuminantów
        if temperature < 4000:
            # Ciepłe światło
            x = 0.23703 - 0.2441 * (1e9 / temperature**2) - \
                1.1063 * (1e6 / temperature) + 0.0003 * temperature
        else:
            # Zimne światło
            x = 0.23704 + 0.09456 * (1e9 / temperature**2) - \
                2.0064 * (1e6 / temperature) + 0.0001 * temperature
        
        y = -3.0 * x**2 + 2.87 * x - 0.275
        z = 1.0 - x - y
        
        # Normalizacja do Y = 1
        return np.array([x/y, 1.0, z/y])
```

### 3.2 Aplikacja Transformacji

```python
    def _apply_transformation(
        self, 
        source_aces: np.ndarray, 
        transform_data: Dict[str, Any],
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Aplikacja transformacji do obrazu ACES."""
        
        result = source_aces.copy()
        original_shape = result.shape
        
        # Przygotowanie maski
        if mask is not None:
            mask_3d = np.stack([mask] * 3, axis=-1)
            pixels_to_transform = result[mask_3d].reshape(-1, 3)
        else:
            pixels_to_transform = result.reshape(-1, 3)
            mask_3d = None
        
        # Aplikacja transformacji według metody
        method = transform_data['method']
        
        if method == 'statistical':
            # T(x) = (x - μ_src) * σ_ratio + μ_tgt
            centered = pixels_to_transform - transform_data['source_mean']
            scaled = centered * transform_data['std_ratio']
            transformed = scaled + transform_data['target_mean']
        
        elif method == 'chromatic_adaptation':
            # Aplikacja macierzy adaptacji
            transformed = np.dot(
                pixels_to_transform, 
                transform_data['matrix'].T
            )
        
        elif method == 'histogram_matching':
            transformed = self._apply_histogram_matching(
                pixels_to_transform, transform_data['luts']
            )
        
        else:
            raise ValueError(f"Nieobsługiwana metoda: {method}")
        
        # Ograniczenie do zakresu ACES
        transformed = np.clip(transformed, 0.0, 65504.0)
        
        # Zapisanie wyniku
        if mask is not None:
            result[mask_3d] = transformed.flatten()
        else:
            result = transformed.reshape(original_shape)
        
        return result
    
    def _apply_histogram_matching(
        self, 
        pixels: np.ndarray, 
        luts: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Aplikacja dopasowania histogramów."""
        
        result = pixels.copy()
        
        for i, channel in enumerate(['R', 'G', 'B']):
            lut = luts[channel]['lut']
            source_bins = luts[channel]['source_bins']
            
            # Interpolacja LUT
            result[:, i] = np.interp(
                pixels[:, i], 
                source_bins[:-1],  # Usunięcie ostatniego binu
                lut
            )
        
        return result
```

---

## 4. Post-Processing

### 4.1 Tone Mapping

```python
    def _apply_tone_mapping(self, aces_image: np.ndarray) -> np.ndarray:
        """Aplikacja tone mappingu ACES RRT."""
        
        # Parametry krzywej tonalnej
        a = self.params.tone_curve_a
        b = self.params.tone_curve_b
        c = self.params.tone_curve_c
        d = self.params.tone_curve_d
        e = self.params.tone_curve_e
        
        # ACES RRT formula
        numerator = aces_image * (a * aces_image + b)
        denominator = aces_image * (c * aces_image + d) + e
        
        # Zabezpieczenie przed dzieleniem przez zero
        safe_denominator = np.maximum(denominator, 1e-10)
        
        tone_mapped = numerator / safe_denominator
        
        # Ograniczenie do [0, 1]
        return np.clip(tone_mapped, 0.0, 1.0)
    
    def _preserve_luminance(
        self, 
        source_aces: np.ndarray, 
        result_aces: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Zachowanie luminancji źródłowej."""
        
        # Obliczenie luminancji
        source_lum = self._calculate_luminance(source_aces)
        result_lum = self._calculate_luminance(result_aces)
        
        # Korekcja luminancji
        weight = self.params.luminance_weight
        target_lum = weight * source_lum + (1 - weight) * result_lum
        
        # Aplikacja korekcji
        lum_ratio = np.divide(
            target_lum, 
            result_lum + 1e-10,
            out=np.ones_like(target_lum),
            where=(result_lum > 1e-10)
        )
        
        # Rozszerzenie na 3 kanały
        if len(lum_ratio.shape) == 2:
            lum_ratio = np.stack([lum_ratio] * 3, axis=-1)
        
        corrected = result_aces * lum_ratio
        
        # Aplikacja maski jeśli podana
        if mask is not None:
            mask_3d = np.stack([mask] * 3, axis=-1)
            result = result_aces.copy()
            result[mask_3d] = corrected[mask_3d]
            return result
        
        return corrected
    
    def _calculate_luminance(self, aces_image: np.ndarray) -> np.ndarray:
        """Obliczenie luminancji w przestrzeni ACES."""
        # Współczynniki luminancji ACES
        return (
            0.2722287168 * aces_image[..., 0] + 
            0.6740817658 * aces_image[..., 1] + 
            0.0536895174 * aces_image[..., 2]
        )
    
    def _compress_gamut(self, aces_image: np.ndarray) -> np.ndarray:
        """Kompresja gamut dla zachowania kolorów."""
        
        strength = self.params.compression_strength
        
        # Konwersja do sRGB dla sprawdzenia gamut
        original_shape = aces_image.shape
        pixels = aces_image.reshape(-1, 3)
        srgb_pixels = np.dot(pixels, self.aces_to_srgb_matrix.T)
        
        # Identyfikacja pikseli poza gamut
        out_of_gamut = np.any((srgb_pixels < 0) | (srgb_pixels > 1), axis=1)
        
        if not np.any(out_of_gamut):
            return aces_image  # Brak kompresji potrzebnej
        
        # Kompresja tylko pikseli poza gamut
        compressed_pixels = pixels.copy()
        
        for i in np.where(out_of_gamut)[0]:
            pixel = pixels[i]
            
            # Soft clipping z zachowaniem proporcji
            max_val = np.max(pixel)
            if max_val > 1.0:
                # Kompresja z krzywą sigmoid
                compression_factor = 1.0 / (1.0 + np.exp(-strength * (max_val - 1.0)))
                compressed_pixels[i] = pixel * compression_factor
        
        return compressed_pixels.reshape(original_shape)
```

---

## 5. Walidacja i Pomocnicze

### 5.1 Walidacja Wejścia

```python
    def _validate_inputs(
        self, 
        source_image: np.ndarray, 
        target_image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> None:
        """Walidacja danych wejściowych."""
        
        # Sprawdzenie typu i kształtu
        if not isinstance(source_image, np.ndarray):
            raise TypeError("source_image musi być numpy.ndarray")
        
        if not isinstance(target_image, np.ndarray):
            raise TypeError("target_image musi być numpy.ndarray")
        
        if len(source_image.shape) != 3 or source_image.shape[2] != 3:
            raise ValueError("source_image musi mieć kształt (H, W, 3)")
        
        if len(target_image.shape) != 3 or target_image.shape[2] != 3:
            raise ValueError("target_image musi mieć kształt (H, W, 3)")
        
        # Sprawdzenie maski
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError("mask musi być numpy.ndarray")
            
            if len(mask.shape) != 2:
                raise ValueError("mask musi mieć kształt (H, W)")
            
            if mask.shape != source_image.shape[:2]:
                raise ValueError(
                    "mask musi mieć ten sam rozmiar co obrazy"
                )
        
        # Sprawdzenie zakresu wartości
        if source_image.dtype in [np.uint8, np.uint16]:
            # OK - obsługiwane typy
            pass
        elif source_image.dtype in [np.float32, np.float64]:
            if np.any(source_image < 0) or np.any(source_image > 1):
                print("Ostrzeżenie: wartości float poza zakresem [0, 1]")
        else:
            raise ValueError(f"Nieobsługiwany typ danych: {source_image.dtype}")
```

### 5.2 Predykcja Jakości

```python
    def _predict_quality(
        self, 
        source_stats: Dict[str, Any], 
        target_stats: Dict[str, Any],
        transform_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predykcja jakości transformacji."""
        
        # Analiza podobieństwa statystyk
        mean_similarity = self._calculate_similarity(
            source_stats['mean'], target_stats['mean']
        )
        
        std_similarity = self._calculate_similarity(
            source_stats['std'], target_stats['std']
        )
        
        # Analiza różnicy temperatur kolorów
        temp_diff = abs(
            source_stats['color_temperature'] - 
            target_stats['color_temperature']
        )
        temp_score = max(0, 1 - temp_diff / 5000)  # Normalizacja
        
        # Analiza zakresu dynamicznego
        source_range = source_stats['luminance']['dynamic_range']
        target_range = target_stats['luminance']['dynamic_range']
        range_similarity = min(source_range, target_range) / \
                          (max(source_range, target_range) + 1e-10)
        
        # Kombinacja metryk
        confidence = (
            0.3 * mean_similarity +
            0.2 * std_similarity +
            0.3 * temp_score +
            0.2 * range_similarity
        )
        
        return {
            'confidence': confidence,
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'temperature_score': temp_score,
            'range_similarity': range_similarity
        }
    
    def _calculate_similarity(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> float:
        """Obliczenie podobieństwa między wektorami."""
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
```

---

**Następna część:** [4of6 - Parametry i Konfiguracja](gatto-WORKING-03-algorithms-08-advanced-01-aces-4of6.md)  
**Poprzednia część:** [2of6 - Pseudokod i Architektura](gatto-WORKING-03-algorithms-08-advanced-01-aces-2of6.md)  
**Powrót do:** [Spis treści](gatto-WORKING-03-algorithms-08-advanced-01-aces-0of6.md)

*Część 3of6 - Implementacja Core | Wersja 1.0 | 2024-01-20*