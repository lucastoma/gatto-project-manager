# ACES Color Space Transfer - Część 4of6: Parametry i Konfiguracja ⚙️

> **Seria:** ACES Color Space Transfer  
> **Część:** 4 z 6 - Parametry i Konfiguracja  
> **Wymagania:** [3of6 - Implementacja Core](gatto-WORKING-03-algorithms-08-advanced-01-aces-3of6.md)  
> **Następna część:** [5of6 - Analiza Wydajności](gatto-WORKING-03-algorithms-08-advanced-01-aces-5of6.md)

---

## 1. Rozszerzona Klasa Parametrów

### 1.1 Kompletna Definicja ACESParameters

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import json
from pathlib import Path

class TransformMethod(Enum):
    """Metody transformacji ACES."""
    CHROMATIC_ADAPTATION = "chromatic_adaptation"
    STATISTICAL_MATCHING = "statistical_matching"
    HISTOGRAM_MATCHING = "histogram_matching"
    PERCEPTUAL_MATCHING = "perceptual_matching"
    HYBRID = "hybrid"
    CUSTOM = "custom"

class ToneMappingCurve(Enum):
    """Typy krzywych tone mappingu."""
    ACES_RRT = "aces_rrt"
    REINHARD = "reinhard"
    FILMIC = "filmic"
    UCHIMURA = "uchimura"
    CUSTOM = "custom"

class ColorProfile(Enum):
    """Profile kolorów."""
    SRGB = "sRGB"
    ADOBE_RGB = "Adobe RGB"
    PROPHOTO_RGB = "ProPhoto RGB"
    REC2020 = "Rec. 2020"
    DCI_P3 = "DCI-P3"
    ACES_AP0 = "ACES AP0"
    ACES_AP1 = "ACES AP1"

@dataclass
class ToneMappingParameters:
    """Parametry tone mappingu."""
    curve_type: ToneMappingCurve = ToneMappingCurve.ACES_RRT
    
    # Parametry ACES RRT
    aces_a: float = 2.51
    aces_b: float = 0.03
    aces_c: float = 2.43
    aces_d: float = 0.59
    aces_e: float = 0.14
    
    # Parametry Reinhard
    reinhard_white: float = 1.0
    reinhard_adaptation: float = 1.0
    
    # Parametry Filmic
    filmic_shoulder_strength: float = 0.22
    filmic_linear_strength: float = 0.30
    filmic_linear_angle: float = 0.10
    filmic_toe_strength: float = 0.20
    filmic_toe_numerator: float = 0.01
    filmic_toe_denominator: float = 0.30
    filmic_linear_white: float = 11.2
    
    # Parametry Uchimura
    uchimura_max_brightness: float = 1.0
    uchimura_contrast: float = 1.0
    uchimura_linear_start: float = 0.22
    uchimura_linear_length: float = 0.4
    uchimura_black: float = 1.33
    uchimura_pedestal: float = 0.0
    
    # Parametry custom
    custom_function: Optional[callable] = None
    custom_params: Dict = field(default_factory=dict)

@dataclass
class LuminanceParameters:
    """Parametry zachowania luminancji."""
    preserve: bool = True
    weight: float = 0.8
    method: str = "weighted_blend"  # "weighted_blend", "histogram_match", "statistical"
    
    # Parametry dla histogram matching
    histogram_bins: int = 256
    histogram_smooth: bool = True
    
    # Parametry dla statistical matching
    match_mean: bool = True
    match_std: bool = True
    match_percentiles: List[int] = field(default_factory=lambda: [5, 95])

@dataclass
class GamutParameters:
    """Parametry kompresji gamut."""
    enable_compression: bool = True
    compression_strength: float = 0.7
    method: str = "soft_clip"  # "soft_clip", "desaturate", "hybrid"
    
    # Parametry soft clipping
    soft_clip_knee: float = 0.8
    soft_clip_ratio: float = 0.2
    
    # Parametry desaturacji
    desaturate_threshold: float = 1.0
    desaturate_strength: float = 0.5
    
    # Parametry hybrid
    hybrid_clip_weight: float = 0.6
    hybrid_desat_weight: float = 0.4

@dataclass
class QualityParameters:
    """Parametry kontroli jakości."""
    min_confidence: float = 0.7
    quality_threshold: float = 0.8
    
    # Wagi dla różnych metryk
    similarity_weight: float = 0.3
    temperature_weight: float = 0.3
    contrast_weight: float = 0.2
    gamut_weight: float = 0.2
    
    # Progi dla ostrzeżeń
    temperature_diff_warning: float = 1000  # K
    contrast_ratio_warning: float = 2.0
    gamut_coverage_warning: float = 0.5

@dataclass
class PerformanceParameters:
    """Parametry wydajności."""
    chunk_size: int = 1024
    use_parallel: bool = True
    num_threads: int = 4
    
    # Parametry pamięci
    max_memory_usage: float = 0.8  # Procent dostępnej RAM
    enable_memory_mapping: bool = False
    
    # Parametry GPU (jeśli dostępne)
    use_gpu: bool = False
    gpu_device_id: int = 0
    gpu_memory_fraction: float = 0.5
    
    # Cache
    enable_caching: bool = True
    cache_size: int = 100  # MB
    cache_directory: Optional[Path] = None

@dataclass
class DebugParameters:
    """Parametry debugowania."""
    enable_debug: bool = False
    save_intermediate: bool = False
    intermediate_directory: Optional[Path] = None
    
    # Logging
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    log_file: Optional[Path] = None
    
    # Profiling
    enable_profiling: bool = False
    profile_output: Optional[Path] = None
    
    # Visualization
    show_histograms: bool = False
    show_statistics: bool = False
    save_plots: bool = False
    plot_directory: Optional[Path] = None

@dataclass
class ACESParameters:
    """Główna klasa parametrów ACES Color Transfer."""
    
    # Podstawowe parametry
    method: TransformMethod = TransformMethod.STATISTICAL_MATCHING
    input_profile: ColorProfile = ColorProfile.SRGB
    output_profile: ColorProfile = ColorProfile.SRGB
    
    # Parametry komponentów
    tone_mapping: ToneMappingParameters = field(default_factory=ToneMappingParameters)
    luminance: LuminanceParameters = field(default_factory=LuminanceParameters)
    gamut: GamutParameters = field(default_factory=GamutParameters)
    quality: QualityParameters = field(default_factory=QualityParameters)
    performance: PerformanceParameters = field(default_factory=PerformanceParameters)
    debug: DebugParameters = field(default_factory=DebugParameters)
    
    # Parametry zaawansowane
    custom_transform_matrix: Optional[np.ndarray] = None
    custom_white_point: Optional[Tuple[float, float]] = None
    
    # Metadane
    name: str = "default"
    description: str = ""
    version: str = "1.0"
    author: str = ""
    created_date: Optional[str] = None
    
    def __post_init__(self):
        """Walidacja parametrów po inicjalizacji."""
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Walidacja spójności parametrów."""
        # Walidacja zakresów
        if not 0.0 <= self.tone_mapping.aces_a <= 10.0:
            raise ValueError("tone_mapping.aces_a musi być w zakresie [0, 10]")
        
        if not 0.0 <= self.luminance.weight <= 1.0:
            raise ValueError("luminance.weight musi być w zakresie [0, 1]")
        
        if not 0.0 <= self.gamut.compression_strength <= 1.0:
            raise ValueError("gamut.compression_strength musi być w zakresie [0, 1]")
        
        if not 0.1 <= self.quality.min_confidence <= 1.0:
            raise ValueError("quality.min_confidence musi być w zakresie [0.1, 1]")
        
        # Walidacja spójności
        if self.performance.chunk_size < 64:
            print("Ostrzeżenie: Mały chunk_size może wpłynąć na wydajność")
        
        if self.performance.num_threads > 16:
            print("Ostrzeżenie: Duża liczba wątków może nie przynieść korzyści")
    
    def to_dict(self) -> Dict:
        """Konwersja do słownika."""
        def convert_value(value):
            if isinstance(value, Enum):
                return value.value
            elif isinstance(value, Path):
                return str(value)
            elif hasattr(value, '__dict__'):
                return {k: convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            else:
                return value
        
        return {k: convert_value(v) for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ACESParameters':
        """Tworzenie z słownika."""
        # Implementacja deserializacji
        # (uproszczona wersja)
        return cls(**data)
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Zapis parametrów do pliku JSON."""
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'ACESParameters':
        """Wczytanie parametrów z pliku JSON."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
```

---

## 2. Predefiniowane Konfiguracje

### 2.1 Konfiguracje dla Różnych Scenariuszy

```python
class ACESPresets:
    """Predefiniowane konfiguracje ACES."""
    
    @staticmethod
    def portrait_photography() -> ACESParameters:
        """Konfiguracja dla fotografii portretowej."""
        params = ACESParameters(
            name="Portrait Photography",
            description="Optymalizowane dla fotografii portretowej z zachowaniem tonów skóry",
            method=TransformMethod.PERCEPTUAL_MATCHING
        )
        
        # Delikatne tone mapping
        params.tone_mapping.curve_type = ToneMappingCurve.ACES_RRT
        params.tone_mapping.aces_a = 2.2  # Mniej agresywne
        params.tone_mapping.aces_c = 2.1
        
        # Zachowanie luminancji skóry
        params.luminance.preserve = True
        params.luminance.weight = 0.9
        params.luminance.method = "weighted_blend"
        
        # Delikatna kompresja gamut
        params.gamut.compression_strength = 0.5
        params.gamut.method = "soft_clip"
        
        # Wysokie wymagania jakościowe
        params.quality.min_confidence = 0.8
        params.quality.temperature_weight = 0.4  # Ważna temperatura kolorów
        
        return params
    
    @staticmethod
    def landscape_photography() -> ACESParameters:
        """Konfiguracja dla fotografii krajobrazowej."""
        params = ACESParameters(
            name="Landscape Photography",
            description="Optymalizowane dla fotografii krajobrazowej z naciskiem na kontrast",
            method=TransformMethod.STATISTICAL_MATCHING
        )
        
        # Bardziej agresywne tone mapping
        params.tone_mapping.curve_type = ToneMappingCurve.FILMIC
        params.tone_mapping.filmic_shoulder_strength = 0.25
        params.tone_mapping.filmic_linear_strength = 0.35
        
        # Zachowanie kontrastu
        params.luminance.preserve = True
        params.luminance.weight = 0.7
        params.luminance.method = "statistical"
        params.luminance.match_percentiles = [1, 99]  # Pełny zakres
        
        # Silniejsza kompresja gamut
        params.gamut.compression_strength = 0.8
        params.gamut.method = "hybrid"
        
        # Nacisk na kontrast
        params.quality.contrast_weight = 0.4
        params.quality.gamut_weight = 0.3
        
        return params
    
    @staticmethod
    def cinematic_grading() -> ACESParameters:
        """Konfiguracja dla gradingu filmowego."""
        params = ACESParameters(
            name="Cinematic Grading",
            description="Profesjonalny grading filmowy z pełną kontrolą",
            method=TransformMethod.HYBRID
        )
        
        # Filmowe tone mapping
        params.tone_mapping.curve_type = ToneMappingCurve.UCHIMURA
        params.tone_mapping.uchimura_max_brightness = 1.2
        params.tone_mapping.uchimura_contrast = 1.1
        
        # Precyzyjne zachowanie luminancji
        params.luminance.preserve = True
        params.luminance.weight = 0.85
        params.luminance.method = "histogram_match"
        params.luminance.histogram_bins = 512
        
        # Zaawansowana kompresja gamut
        params.gamut.compression_strength = 0.9
        params.gamut.method = "hybrid"
        params.gamut.hybrid_clip_weight = 0.7
        
        # Wysokie standardy jakości
        params.quality.min_confidence = 0.85
        params.quality.quality_threshold = 0.9
        
        # Wydajność dla dużych plików
        params.performance.chunk_size = 2048
        params.performance.use_parallel = True
        params.performance.num_threads = 8
        
        return params
    
    @staticmethod
    def web_optimization() -> ACESParameters:
        """Konfiguracja dla optymalizacji webowej."""
        params = ACESParameters(
            name="Web Optimization",
            description="Szybkie przetwarzanie dla aplikacji webowych",
            method=TransformMethod.CHROMATIC_ADAPTATION
        )
        
        # Proste tone mapping
        params.tone_mapping.curve_type = ToneMappingCurve.REINHARD
        params.tone_mapping.reinhard_white = 0.9
        
        # Podstawowe zachowanie luminancji
        params.luminance.preserve = True
        params.luminance.weight = 0.6
        params.luminance.method = "weighted_blend"
        
        # Szybka kompresja
        params.gamut.compression_strength = 0.6
        params.gamut.method = "soft_clip"
        
        # Niższe wymagania jakościowe dla szybkości
        params.quality.min_confidence = 0.6
        
        # Optymalizacja wydajności
        params.performance.chunk_size = 512
        params.performance.use_parallel = True
        params.performance.num_threads = 2
        params.performance.enable_caching = True
        
        return params
    
    @staticmethod
    def high_quality_print() -> ACESParameters:
        """Konfiguracja dla wysokiej jakości druku."""
        params = ACESParameters(
            name="High Quality Print",
            description="Maksymalna jakość dla druku profesjonalnego",
            method=TransformMethod.PERCEPTUAL_MATCHING,
            output_profile=ColorProfile.ADOBE_RGB
        )
        
        # Precyzyjne tone mapping
        params.tone_mapping.curve_type = ToneMappingCurve.ACES_RRT
        params.tone_mapping.aces_a = 2.51
        params.tone_mapping.aces_b = 0.03
        
        # Maksymalne zachowanie luminancji
        params.luminance.preserve = True
        params.luminance.weight = 0.95
        params.luminance.method = "histogram_match"
        params.luminance.histogram_bins = 1024
        
        # Minimalna kompresja gamut
        params.gamut.compression_strength = 0.3
        params.gamut.method = "desaturate"
        
        # Najwyższe standardy jakości
        params.quality.min_confidence = 0.9
        params.quality.quality_threshold = 0.95
        
        # Maksymalna precyzja
        params.performance.chunk_size = 4096
        params.performance.use_parallel = True
        params.performance.num_threads = 16
        
        # Debug dla kontroli jakości
        params.debug.enable_debug = True
        params.debug.save_intermediate = True
        params.debug.show_statistics = True
        
        return params
    
    @staticmethod
    def mobile_processing() -> ACESParameters:
        """Konfiguracja dla urządzeń mobilnych."""
        params = ACESParameters(
            name="Mobile Processing",
            description="Optymalizowane dla ograniczonych zasobów mobilnych",
            method=TransformMethod.STATISTICAL_MATCHING
        )
        
        # Uproszczone tone mapping
        params.tone_mapping.curve_type = ToneMappingCurve.REINHARD
        
        # Podstawowe zachowanie luminancji
        params.luminance.preserve = True
        params.luminance.weight = 0.7
        params.luminance.method = "weighted_blend"
        
        # Szybka kompresja
        params.gamut.compression_strength = 0.7
        params.gamut.method = "soft_clip"
        
        # Zbalansowane wymagania jakościowe
        params.quality.min_confidence = 0.65
        
        # Optymalizacja dla mobilnych
        params.performance.chunk_size = 256
        params.performance.use_parallel = False  # Unikanie overhead
        params.performance.num_threads = 1
        params.performance.max_memory_usage = 0.5  # Oszczędność pamięci
        params.performance.enable_caching = False
        
        return params
```

---

## 3. Dynamiczna Konfiguracja

### 3.1 Automatyczne Dostosowanie Parametrów

```python
class ACESConfigurationManager:
    """Manager automatycznej konfiguracji ACES."""
    
    def __init__(self):
        self.presets = ACESPresets()
        self._analysis_cache = {}
    
    def auto_configure(
        self, 
        source_image: np.ndarray, 
        target_image: np.ndarray,
        use_case: str = "general",
        performance_priority: str = "balanced"  # "speed", "quality", "balanced"
    ) -> ACESParameters:
        """Automatyczna konfiguracja na podstawie analizy obrazów."""
        
        # Analiza obrazów
        source_analysis = self._analyze_image_characteristics(source_image)
        target_analysis = self._analyze_image_characteristics(target_image)
        
        # Wybór bazowej konfiguracji
        base_params = self._select_base_configuration(
            source_analysis, target_analysis, use_case
        )
        
        # Dostosowanie do charakterystyk obrazów
        adjusted_params = self._adjust_for_image_characteristics(
            base_params, source_analysis, target_analysis
        )
        
        # Optymalizacja wydajności
        final_params = self._optimize_for_performance(
            adjusted_params, source_image.shape, performance_priority
        )
        
        return final_params
    
    def _analyze_image_characteristics(self, image: np.ndarray) -> Dict:
        """Analiza charakterystyk obrazu."""
        # Cache dla wydajności
        image_hash = hash(image.tobytes())
        if image_hash in self._analysis_cache:
            return self._analysis_cache[image_hash]
        
        # Podstawowe statystyki
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        analysis = {
            'resolution': image.shape[:2],
            'channels': image.shape[2] if len(image.shape) == 3 else 1,
            'dtype': image.dtype,
            'mean_brightness': np.mean(gray),
            'contrast': np.std(gray),
            'dynamic_range': np.max(gray) - np.min(gray)
        }
        
        # Analiza kolorów (jeśli kolorowy)
        if len(image.shape) == 3:
            analysis.update({
                'color_cast': self._detect_color_cast(image),
                'saturation': self._calculate_saturation(image),
                'temperature': self._estimate_temperature(image)
            })
        
        # Analiza zawartości
        analysis.update({
            'content_type': self._classify_content(image),
            'noise_level': self._estimate_noise(gray),
            'sharpness': self._calculate_sharpness(gray)
        })
        
        # Cache wyników
        self._analysis_cache[image_hash] = analysis
        return analysis
    
    def _detect_color_cast(self, image: np.ndarray) -> Dict[str, float]:
        """Detekcja dominującej barwy."""
        mean_rgb = np.mean(image.reshape(-1, 3), axis=0)
        
        # Normalizacja
        total = np.sum(mean_rgb)
        if total > 0:
            normalized = mean_rgb / total
        else:
            normalized = np.array([1/3, 1/3, 1/3])
        
        return {
            'red_cast': normalized[0] - 1/3,
            'green_cast': normalized[1] - 1/3,
            'blue_cast': normalized[2] - 1/3,
            'strength': np.std(normalized)
        }
    
    def _calculate_saturation(self, image: np.ndarray) -> float:
        """Obliczenie średniego nasycenia."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return np.mean(hsv[:, :, 1]) / 255.0
    
    def _estimate_temperature(self, image: np.ndarray) -> float:
        """Oszacowanie temperatury kolorów."""
        # Uproszczona metoda na podstawie stosunku R/B
        mean_rgb = np.mean(image.reshape(-1, 3), axis=0)
        
        if mean_rgb[2] > 0:  # Unikanie dzielenia przez zero
            rb_ratio = mean_rgb[0] / mean_rgb[2]
            # Przybliżona konwersja na temperaturę
            temperature = 4000 + (rb_ratio - 1) * 2000
            return np.clip(temperature, 2000, 10000)
        
        return 5500  # Domyślna temperatura
    
    def _classify_content(self, image: np.ndarray) -> str:
        """Klasyfikacja typu zawartości obrazu."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Analiza histogramu
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / np.sum(hist)
        
        # Charakterystyki różnych typów
        dark_pixels = np.sum(hist_norm[:64])  # Ciemne piksele
        bright_pixels = np.sum(hist_norm[192:])  # Jasne piksele
        mid_pixels = np.sum(hist_norm[64:192])  # Średnie piksele
        
        # Analiza gradientów (tekstura)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_score = np.mean(gradient_magnitude)
        
        # Klasyfikacja
        if dark_pixels > 0.4 and bright_pixels < 0.1:
            return "low_key"  # Ciemny obraz
        elif bright_pixels > 0.3 and dark_pixels < 0.1:
            return "high_key"  # Jasny obraz
        elif texture_score > 50:
            return "detailed"  # Dużo detali
        elif mid_pixels > 0.6:
            return "portrait"  # Prawdopodobnie portret
        else:
            return "general"
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Oszacowanie poziomu szumu."""
        # Laplacian variance method
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Obliczenie ostrości obrazu."""
        # Tenengrad method
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.mean(grad_x**2 + grad_y**2)
    
    def _select_base_configuration(
        self, 
        source_analysis: Dict, 
        target_analysis: Dict, 
        use_case: str
    ) -> ACESParameters:
        """Wybór bazowej konfiguracji."""
        
        # Mapowanie przypadków użycia
        use_case_mapping = {
            "portrait": self.presets.portrait_photography,
            "landscape": self.presets.landscape_photography,
            "cinematic": self.presets.cinematic_grading,
            "web": self.presets.web_optimization,
            "print": self.presets.high_quality_print,
            "mobile": self.presets.mobile_processing
        }
        
        # Automatyczna detekcja na podstawie analizy
        if use_case == "auto":
            source_content = source_analysis.get('content_type', 'general')
            if source_content == "portrait":
                use_case = "portrait"
            elif source_content in ["low_key", "high_key"]:
                use_case = "cinematic"
            elif source_analysis['resolution'][0] * source_analysis['resolution'][1] < 1000000:
                use_case = "web"
            else:
                use_case = "general"
        
        # Wybór konfiguracji
        if use_case in use_case_mapping:
            return use_case_mapping[use_case]()
        else:
            return self.presets.portrait_photography()  # Domyślna
    
    def _adjust_for_image_characteristics(
        self, 
        params: ACESParameters, 
        source_analysis: Dict, 
        target_analysis: Dict
    ) -> ACESParameters:
        """Dostosowanie parametrów do charakterystyk obrazów."""
        
        adjusted = params
        
        # Dostosowanie tone mappingu do kontrastu
        contrast_diff = abs(source_analysis['contrast'] - target_analysis['contrast'])
        if contrast_diff > 50:  # Duża różnica kontrastu
            if params.tone_mapping.curve_type == ToneMappingCurve.ACES_RRT:
                adjusted.tone_mapping.aces_a *= 1.2  # Bardziej agresywne
        
        # Dostosowanie luminancji do jasności
        brightness_diff = abs(source_analysis['mean_brightness'] - target_analysis['mean_brightness'])
        if brightness_diff > 50:
            adjusted.luminance.weight = min(0.95, adjusted.luminance.weight + 0.1)
        
        # Dostosowanie gamut do nasycenia
        if 'saturation' in source_analysis and 'saturation' in target_analysis:
            sat_diff = abs(source_analysis['saturation'] - target_analysis['saturation'])
            if sat_diff > 0.2:
                adjusted.gamut.compression_strength = min(0.9, adjusted.gamut.compression_strength + 0.1)
        
        # Dostosowanie do temperatury kolorów
        if 'temperature' in source_analysis and 'temperature' in target_analysis:
            temp_diff = abs(source_analysis['temperature'] - target_analysis['temperature'])
            if temp_diff > 1000:  # Duża różnica temperatury
                if adjusted.method == TransformMethod.STATISTICAL_MATCHING:
                    adjusted.method = TransformMethod.CHROMATIC_ADAPTATION
        
        return adjusted
    
    def _optimize_for_performance(
        self, 
        params: ACESParameters, 
        image_shape: Tuple[int, ...], 
        priority: str
    ) -> ACESParameters:
        """Optymalizacja wydajności."""
        
        optimized = params
        image_size = image_shape[0] * image_shape[1]
        
        if priority == "speed":
            # Priorytet szybkości
            optimized.performance.chunk_size = min(512, optimized.performance.chunk_size)
            optimized.performance.use_parallel = True
            optimized.quality.min_confidence = max(0.5, optimized.quality.min_confidence - 0.1)
            
            # Uproszczenie algorytmów
            if optimized.method == TransformMethod.PERCEPTUAL_MATCHING:
                optimized.method = TransformMethod.STATISTICAL_MATCHING
            
        elif priority == "quality":
            # Priorytet jakości
            optimized.performance.chunk_size = min(4096, max(1024, image_size // 1000))
            optimized.quality.min_confidence = min(0.95, optimized.quality.min_confidence + 0.1)
            
            # Włączenie debugowania
            optimized.debug.enable_debug = True
            optimized.debug.save_intermediate = True
            
        else:  # balanced
            # Zbalansowane podejście
            if image_size > 4000000:  # Duże obrazy
                optimized.performance.chunk_size = 2048
                optimized.performance.num_threads = min(8, optimized.performance.num_threads)
            elif image_size < 500000:  # Małe obrazy
                optimized.performance.chunk_size = 256
                optimized.performance.use_parallel = False
        
        return optimized
```

---

## 4. Walidacja i Testowanie Konfiguracji

### 4.1 System Walidacji

```python
class ACESConfigurationValidator:
    """Walidator konfiguracji ACES."""
    
    def __init__(self):
        self.validation_rules = self._initialize_rules()
    
    def validate_configuration(self, params: ACESParameters) -> Dict[str, List[str]]:
        """Walidacja kompletnej konfiguracji."""
        
        issues = {
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Walidacja podstawowych parametrów
        self._validate_basic_parameters(params, issues)
        
        # Walidacja tone mappingu
        self._validate_tone_mapping(params.tone_mapping, issues)
        
        # Walidacja luminancji
        self._validate_luminance(params.luminance, issues)
        
        # Walidacja gamut
        self._validate_gamut(params.gamut, issues)
        
        # Walidacja wydajności
        self._validate_performance(params.performance, issues)
        
        # Walidacja spójności
        self._validate_consistency(params, issues)
        
        return issues
    
    def _validate_basic_parameters(self, params: ACESParameters, issues: Dict):
        """Walidacja podstawowych parametrów."""
        
        # Sprawdzenie metody
        if not isinstance(params.method, TransformMethod):
            issues['errors'].append("Nieprawidłowa metoda transformacji")
        
        # Sprawdzenie profili kolorów
        if params.input_profile == params.output_profile:
            issues['warnings'].append(
                "Profile wejściowy i wyjściowy są identyczne"
            )
        
        # Sprawdzenie metadanych
        if not params.name:
            issues['suggestions'].append("Dodaj nazwę konfiguracji")
        
        if not params.description:
            issues['suggestions'].append("Dodaj opis konfiguracji")
    
    def _validate_tone_mapping(self, tm_params: ToneMappingParameters, issues: Dict):
        """Walidacja parametrów tone mappingu."""
        
        if tm_params.curve_type == ToneMappingCurve.ACES_RRT:
            # Walidacja parametrów ACES
            if not 0.1 <= tm_params.aces_a <= 10.0:
                issues['errors'].append("aces_a poza zakresem [0.1, 10.0]")
            
            if not 0.0 <= tm_params.aces_b <= 1.0:
                issues['errors'].append("aces_b poza zakresem [0.0, 1.0]")
            
            # Sprawdzenie typowych wartości
            if tm_params.aces_a > 5.0:
                issues['warnings'].append(
                    "Wysoka wartość aces_a może powodować przesycenie"
                )
        
        elif tm_params.curve_type == ToneMappingCurve.CUSTOM:
            if tm_params.custom_function is None:
                issues['errors'].append(
                    "Brak funkcji custom dla tone mappingu CUSTOM"
                )
    
    def _validate_performance(self, perf_params: PerformanceParameters, issues: Dict):
        """Walidacja parametrów wydajności."""
        
        # Sprawdzenie chunk_size
        if perf_params.chunk_size < 64:
            issues['warnings'].append(
                "Mały chunk_size może wpłynąć na wydajność"
            )
        elif perf_params.chunk_size > 8192:
            issues['warnings'].append(
                "Duży chunk_size może powodować problemy z pamięcią"
            )
        
        # Sprawdzenie liczby wątków
        import os
        cpu_count = os.cpu_count() or 1
        
        if perf_params.num_threads > cpu_count * 2:
            issues['warnings'].append(
                f"Liczba wątków ({perf_params.num_threads}) > 2x CPU cores ({cpu_count})"
            )
        
        # Sprawdzenie pamięci
        if perf_params.max_memory_usage > 0.9:
            issues['warnings'].append(
                "Wysokie użycie pamięci może powodować problemy systemowe"
            )
    
    def _validate_consistency(self, params: ACESParameters, issues: Dict):
        """Walidacja spójności między parametrami."""
        
        # Spójność metody i parametrów
        if params.method == TransformMethod.CHROMATIC_ADAPTATION:
            if params.luminance.method == "histogram_match":
                issues['suggestions'].append(
                    "Dla adaptacji chromatycznej lepiej użyć weighted_blend"
                )
        
        # Spójność jakości i wydajności
        if (params.quality.min_confidence > 0.9 and 
            params.performance.chunk_size < 512):
            issues['suggestions'].append(
                "Wysokie wymagania jakościowe + mały chunk_size = wolne przetwarzanie"
            )
        
        # Spójność debug i wydajności
        if (params.debug.enable_debug and 
            params.performance.use_parallel and 
            params.performance.num_threads > 4):
            issues['warnings'].append(
                "Debug mode z wieloma wątkami może utrudnić analizę"
            )
    
    def _initialize_rules(self) -> Dict:
        """Inicjalizacja reguł walidacji."""
        return {
            'parameter_ranges': {
                'tone_mapping.aces_a': (0.1, 10.0),
                'tone_mapping.aces_b': (0.0, 1.0),
                'luminance.weight': (0.0, 1.0),
                'gamut.compression_strength': (0.0, 1.0),
                'quality.min_confidence': (0.1, 1.0)
            },
            'recommended_values': {
                'tone_mapping.aces_a': (2.0, 3.0),
                'luminance.weight': (0.7, 0.9),
                'performance.chunk_size': (512, 2048)
            }
        }
```

---

## 5. Przykłady Użycia

### 5.1 Podstawowe Użycie Presetów

```python
# Przykład 1: Użycie presetu dla fotografii portretowej
from aces_color_transfer import ACESColorTransfer, ACESPresets

# Wczytanie obrazów
source_image = cv2.imread('portrait_source.jpg')
target_image = cv2.imread('portrait_target.jpg')

# Użycie presetu
params = ACESPresets.portrait_photography()
aces_transfer = ACESColorTransfer(params)

# Transfer kolorów
result = aces_transfer.transfer_colors(source_image, target_image)

print(f"Jakość transferu: {result['quality']['overall_score']:.3f}")
cv2.imwrite('result_portrait.jpg', result['image'])
```

### 5.2 Automatyczna Konfiguracja

```python
# Przykład 2: Automatyczna konfiguracja
from aces_color_transfer import ACESConfigurationManager

config_manager = ACESConfigurationManager()

# Automatyczne dostosowanie parametrów
auto_params = config_manager.auto_configure(
    source_image, 
    target_image,
    use_case="auto",
    performance_priority="balanced"
)

# Walidacja konfiguracji
validator = ACESConfigurationValidator()
issues = validator.validate_configuration(auto_params)

if issues['errors']:
    print("Błędy konfiguracji:", issues['errors'])
else:
    # Użycie automatycznej konfiguracji
    aces_transfer = ACESColorTransfer(auto_params)
    result = aces_transfer.transfer_colors(source_image, target_image)
```

### 5.3 Niestandardowa Konfiguracja

```python
# Przykład 3: Tworzenie niestandardowej konfiguracji
from aces_color_transfer import ACESParameters, TransformMethod, ToneMappingCurve

# Tworzenie niestandardowych parametrów
custom_params = ACESParameters(
    name="Custom Artistic",
    description="Artystyczna stylizacja z mocnym kontrastem",
    method=TransformMethod.HYBRID
)

# Dostosowanie tone mappingu
custom_params.tone_mapping.curve_type = ToneMappingCurve.FILMIC
custom_params.tone_mapping.filmic_shoulder_strength = 0.3
custom_params.tone_mapping.filmic_linear_strength = 0.4

# Dostosowanie luminancji
custom_params.luminance.weight = 0.6
custom_params.luminance.method = "statistical"

# Dostosowanie gamut
custom_params.gamut.compression_strength = 0.9
custom_params.gamut.method = "hybrid"

# Zapis konfiguracji
custom_params.save_to_file("custom_artistic.json")

# Użycie
aces_transfer = ACESColorTransfer(custom_params)
result = aces_transfer.transfer_colors(source_image, target_image)
```

---

**Następna część:** [5of6 - Analiza Wydajności](gatto-WORKING-03-algorithms-08-advanced-01-aces-5of6.md)  
**Poprzednia część:** [3of6 - Implementacja Core](gatto-WORKING-03-algorithms-08-advanced-01-aces-3of6.md)  
**Powrót do:** [Spis treści](gatto-WORKING-03-algorithms-08-advanced-01-aces-0of6.md)

*Część 4of6 - Parametry i Konfiguracja | Wersja 1.0 | 2024-01-20*