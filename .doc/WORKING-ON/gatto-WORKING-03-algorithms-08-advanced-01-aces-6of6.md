# ACES Color Space Transfer - Część 6of6: Aplikacje Praktyczne i Rozwiązywanie Problemów 🎯

> **Seria:** ACES Color Space Transfer  
> **Część:** 6 z 6 - Aplikacje Praktyczne i Rozwiązywanie Problemów  
> **Wymagania:** [5of6 - Analiza Wydajności](gatto-WORKING-03-algorithms-08-advanced-01-aces-5of6.md)  
> **Powrót do:** [Spis treści](gatto-WORKING-03-algorithms-08-advanced-01-aces-0of6.md)

---

## 1. Typowe Problemy i Rozwiązania

### 1.1 Problem: Przesycone Kolory

**Objawy:**
- Kolory wyglądają nienaturalnie intensywnie
- Utrata detali w obszarach o wysokiej saturacji
- Nieprawidłowe odwzorowanie skóry

**Rozwiązanie:**

```python
def fix_oversaturated_colors(
    aces_transfer: ACESColorTransfer,
    source_image: np.ndarray,
    target_image: np.ndarray
) -> Dict:
    """Naprawienie problemu przesyconych kolorów."""
    
    # Analiza saturacji
    source_hsv = cv2.cvtColor(source_image, cv2.COLOR_RGB2HSV)
    target_hsv = cv2.cvtColor(target_image, cv2.COLOR_RGB2HSV)
    
    source_saturation = source_hsv[:, :, 1].mean()
    target_saturation = target_hsv[:, :, 1].mean()
    
    saturation_ratio = target_saturation / max(source_saturation, 0.01)
    
    # Modyfikacja parametrów
    params = aces_transfer.params
    
    if saturation_ratio > 1.5:  # Cel jest znacznie bardziej nasycony
        # Zmniejszenie siły transformacji
        params.gamut.compression_strength = min(0.7, params.gamut.compression_strength)
        params.gamut.saturation_limit = 0.8
        
        # Użycie łagodniejszej metody
        if params.method == TransformMethod.PERCEPTUAL_MATCHING:
            params.method = TransformMethod.STATISTICAL_MATCHING
        
        # Dodatkowa kontrola saturacji
        params.gamut.enable_saturation_control = True
        params.gamut.max_saturation_boost = 1.3
    
    # Transfer z poprawionymi parametrami
    result = aces_transfer.transfer_colors(source_image, target_image)
    
    # Post-processing: dodatkowa kontrola saturacji
    if 'image' in result:
        result['image'] = apply_saturation_control(
            result['image'], max_saturation=0.9
        )
    
    result['fixes_applied'] = ['oversaturation_control']
    return result

def apply_saturation_control(
    image: np.ndarray, 
    max_saturation: float = 0.9
) -> np.ndarray:
    """Kontrola saturacji w obrazie wynikowym."""
    
    # Konwersja do HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] /= 255.0  # Normalizacja saturacji
    hsv[:, :, 2] /= 255.0  # Normalizacja value
    
    # Soft clipping saturacji
    hsv[:, :, 1] = np.where(
        hsv[:, :, 1] > max_saturation,
        max_saturation + (hsv[:, :, 1] - max_saturation) * 0.3,
        hsv[:, :, 1]
    )
    
    # Clamp do [0, 1]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
    
    # Konwersja z powrotem
    hsv[:, :, 1] *= 255.0
    hsv[:, :, 2] *= 255.0
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
```

### 1.2 Problem: Utrata Detali w Cieniach

**Objawy:**
- Ciemne obszary stają się jednolicie czarne
- Brak gradacji w cieniach
- Utrata tekstur w ciemnych partiach

**Rozwiązanie:**

```python
def fix_shadow_detail_loss(
    aces_transfer: ACESColorTransfer,
    source_image: np.ndarray,
    target_image: np.ndarray
) -> Dict:
    """Naprawienie utraty detali w cieniach."""
    
    # Analiza rozkładu luminancji
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB)
    
    source_l = source_lab[:, :, 0]
    target_l = target_lab[:, :, 0]
    
    # Analiza cieni (dolne 25% luminancji)
    source_shadow_threshold = np.percentile(source_l, 25)
    target_shadow_threshold = np.percentile(target_l, 25)
    
    shadow_ratio = target_shadow_threshold / max(source_shadow_threshold, 1)
    
    # Modyfikacja parametrów dla ochrony cieni
    params = aces_transfer.params
    
    if shadow_ratio < 0.7:  # Cel ma ciemniejsze cienie
        # Ochrona cieni
        params.luminance.shadow_protection = True
        params.luminance.shadow_threshold = 0.1
        params.luminance.shadow_boost = 1.2
        
        # Łagodniejsze tone mapping
        params.tone_mapping.shadow_contrast = 0.8
        params.tone_mapping.preserve_shadows = True
        
        # Użycie metody zachowującej luminancję
        params.luminance.method = "preserve_local"
        params.luminance.local_adaptation = True
    
    # Transfer z ochroną cieni
    result = aces_transfer.transfer_colors(source_image, target_image)
    
    # Post-processing: dodatkowe podniesienie cieni
    if 'image' in result:
        result['image'] = enhance_shadow_details(
            result['image'], 
            boost_factor=1.1,
            preserve_midtones=True
        )
    
    result['fixes_applied'] = ['shadow_detail_preservation']
    return result

def enhance_shadow_details(
    image: np.ndarray,
    boost_factor: float = 1.1,
    preserve_midtones: bool = True
) -> np.ndarray:
    """Wzmocnienie detali w cieniach."""
    
    # Konwersja do LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0] / 255.0
    
    # Maska cieni (adaptacyjna)
    shadow_mask = np.where(l_channel < 0.3, 1.0, 0.0)
    
    if preserve_midtones:
        # Smooth transition
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    
    # Boost cieni
    l_boosted = l_channel + (l_channel * (boost_factor - 1.0) * shadow_mask)
    l_boosted = np.clip(l_boosted, 0, 1)
    
    # Aktualizacja kanału L
    lab[:, :, 0] = l_boosted * 255.0
    
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
```

### 1.3 Problem: Nieprawidłowa Temperatura Barwowa

**Objawy:**
- Obraz ma nieprawidłowy odcień (zbyt ciepły/zimny)
- Białe obszary mają kolorowy odcień
- Nienaturalne kolory skóry

**Rozwiązanie:**

```python
def fix_color_temperature(
    aces_transfer: ACESColorTransfer,
    source_image: np.ndarray,
    target_image: np.ndarray
) -> Dict:
    """Naprawienie nieprawidłowej temperatury barwowej."""
    
    # Estymacja temperatury barwowej
    source_temp = estimate_color_temperature(source_image)
    target_temp = estimate_color_temperature(target_image)
    
    temp_difference = abs(target_temp - source_temp)
    
    # Modyfikacja parametrów
    params = aces_transfer.params
    
    if temp_difference > 1000:  # Znacząca różnica temperatury
        # Włączenie adaptacji chromatycznej
        params.chromatic_adaptation.enabled = True
        params.chromatic_adaptation.method = "bradford"
        
        # Ograniczenie siły adaptacji dla naturalności
        params.chromatic_adaptation.strength = 0.8
        params.chromatic_adaptation.preserve_neutrals = True
        
        # Dodatkowa kontrola white balance
        params.white_balance.auto_correct = True
        params.white_balance.preserve_skin_tones = True
    
    # Transfer z korekcją temperatury
    result = aces_transfer.transfer_colors(source_image, target_image)
    
    # Post-processing: fine-tuning white balance
    if 'image' in result:
        result['image'] = fine_tune_white_balance(
            result['image'],
            target_temperature=source_temp,
            strength=0.5
        )
    
    result['fixes_applied'] = ['color_temperature_correction']
    result['metadata']['temperature_correction'] = {
        'source_temp': source_temp,
        'target_temp': target_temp,
        'difference': temp_difference
    }
    
    return result

def estimate_color_temperature(image: np.ndarray) -> float:
    """Estymacja temperatury barwowej obrazu."""
    
    # Konwersja do XYZ
    rgb_normalized = image.astype(np.float32) / 255.0
    
    # Macierz sRGB -> XYZ
    srgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    # Reshape i transformacja
    pixels = rgb_normalized.reshape(-1, 3)
    xyz_pixels = np.dot(pixels, srgb_to_xyz.T)
    
    # Średnie wartości XYZ
    mean_xyz = np.mean(xyz_pixels, axis=0)
    
    # Obliczenie chromatyczności
    xyz_sum = np.sum(mean_xyz)
    if xyz_sum > 0:
        x = mean_xyz[0] / xyz_sum
        y = mean_xyz[1] / xyz_sum
    else:
        return 6500  # Domyślna temperatura
    
    # Konwersja chromatyczności na temperaturę (McCamy's formula)
    n = (x - 0.3320) / (0.1858 - y)
    temp = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
    
    # Ograniczenie do rozsądnego zakresu
    return np.clip(temp, 2000, 12000)

def fine_tune_white_balance(
    image: np.ndarray,
    target_temperature: float,
    strength: float = 0.5
) -> np.ndarray:
    """Fine-tuning white balance."""
    
    current_temp = estimate_color_temperature(image)
    temp_diff = target_temperature - current_temp
    
    if abs(temp_diff) < 100:  # Mała różnica
        return image
    
    # Obliczenie korekcji
    if temp_diff > 0:  # Obraz za zimny, dodaj ciepła
        red_gain = 1.0 + (temp_diff / 10000) * strength
        blue_gain = 1.0 - (temp_diff / 15000) * strength
    else:  # Obraz za ciepły, dodaj chłodu
        red_gain = 1.0 + (temp_diff / 15000) * strength
        blue_gain = 1.0 - (temp_diff / 10000) * strength
    
    # Aplikacja korekcji
    corrected = image.astype(np.float32)
    corrected[:, :, 0] *= red_gain   # Czerwony
    corrected[:, :, 2] *= blue_gain  # Niebieski
    
    return np.clip(corrected, 0, 255).astype(np.uint8)
```

---

## 2. Narzędzie Debugowania

### 2.1 Kompleksowy Debugger ACES

```python
import matplotlib.pyplot as plt
from typing import List, Optional
import seaborn as sns

class ACESDebugger:
    """Narzędzie do debugowania transferu kolorów ACES."""
    
    def __init__(self):
        self.debug_data = {}
        self.visualization_enabled = True
    
    def debug_aces_transfer(
        self,
        aces_transfer: ACESColorTransfer,
        source_image: np.ndarray,
        target_image: np.ndarray,
        save_path: Optional[str] = None
    ) -> Dict:
        """Kompleksowe debugowanie transferu ACES."""
        
        debug_results = {
            'input_analysis': {},
            'processing_steps': {},
            'output_analysis': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Analiza wejściowa
        debug_results['input_analysis'] = self._analyze_input_images(
            source_image, target_image
        )
        
        # Śledzenie kroków przetwarzania
        debug_results['processing_steps'] = self._trace_processing_steps(
            aces_transfer, source_image, target_image
        )
        
        # Transfer z debugowaniem
        result = aces_transfer.transfer_colors(source_image, target_image)
        
        if 'image' in result:
            # Analiza wyjściowa
            debug_results['output_analysis'] = self._analyze_output_image(
                source_image, target_image, result['image']
            )
            
            # Metryki jakości
            debug_results['quality_metrics'] = self._calculate_quality_metrics(
                source_image, target_image, result['image']
            )
        
        # Generowanie rekomendacji
        debug_results['recommendations'] = self._generate_recommendations(
            debug_results
        )
        
        # Wizualizacja (jeśli włączona)
        if self.visualization_enabled:
            self._create_debug_visualization(
                source_image, target_image, result.get('image'),
                debug_results, save_path
            )
        
        self.debug_data = debug_results
        return debug_results
    
    def _analyze_input_images(
        self, 
        source_image: np.ndarray, 
        target_image: np.ndarray
    ) -> Dict:
        """Analiza obrazów wejściowych."""
        
        analysis = {
            'source': self._analyze_single_image(source_image, 'source'),
            'target': self._analyze_single_image(target_image, 'target'),
            'compatibility': {}
        }
        
        # Analiza kompatybilności
        source_stats = analysis['source']
        target_stats = analysis['target']
        
        # Różnice w rozmiarze
        size_ratio = (
            source_stats['dimensions'][0] * source_stats['dimensions'][1]
        ) / (
            target_stats['dimensions'][0] * target_stats['dimensions'][1]
        )
        
        analysis['compatibility'] = {
            'size_ratio': size_ratio,
            'size_compatible': 0.5 <= size_ratio <= 2.0,
            'dynamic_range_ratio': (
                source_stats['dynamic_range'] / max(target_stats['dynamic_range'], 0.01)
            ),
            'color_space_similarity': self._calculate_color_space_similarity(
                source_image, target_image
            ),
            'recommended_method': self._recommend_transfer_method(
                source_stats, target_stats
            )
        }
        
        return analysis
    
    def _analyze_single_image(self, image: np.ndarray, name: str) -> Dict:
        """Analiza pojedynczego obrazu."""
        
        # Podstawowe informacje
        stats = {
            'name': name,
            'dimensions': image.shape[:2],
            'channels': image.shape[2] if len(image.shape) > 2 else 1,
            'dtype': str(image.dtype),
            'size_mb': image.nbytes / 1024 / 1024
        }
        
        # Statystyki kolorów
        if len(image.shape) == 3:
            # RGB statistics
            for i, channel in enumerate(['red', 'green', 'blue']):
                channel_data = image[:, :, i]
                stats[f'{channel}_stats'] = {
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'min': int(np.min(channel_data)),
                    'max': int(np.max(channel_data)),
                    'median': float(np.median(channel_data))
                }
        
        # Analiza luminancji
        if len(image.shape) == 3:
            # Konwersja do grayscale (luminancja)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            stats['luminance'] = {
                'mean': float(np.mean(gray)),
                'std': float(np.std(gray)),
                'dynamic_range': float(np.max(gray) - np.min(gray)),
                'contrast': float(np.std(gray) / max(np.mean(gray), 1))
            }
        
        # Analiza histogramów
        stats['histograms'] = {}
        if len(image.shape) == 3:
            for i, channel in enumerate(['red', 'green', 'blue']):
                hist, _ = np.histogram(image[:, :, i], bins=256, range=(0, 256))
                stats['histograms'][channel] = hist.tolist()
        
        # Analiza temperatury barwowej
        if len(image.shape) == 3:
            stats['color_temperature'] = estimate_color_temperature(image)
        
        # Analiza saturacji
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            stats['saturation'] = {
                'mean': float(np.mean(hsv[:, :, 1])),
                'std': float(np.std(hsv[:, :, 1])),
                'max': int(np.max(hsv[:, :, 1]))
            }
        
        return stats
    
    def _calculate_color_space_similarity(
        self, 
        source_image: np.ndarray, 
        target_image: np.ndarray
    ) -> float:
        """Obliczenie podobieństwa przestrzeni kolorów."""
        
        # Porównanie rozkładów kolorów
        source_hist = []
        target_hist = []
        
        for i in range(3):  # RGB
            s_hist, _ = np.histogram(source_image[:, :, i], bins=64, range=(0, 256))
            t_hist, _ = np.histogram(target_image[:, :, i], bins=64, range=(0, 256))
            
            # Normalizacja
            s_hist = s_hist / np.sum(s_hist)
            t_hist = t_hist / np.sum(t_hist)
            
            source_hist.extend(s_hist)
            target_hist.extend(t_hist)
        
        # Korelacja histogramów
        correlation = np.corrcoef(source_hist, target_hist)[0, 1]
        
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0
    
    def _recommend_transfer_method(
        self, 
        source_stats: Dict, 
        target_stats: Dict
    ) -> str:
        """Rekomendacja metody transferu na podstawie analizy."""
        
        # Analiza różnic
        luminance_diff = abs(
            source_stats['luminance']['mean'] - target_stats['luminance']['mean']
        )
        
        contrast_diff = abs(
            source_stats['luminance']['contrast'] - target_stats['luminance']['contrast']
        )
        
        temp_diff = abs(
            source_stats.get('color_temperature', 6500) - 
            target_stats.get('color_temperature', 6500)
        )
        
        # Logika rekomendacji
        if temp_diff > 1500:
            return "chromatic_adaptation"
        elif contrast_diff > 0.3:
            return "histogram_matching"
        elif luminance_diff > 50:
            return "statistical_matching"
        else:
            return "perceptual_matching"
    
    def _trace_processing_steps(
        self,
        aces_transfer: ACESColorTransfer,
        source_image: np.ndarray,
        target_image: np.ndarray
    ) -> Dict:
        """Śledzenie kroków przetwarzania."""
        
        steps = {}
        
        # Symulacja kroków (w rzeczywistej implementacji byłoby to zintegrowane)
        try:
            # Krok 1: Konwersja do ACES
            source_aces = aces_transfer._convert_to_aces(source_image)
            target_aces = aces_transfer._convert_to_aces(target_image)
            
            steps['conversion_to_aces'] = {
                'success': True,
                'source_range': [float(np.min(source_aces)), float(np.max(source_aces))],
                'target_range': [float(np.min(target_aces)), float(np.max(target_aces))]
            }
            
            # Krok 2: Analiza statystyk
            source_stats = aces_transfer._analyze_statistics(source_aces)
            target_stats = aces_transfer._analyze_statistics(target_aces)
            
            steps['statistics_analysis'] = {
                'success': True,
                'source_mean': source_stats['mean'].tolist(),
                'target_mean': target_stats['mean'].tolist(),
                'source_std': source_stats['std'].tolist(),
                'target_std': target_stats['std'].tolist()
            }
            
            # Krok 3: Obliczenie transformacji
            transform_data = aces_transfer._calculate_transformation(
                source_stats, target_stats
            )
            
            steps['transformation_calculation'] = {
                'success': True,
                'method': transform_data.get('method', 'unknown'),
                'parameters': {k: v for k, v in transform_data.items() 
                             if k != 'method' and isinstance(v, (int, float, list))}
            }
            
        except Exception as e:
            steps['error'] = {
                'message': str(e),
                'step': 'processing_trace'
            }
        
        return steps
    
    def _analyze_output_image(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        output_image: np.ndarray
    ) -> Dict:
        """Analiza obrazu wyjściowego."""
        
        analysis = {
            'basic_stats': self._analyze_single_image(output_image, 'output'),
            'transfer_effectiveness': {},
            'artifacts_detection': {}
        }
        
        # Efektywność transferu
        source_temp = estimate_color_temperature(source_image)
        target_temp = estimate_color_temperature(target_image)
        output_temp = estimate_color_temperature(output_image)
        
        temp_improvement = abs(output_temp - target_temp) < abs(source_temp - target_temp)
        
        analysis['transfer_effectiveness'] = {
            'temperature_improvement': temp_improvement,
            'source_temp': source_temp,
            'target_temp': target_temp,
            'output_temp': output_temp,
            'color_shift': self._calculate_color_shift(source_image, output_image)
        }
        
        # Detekcja artefaktów
        analysis['artifacts_detection'] = self._detect_artifacts(output_image)
        
        return analysis
    
    def _calculate_color_shift(self, source: np.ndarray, output: np.ndarray) -> Dict:
        """Obliczenie przesunięcia kolorów."""
        
        # Średnie wartości kanałów
        source_mean = np.mean(source.reshape(-1, 3), axis=0)
        output_mean = np.mean(output.reshape(-1, 3), axis=0)
        
        shift = output_mean - source_mean
        
        return {
            'red_shift': float(shift[0]),
            'green_shift': float(shift[1]),
            'blue_shift': float(shift[2]),
            'magnitude': float(np.linalg.norm(shift))
        }
    
    def _detect_artifacts(self, image: np.ndarray) -> Dict:
        """Detekcja artefaktów w obrazie."""
        
        artifacts = {
            'clipping': {},
            'banding': False,
            'noise_level': 0.0,
            'oversaturation': False
        }
        
        # Detekcja clippingu
        for i, channel in enumerate(['red', 'green', 'blue']):
            channel_data = image[:, :, i]
            clipped_low = np.sum(channel_data == 0)
            clipped_high = np.sum(channel_data == 255)
            total_pixels = channel_data.size
            
            artifacts['clipping'][channel] = {
                'low_percent': (clipped_low / total_pixels) * 100,
                'high_percent': (clipped_high / total_pixels) * 100
            }
        
        # Detekcja bandingu (uproszczona)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grad = np.gradient(gray.astype(np.float32))
        grad_magnitude = np.sqrt(grad[0]**2 + grad[1]**2)
        
        # Banding charakteryzuje się nagłymi zmianami gradientu
        artifacts['banding'] = np.std(grad_magnitude) > 10.0
        
        # Poziom szumu
        artifacts['noise_level'] = float(np.std(grad_magnitude))
        
        # Oversaturacja
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        high_saturation_pixels = np.sum(hsv[:, :, 1] > 240)
        artifacts['oversaturation'] = (high_saturation_pixels / hsv[:, :, 1].size) > 0.1
        
        return artifacts
    
    def _calculate_quality_metrics(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        output_image: np.ndarray
    ) -> Dict:
        """Obliczenie metryk jakości."""
        
        metrics = {}
        
        # PSNR między source a output
        mse = np.mean((source_image.astype(np.float32) - output_image.astype(np.float32))**2)
        if mse > 0:
            metrics['psnr_source_output'] = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            metrics['psnr_source_output'] = float('inf')
        
        # Structural Similarity (uproszczona)
        metrics['ssim_approximation'] = self._calculate_ssim_approximation(
            source_image, output_image
        )
        
        # Color fidelity
        metrics['color_fidelity'] = self._calculate_color_fidelity(
            target_image, output_image
        )
        
        # Overall quality score
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _calculate_ssim_approximation(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray
    ) -> float:
        """Uproszczona wersja SSIM."""
        
        # Konwersja do grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Podstawowe statystyki
        mu1 = np.mean(gray1)
        mu2 = np.mean(gray2)
        
        sigma1 = np.std(gray1)
        sigma2 = np.std(gray2)
        
        sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
        
        # Stałe SSIM
        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2
        
        # SSIM formula
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
        
        return numerator / max(denominator, 1e-10)
    
    def _calculate_color_fidelity(
        self, 
        target: np.ndarray, 
        output: np.ndarray
    ) -> float:
        """Obliczenie wierności kolorów."""
        
        # Delta E w przestrzeni LAB (uproszczona)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        output_lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
        
        # Różnice w każdym kanale
        delta_l = target_lab[:, :, 0].astype(np.float32) - output_lab[:, :, 0].astype(np.float32)
        delta_a = target_lab[:, :, 1].astype(np.float32) - output_lab[:, :, 1].astype(np.float32)
        delta_b = target_lab[:, :, 2].astype(np.float32) - output_lab[:, :, 2].astype(np.float32)
        
        # Delta E
        delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
        mean_delta_e = np.mean(delta_e)
        
        # Konwersja na score (0-1, gdzie 1 = idealne)
        return max(0.0, 1.0 - (mean_delta_e / 100.0))
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Obliczenie ogólnego wyniku jakości."""
        
        # Wagi dla różnych metryk
        weights = {
            'ssim_approximation': 0.4,
            'color_fidelity': 0.6
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        return score / max(total_weight, 1.0)
    
    def _generate_recommendations(self, debug_results: Dict) -> List[str]:
        """Generowanie rekomendacji na podstawie analizy."""
        
        recommendations = []
        
        # Analiza kompatybilności
        compatibility = debug_results['input_analysis']['compatibility']
        
        if not compatibility['size_compatible']:
            recommendations.append(
                "Rozważ zmianę rozmiaru obrazów dla lepszej kompatybilności"
            )
        
        if compatibility['color_space_similarity'] < 0.3:
            recommendations.append(
                "Obrazy mają bardzo różne przestrzenie kolorów - rozważ preprocessing"
            )
        
        # Analiza artefaktów
        if 'artifacts_detection' in debug_results.get('output_analysis', {}):
            artifacts = debug_results['output_analysis']['artifacts_detection']
            
            # Clipping
            for channel, clipping in artifacts['clipping'].items():
                if clipping['high_percent'] > 5.0:
                    recommendations.append(
                        f"Wysokie clipping w kanale {channel} ({clipping['high_percent']:.1f}%) - "
                        "zmniejsz siłę transformacji"
                    )
                
                if clipping['low_percent'] > 5.0:
                    recommendations.append(
                        f"Clipping cieni w kanale {channel} ({clipping['low_percent']:.1f}%) - "
                        "włącz ochronę cieni"
                    )
            
            # Oversaturacja
            if artifacts['oversaturation']:
                recommendations.append(
                    "Wykryto oversaturację - włącz kontrolę saturacji"
                )
            
            # Banding
            if artifacts['banding']:
                recommendations.append(
                    "Wykryto banding - użyj wyższej precyzji lub dithering"
                )
        
        # Analiza jakości
        if 'quality_metrics' in debug_results:
            quality = debug_results['quality_metrics']
            
            if quality['overall_score'] < 0.7:
                recommendations.append(
                    "Niska jakość transferu - spróbuj innej metody lub dostosuj parametry"
                )
            
            if quality['color_fidelity'] < 0.6:
                recommendations.append(
                    "Niska wierność kolorów - włącz chromatic adaptation"
                )
        
        # Rekomendacje metody
        recommended_method = compatibility['recommended_method']
        recommendations.append(
            f"Rekomendowana metoda: {recommended_method}"
        )
        
        return recommendations
    
    def _create_debug_visualization(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        output_image: Optional[np.ndarray],
        debug_results: Dict,
        save_path: Optional[str] = None
    ) -> None:
        """Tworzenie wizualizacji debugowania."""
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('ACES Color Transfer - Debug Analysis', fontsize=16)
        
        # Obrazy
        axes[0, 0].imshow(source_image)
        axes[0, 0].set_title('Source Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(target_image)
        axes[0, 1].set_title('Target Image')
        axes[0, 1].axis('off')
        
        if output_image is not None:
            axes[0, 2].imshow(output_image)
            axes[0, 2].set_title('Output Image')
            axes[0, 2].axis('off')
        
        # Histogramy
        for i, (img, title) in enumerate([
            (source_image, 'Source'), 
            (target_image, 'Target'),
            (output_image, 'Output')
        ]):
            if img is not None and i < 3:
                for ch, color in enumerate(['red', 'green', 'blue']):
                    hist, bins = np.histogram(img[:, :, ch], bins=256, range=(0, 256))
                    axes[1, i].plot(bins[:-1], hist, color=color, alpha=0.7)
                axes[1, i].set_title(f'{title} Histogram')
                axes[1, i].set_xlim(0, 255)
        
        # Metryki jakości
        if 'quality_metrics' in debug_results:
            metrics = debug_results['quality_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            axes[2, 0].bar(range(len(metric_names)), metric_values)
            axes[2, 0].set_xticks(range(len(metric_names)))
            axes[2, 0].set_xticklabels(metric_names, rotation=45, ha='right')
            axes[2, 0].set_title('Quality Metrics')
            axes[2, 0].set_ylim(0, 1)
        
        # Rekomendacje
        if 'recommendations' in debug_results:
            recommendations = debug_results['recommendations'][:5]  # Top 5
            axes[2, 1].text(0.05, 0.95, '\n'.join(recommendations), 
                           transform=axes[2, 1].transAxes, 
                           verticalalignment='top',
                           fontsize=8, wrap=True)
            axes[2, 1].set_title('Recommendations')
            axes[2, 1].axis('off')
        
        # Usunięcie pustych subplotów
        for i in range(3):
            for j in range(4):
                if i == 0 and j == 3:
                    axes[i, j].axis('off')
                elif i == 2 and j >= 2:
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_debug_report(self, filename: str) -> None:
        """Eksport raportu debugowania do pliku."""
        
        if not self.debug_data:
            print("Brak danych debugowania. Uruchom debug_aces_transfer() najpierw.")
            return
        
        import json
        
        # Konwersja numpy arrays na listy dla JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        debug_data_json = convert_numpy(self.debug_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(debug_data_json, f, indent=2, ensure_ascii=False)
        
        print(f"Raport debugowania zapisany do: {filename}")

# Funkcja pomocnicza dla łatwego debugowania
def debug_aces_transfer(
    source_image: np.ndarray,
    target_image: np.ndarray,
    params: Optional[ACESParameters] = None,
    save_visualization: Optional[str] = None,
    save_report: Optional[str] = None
) -> Dict:
    """Funkcja pomocnicza do debugowania transferu ACES."""
    
    if params is None:
        params = ACESParameters()
    
    # Tworzenie instancji
    aces_transfer = ACESColorTransfer(params)
    debugger = ACESDebugger()
    
    # Debugowanie
    debug_results = debugger.debug_aces_transfer(
        aces_transfer, source_image, target_image, save_visualization
    )
    
    # Eksport raportu
    if save_report:
        debugger.export_debug_report(save_report)
    
    return debug_results
```

---

## 3. Przyszłe Ulepszenia

### 3.1 Planowane Funkcje

```python
class ACESFutureEnhancements:
    """Planowane ulepszenia dla ACES Color Transfer."""
    
    def __init__(self):
        self.roadmap_2024 = {
            'Q1': [
                'Adaptive tone mapping',
                'Machine learning enhancement',
                'Real-time processing optimization'
            ],
            'Q2': [
                'Advanced color science integration',
                'Multi-scale processing',
                'HDR support'
            ],
            'Q3': [
                'Neural network acceleration',
                'Cloud processing integration',
                'Mobile optimization'
            ],
            'Q4': [
                'VR/AR color matching',
                'Video processing pipeline',
                'Professional workflow integration'
            ]
        }
    
    def adaptive_tone_mapping(self, image: np.ndarray) -> np.ndarray:
        """Adaptacyjne tone mapping (prototyp)."""
        
        # Analiza lokalnego kontrastu
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale analysis
        scales = [1, 2, 4, 8]
        local_contrast_maps = []
        
        for scale in scales:
            kernel_size = 2 * scale + 1
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), scale)
            contrast_map = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
            local_contrast_maps.append(contrast_map)
        
        # Kombinacja map kontrastu
        combined_contrast = np.mean(local_contrast_maps, axis=0)
        
        # Adaptacyjne parametry tone mapping
        adaptive_params = self._calculate_adaptive_parameters(combined_contrast)
        
        # Aplikacja adaptacyjnego tone mapping
        result = self._apply_adaptive_tone_mapping(image, adaptive_params)
        
        return result
    
    def _calculate_adaptive_parameters(self, contrast_map: np.ndarray) -> Dict:
        """Obliczenie adaptacyjnych parametrów."""
        
        # Analiza rozkładu kontrastu
        contrast_percentiles = np.percentile(contrast_map, [25, 50, 75, 95])
        
        # Adaptacyjne parametry na podstawie lokalnego kontrastu
        params = {
            'low_contrast_boost': 1.0 + (50 - contrast_percentiles[0]) / 100,
            'mid_contrast_preserve': 1.0,
            'high_contrast_compress': 1.0 - (contrast_percentiles[3] - 100) / 200
        }
        
        # Ograniczenia
        params['low_contrast_boost'] = np.clip(params['low_contrast_boost'], 0.8, 1.5)
        params['high_contrast_compress'] = np.clip(params['high_contrast_compress'], 0.5, 1.0)
        
        return params
    
    def _apply_adaptive_tone_mapping(
        self, 
        image: np.ndarray, 
        params: Dict
    ) -> np.ndarray:
        """Aplikacja adaptacyjnego tone mapping."""
        
        # Konwersja do LAB dla lepszej kontroli luminancji
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = lab[:, :, 0] / 255.0
        
        # Maska dla różnych zakresów luminancji
        low_mask = l_channel < 0.3
        mid_mask = (l_channel >= 0.3) & (l_channel <= 0.7)
        high_mask = l_channel > 0.7
        
        # Adaptacyjne przetwarzanie
        l_processed = l_channel.copy()
        
        # Boost dla niskich luminancji
        l_processed[low_mask] *= params['low_contrast_boost']
        
        # Kompresja dla wysokich luminancji
        l_processed[high_mask] = (
            l_processed[high_mask] * params['high_contrast_compress'] + 
            (1 - params['high_contrast_compress']) * 0.8
        )
        
        # Clamp i konwersja z powrotem
        l_processed = np.clip(l_processed, 0, 1)
        lab[:, :, 0] = l_processed * 255.0
        
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    def ml_enhancement_prototype(self, image: np.ndarray) -> np.ndarray:
        """Prototyp ulepszenia ML (symulacja)."""
        
        # Symulacja sieci neuronowej dla ulepszenia kolorów
        # W rzeczywistości byłaby to wytrenowana sieć
        
        # Feature extraction (uproszczona)
        features = self._extract_color_features(image)
        
        # Symulacja predykcji sieci
        enhancement_map = self._simulate_ml_prediction(features)
        
        # Aplikacja ulepszenia
        enhanced = self._apply_ml_enhancement(image, enhancement_map)
        
        return enhanced
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Ekstrakcja cech kolorów (symulacja)."""
        
        # Konwersja do różnych przestrzeni kolorów
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Podstawowe cechy
        features = np.stack([
            image[:, :, 0],  # R
            image[:, :, 1],  # G
            image[:, :, 2],  # B
            hsv[:, :, 0],    # H
            hsv[:, :, 1],    # S
            hsv[:, :, 2],    # V
            lab[:, :, 0],    # L
            lab[:, :, 1],    # a
            lab[:, :, 2],    # b
        ], axis=-1)
        
        return features
    
    def _simulate_ml_prediction(self, features: np.ndarray) -> np.ndarray:
        """Symulacja predykcji ML."""
        
        # Symulacja prostej sieci - w rzeczywistości byłaby to wytrenowana sieć
        # Tutaj używamy prostych heurystyk
        
        height, width, channels = features.shape
        enhancement_map = np.zeros((height, width, 3))
        
        # Symulacja: boost dla undersaturowanych obszarów
        saturation = features[:, :, 4] / 255.0
        low_sat_mask = saturation < 0.3
        
        enhancement_map[low_sat_mask, :] = [1.1, 1.05, 1.0]  # Boost R, G
        enhancement_map[~low_sat_mask, :] = [1.0, 1.0, 1.0]  # Bez zmian
        
        return enhancement_map
    
    def _apply_ml_enhancement(
        self, 
        image: np.ndarray, 
        enhancement_map: np.ndarray
    ) -> np.ndarray:
        """Aplikacja ulepszenia ML."""
        
        enhanced = image.astype(np.float32) * enhancement_map
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def real_time_processing_optimization(self) -> Dict:
        """Optymalizacje dla przetwarzania real-time."""
        
        optimizations = {
            'reduced_precision': {
                'description': 'Użycie float16 zamiast float32',
                'speedup': '1.5-2x',
                'quality_loss': 'Minimalna'
            },
            'lookup_tables': {
                'description': 'Pre-computed LUTs dla transformacji',
                'speedup': '3-5x',
                'quality_loss': 'Niska'
            },
            'spatial_downsampling': {
                'description': 'Przetwarzanie w niższej rozdzielczości',
                'speedup': '4-16x',
                'quality_loss': 'Średnia'
            },
            'temporal_coherence': {
                'description': 'Wykorzystanie poprzednich klatek',
                'speedup': '2-3x',
                'quality_loss': 'Niska'
            }
        }
        
        return optimizations
    
    def generate_development_roadmap(self) -> str:
        """Generowanie mapy rozwoju."""
        
        roadmap = ["# ACES Color Transfer - Mapa Rozwoju 2024\n"]
        
        for quarter, features in self.roadmap_2024.items():
            roadmap.append(f"## {quarter} 2024\n")
            
            for feature in features:
                roadmap.append(f"- [ ] {feature}")
                
                # Dodanie szczegółów dla niektórych funkcji
                if feature == 'Adaptive tone mapping':
                    roadmap.append("  - Analiza lokalnego kontrastu")
                    roadmap.append("  - Multi-scale processing")
                    roadmap.append("  - Automatyczne dostosowanie parametrów")
                
                elif feature == 'Machine learning enhancement':
                    roadmap.append("  - Trening sieci na dużym zbiorze danych")
                    roadmap.append("  - Integracja z TensorFlow/PyTorch")
                    roadmap.append("  - Optymalizacja dla inference")
                
                elif feature == 'Real-time processing optimization':
                    roadmap.append("  - GPU acceleration")
                    roadmap.append("  - Memory optimization")
                    roadmap.append("  - Parallel processing")
            
            roadmap.append("")
        
        # Dodanie metryk sukcesu
        roadmap.append("## Metryki Sukcesu\n")
        roadmap.append("- Czas przetwarzania < 100ms dla 1080p")
        roadmap.append("- Jakość SSIM > 0.95")
        roadmap.append("- Użycie pamięci < 500MB")
        roadmap.append("- Wsparcie dla 95% przypadków użycia")
        
        return "\n".join(roadmap)
```

---

## 4. Podsumowanie i Wnioski

### 4.1 Kluczowe Punkty

1. **Diagnostyka Problemów**: Systematyczne podejście do identyfikacji i rozwiązywania typowych problemów
2. **Narzędzia Debugowania**: Kompleksowy system analizy i wizualizacji
3. **Przyszły Rozwój**: Jasna mapa rozwoju z naciskiem na ML i real-time processing

### 4.2 Najlepsze Praktyki

- Zawsze używaj narzędzi debugowania dla nowych przypadków
- Monitoruj metryki jakości
- Dostosowuj parametry na podstawie analizy obrazu
- Planuj optymalizacje na podstawie rzeczywistych potrzeb

### 4.3 Kontakt i Wsparcie

- **Dokumentacja**: [Spis treści](gatto-WORKING-03-algorithms-08-advanced-01-aces-0of6.md)
- **Issues**: Zgłaszaj problemy z szczegółowym opisem
- **Contributions**: Wkład w rozwój jest mile widziany

---

**Koniec serii ACES Color Space Transfer**  
**Powrót do:** [Spis treści](gatto-WORKING-03-algorithms-08-advanced-01-aces-0of6.md)  
**Poprzednia część:** [5of6 - Analiza Wydajności](gatto-WORKING-03-algorithms-08-advanced-01-aces-5of6.md)

*Część 6of6 - Aplikacje Praktyczne i Rozwiązywanie Problemów | Wersja 1.0 | 2024-01-20*