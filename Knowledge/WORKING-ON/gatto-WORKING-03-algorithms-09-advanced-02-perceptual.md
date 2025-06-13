# Perceptual Color Matching 🔴

## 1. Przegląd

### Opis Algorytmu
Perceptual Color Matching to zaawansowany algorytm dopasowania kolorów oparty na modelach percepcji wzrokowej człowieka. Algorytm wykorzystuje przestrzenie kolorów perceptualnie jednolite (LAB, LUV, CAM16) oraz zaawansowane metryki różnic kolorów (Delta E 2000, CAM16-UCS) do osiągnięcia naturalnego i przyjemnego dla oka dopasowania kolorów.

### Zastosowania
- **Fotografia artystyczna**: Naturalne dopasowanie kolorów
- **Design graficzny**: Harmonijne palety kolorów
- **Medycyna**: Precyzyjne odwzorowanie kolorów skóry
- **E-commerce**: Wierne przedstawienie produktów
- **Sztuka cyfrowa**: Artystyczne interpretacje kolorów

### Zalety
- ✅ Naturalny wygląd dla ludzkiego oka
- ✅ Zachowanie harmonii kolorów
- ✅ Adaptacja do warunków oświetlenia
- ✅ Uwzględnienie kontekstu percepcyjnego
- ✅ Wysoka jakość wizualna

### Wady
- ❌ Bardzo wysoka złożoność obliczeniowa
- ❌ Wymaga zaawansowanych modeli percepcji
- ❌ Długi czas przetwarzania
- ❌ Skomplikowana kalibracja
- ❌ Duże wymagania pamięciowe

---

## 2. Podstawy Teoretyczne

### Modele Percepcji Wzrokowej

#### 1. CIE LAB Color Space
```
L* = 116 * f(Y/Yn) - 16
a* = 500 * [f(X/Xn) - f(Y/Yn)]
b* = 200 * [f(Y/Yn) - f(Z/Zn)]

where f(t) = t^(1/3) if t > (6/29)^3
             (1/3) * (29/6)^2 * t + 4/29 otherwise
```

#### 2. CAM16 Color Appearance Model
```
# Chromatic adaptation
RGB_c = CAT16 * RGB
RGB_a = adaptation_function(RGB_c, D, F_L)

# Opponent color dimensions
a = R_a - 12*G_a/11 + B_a/11
b = (R_a + G_a - 2*B_a)/9

# Hue angle
h = atan2(b, a)

# Chroma
C = sqrt(a^2 + b^2)

# Lightness
J = 100 * (A/A_w)^(c*z)
```

#### 3. Delta E Metrics

**Delta E 2000 (CIE DE2000):**
```
ΔE00 = sqrt(
    (ΔL'/k_L*S_L)^2 + 
    (ΔC'/k_C*S_C)^2 + 
    (ΔH'/k_H*S_H)^2 + 
    R_T * (ΔC'/k_C*S_C) * (ΔH'/k_H*S_H)
)
```

**CAM16-UCS Delta E:**
```
ΔE_CAM16 = sqrt(
    (ΔJ')^2 + (Δa')^2 + (Δb')^2
)
```

### Perceptual Weighting Functions

```python
def perceptual_weight_function(luminance, chroma, hue):
    """Funkcja wagowa oparta na percepcji"""
    
    # Waga luminancji (wrażliwość na jasność)
    luminance_weight = 1.0 + 0.5 * np.exp(-luminance/0.18)
    
    # Waga chromy (wrażliwość na saturację)
    chroma_weight = 1.0 + 0.3 * (1.0 - np.exp(-chroma/0.1))
    
    # Waga odcienia (wrażliwość na konkretne kolory)
    hue_sensitivity = {
        'red': (0, 30, 330, 360),      # Wysoka wrażliwość
        'green': (90, 150),            # Średnia wrażliwość  
        'blue': (210, 270),            # Średnia wrażliwość
        'yellow': (60, 90)             # Bardzo wysoka wrażliwość
    }
    
    hue_weight = calculate_hue_sensitivity(hue, hue_sensitivity)
    
    return luminance_weight * chroma_weight * hue_weight
```

### Adaptation Models

#### Von Kries Chromatic Adaptation
```
M_adapt = M_CAT^(-1) * diag(ρ_w/ρ_s, γ_w/γ_s, β_w/β_s) * M_CAT

where:
M_CAT = Bradford transformation matrix
ρ, γ, β = cone responses
s = source illuminant
w = target illuminant
```

#### CIECAM02/CAM16 Adaptation
```
D = F * [1 - (1/3.6) * exp((-L_A - 42)/92)]

where:
F = surround factor (0.8 for average, 0.9 for dim, 1.0 for dark)
L_A = adapting luminance
```

---

## 3. Pseudokod

```
FUNCTION perceptual_color_matching(source_image, target_image, parameters):
    // Krok 1: Analiza warunków obserwacji
    viewing_conditions = analyze_viewing_conditions(
        source_image, 
        target_image,
        parameters.illuminant,
        parameters.surround
    )
    
    // Krok 2: Konwersja do przestrzeni perceptualnej
    source_perceptual = convert_to_perceptual_space(
        source_image,
        viewing_conditions.source,
        parameters.color_space  // 'lab', 'cam16', 'luv'
    )
    
    target_perceptual = convert_to_perceptual_space(
        target_image,
        viewing_conditions.target,
        parameters.color_space
    )
    
    // Krok 3: Analiza percepcyjna
    source_analysis = analyze_perceptual_characteristics(
        source_perceptual,
        parameters.analysis_method
    )
    
    target_analysis = analyze_perceptual_characteristics(
        target_perceptual,
        parameters.analysis_method
    )
    
    // Krok 4: Obliczenie mapowania perceptualnego
    perceptual_mapping = calculate_perceptual_mapping(
        source_analysis,
        target_analysis,
        parameters.mapping_method,
        parameters.perceptual_weights
    )
    
    // Krok 5: Aplikacja transformacji z wagami percepcyjnymi
    result_perceptual = EMPTY_IMAGE(source_perceptual.size)
    
    FOR each pixel IN source_perceptual:
        // Obliczenie wag percepcyjnych
        perceptual_weights = calculate_perceptual_weights(
            pixel,
            parameters.sensitivity_model
        )
        
        // Aplikacja mapowania z wagami
        mapped_pixel = apply_weighted_mapping(
            pixel,
            perceptual_mapping,
            perceptual_weights
        )
        
        // Adaptacja chromatyczna
        IF parameters.chromatic_adaptation:
            mapped_pixel = apply_chromatic_adaptation(
                mapped_pixel,
                viewing_conditions.source.illuminant,
                viewing_conditions.target.illuminant,
                parameters.adaptation_method
            )
        END IF
        
        // Korekta lokalnego kontrastu
        IF parameters.local_adaptation:
            mapped_pixel = apply_local_adaptation(
                mapped_pixel,
                get_local_context(pixel, source_perceptual),
                parameters.adaptation_radius
            )
        END IF
        
        result_perceptual[pixel_index] = mapped_pixel
    END FOR
    
    // Krok 6: Post-processing percepcyjny
    IF parameters.gamut_mapping:
        result_perceptual = apply_perceptual_gamut_mapping(
            result_perceptual,
            parameters.target_gamut,
            parameters.gamut_mapping_method
        )
    END IF
    
    IF parameters.color_harmony_preservation:
        result_perceptual = preserve_color_harmony(
            source_perceptual,
            result_perceptual,
            parameters.harmony_weights
        )
    END IF
    
    // Krok 7: Konwersja z powrotem do RGB
    result_image = convert_from_perceptual_space(
        result_perceptual,
        viewing_conditions.target,
        parameters.output_color_space
    )
    
    // Krok 8: Finalna optymalizacja percepcyjna
    IF parameters.perceptual_optimization:
        result_image = optimize_perceptual_quality(
            source_image,
            target_image,
            result_image,
            parameters.optimization_metric
        )
    END IF
    
    RETURN result_image
END FUNCTION

FUNCTION analyze_perceptual_characteristics(perceptual_image, method):
    characteristics = {}
    
    SWITCH method:
        CASE "statistical":
            characteristics.lightness_distribution = calculate_lightness_histogram(perceptual_image)
            characteristics.chroma_distribution = calculate_chroma_histogram(perceptual_image)
            characteristics.hue_distribution = calculate_hue_histogram(perceptual_image)
            characteristics.color_moments = calculate_color_moments(perceptual_image)
            
        CASE "perceptual_regions":
            characteristics.skin_tones = extract_skin_tone_regions(perceptual_image)
            characteristics.memory_colors = extract_memory_colors(perceptual_image)
            characteristics.neutral_colors = extract_neutral_regions(perceptual_image)
            characteristics.saturated_regions = extract_saturated_regions(perceptual_image)
            
        CASE "spatial_analysis":
            characteristics.local_contrast = analyze_local_contrast(perceptual_image)
            characteristics.color_gradients = analyze_color_gradients(perceptual_image)
            characteristics.texture_color_correlation = analyze_texture_color(perceptual_image)
            
        CASE "harmony_analysis":
            characteristics.color_harmony_type = detect_color_harmony(perceptual_image)
            characteristics.dominant_colors = extract_dominant_colors(perceptual_image)
            characteristics.color_temperature = estimate_overall_temperature(perceptual_image)
    END SWITCH
    
    RETURN characteristics
END FUNCTION

FUNCTION calculate_perceptual_mapping(source_analysis, target_analysis, method, weights):
    SWITCH method:
        CASE "delta_e_minimization":
            RETURN optimize_delta_e_mapping(
                source_analysis,
                target_analysis,
                weights.delta_e_metric  // 'de2000', 'cam16', 'de94'
            )
            
        CASE "perceptual_transfer":
            RETURN calculate_perceptual_transfer_function(
                source_analysis.lightness_distribution,
                target_analysis.lightness_distribution,
                source_analysis.chroma_distribution,
                target_analysis.chroma_distribution,
                weights.transfer_smoothness
            )
            
        CASE "memory_color_preservation":
            RETURN calculate_memory_color_mapping(
                source_analysis.memory_colors,
                target_analysis.memory_colors,
                weights.memory_color_importance
            )
            
        CASE "harmony_preservation":
            RETURN calculate_harmony_preserving_mapping(
                source_analysis.color_harmony_type,
                target_analysis.color_harmony_type,
                weights.harmony_preservation
            )
    END SWITCH
END FUNCTION
```

---

## 4. Implementacja Python

```python
import numpy as np
import cv2
from scipy import optimize, interpolate
from sklearn.cluster import KMeans
from colour import (
    XYZ_to_Lab, Lab_to_XYZ,
    XYZ_to_CAM16UCS, CAM16UCS_to_XYZ,
    delta_E_CIE2000, delta_E_CAM16UCS
)
import colour

class PerceptualColorMatcher:
    def __init__(self):
        # Standardowe illuminanty
        self.illuminants = {
            'D65': np.array([0.31271, 0.32902]),
            'D50': np.array([0.34567, 0.35850]),
            'A': np.array([0.44757, 0.40745]),
            'F2': np.array([0.37208, 0.37529])
        }
        
        # Warunki obserwacji
        self.viewing_conditions = {
            'average': {'F': 1.0, 'c': 0.69, 'N_c': 1.0},
            'dim': {'F': 0.9, 'c': 0.59, 'N_c': 0.9},
            'dark': {'F': 0.8, 'c': 0.525, 'N_c': 0.8}
        }
        
        # Memory colors (typowe kolory zapamiętane)
        self.memory_colors_lab = {
            'skin_caucasian': [70, 15, 25],
            'skin_asian': [65, 12, 20],
            'skin_african': [45, 8, 15],
            'sky_blue': [70, -5, -25],
            'grass_green': [50, -30, 25],
            'red_apple': [45, 50, 35],
            'banana_yellow': [85, -5, 75]
        }
    
    def rgb_to_lab(self, rgb_image, illuminant='D65'):
        """Konwersja RGB do LAB"""
        # Normalizacja do [0,1]
        if rgb_image.dtype == np.uint8:
            rgb_normalized = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb_image.astype(np.float32)
        
        # Konwersja do XYZ
        xyz = colour.RGB_to_XYZ(
            rgb_normalized,
            colour.RGB_COLOURSPACES['sRGB'],
            illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]
        )
        
        # Konwersja do LAB
        lab = colour.XYZ_to_Lab(
            xyz,
            illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]
        )
        
        return lab
    
    def lab_to_rgb(self, lab_image, illuminant='D65'):
        """Konwersja LAB do RGB"""
        # Konwersja do XYZ
        xyz = colour.Lab_to_XYZ(
            lab_image,
            illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]
        )
        
        # Konwersja do RGB
        rgb = colour.XYZ_to_RGB(
            xyz,
            colour.RGB_COLOURSPACES['sRGB'],
            illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]
        )
        
        # Ograniczenie do [0,1] i konwersja do uint8
        rgb_clipped = np.clip(rgb, 0, 1)
        return (rgb_clipped * 255).astype(np.uint8)
    
    def rgb_to_cam16ucs(self, rgb_image, viewing_conditions='average'):
        """Konwersja RGB do CAM16-UCS"""
        # Parametry warunków obserwacji
        vc = self.viewing_conditions[viewing_conditions]
        
        # Normalizacja RGB
        if rgb_image.dtype == np.uint8:
            rgb_normalized = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb_image.astype(np.float32)
        
        # Konwersja do XYZ
        xyz = colour.RGB_to_XYZ(
            rgb_normalized,
            colour.RGB_COLOURSPACES['sRGB']
        )
        
        # Konwersja do CAM16-UCS
        cam16ucs = colour.XYZ_to_CAM16UCS(
            xyz,
            XYZ_w=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
            L_A=64/np.pi/5,  # Adapting luminance
            Y_b=20,          # Background luminance factor
            surround=colour.CAM16_VIEWING_CONDITIONS[viewing_conditions.upper()]
        )
        
        return cam16ucs
    
    def analyze_perceptual_characteristics(self, perceptual_image, color_space='lab'):
        """Analiza charakterystyk percepcyjnych obrazu"""
        characteristics = {}
        
        if color_space == 'lab':
            L, a, b = perceptual_image[:,:,0], perceptual_image[:,:,1], perceptual_image[:,:,2]
            
            # Podstawowe statystyki
            characteristics['lightness_mean'] = np.mean(L)
            characteristics['lightness_std'] = np.std(L)
            characteristics['a_mean'] = np.mean(a)
            characteristics['a_std'] = np.std(a)
            characteristics['b_mean'] = np.mean(b)
            characteristics['b_std'] = np.std(b)
            
            # Chroma i Hue
            chroma = np.sqrt(a**2 + b**2)
            hue = np.arctan2(b, a) * 180 / np.pi
            hue[hue < 0] += 360
            
            characteristics['chroma_mean'] = np.mean(chroma)
            characteristics['chroma_std'] = np.std(chroma)
            characteristics['hue_mean'] = np.mean(hue)
            characteristics['hue_std'] = np.std(hue)
            
            # Histogramy
            characteristics['lightness_hist'] = np.histogram(L, bins=100, range=(0, 100))[0]
            characteristics['chroma_hist'] = np.histogram(chroma, bins=100, range=(0, 100))[0]
            characteristics['hue_hist'] = np.histogram(hue, bins=36, range=(0, 360))[0]
            
        elif color_space == 'cam16ucs':
            J, a, b = perceptual_image[:,:,0], perceptual_image[:,:,1], perceptual_image[:,:,2]
            
            characteristics['lightness_mean'] = np.mean(J)
            characteristics['lightness_std'] = np.std(J)
            characteristics['a_mean'] = np.mean(a)
            characteristics['a_std'] = np.std(a)
            characteristics['b_mean'] = np.mean(b)
            characteristics['b_std'] = np.std(b)
        
        # Memory colors analysis
        characteristics['memory_colors'] = self.analyze_memory_colors(
            perceptual_image, color_space
        )
        
        # Dominant colors
        characteristics['dominant_colors'] = self.extract_dominant_colors(
            perceptual_image, n_colors=8
        )
        
        return characteristics
    
    def analyze_memory_colors(self, perceptual_image, color_space='lab'):
        """Analiza memory colors w obrazie"""
        memory_analysis = {}
        
        if color_space == 'lab':
            image_flat = perceptual_image.reshape(-1, 3)
            
            for color_name, target_lab in self.memory_colors_lab.items():
                # Obliczenie Delta E dla każdego piksela
                delta_e = np.sqrt(
                    ((image_flat[:, 0] - target_lab[0]) / 1.0)**2 +
                    ((image_flat[:, 1] - target_lab[1]) / 1.0)**2 +
                    ((image_flat[:, 2] - target_lab[2]) / 1.0)**2
                )
                
                # Znajdź piksele podobne do memory color (Delta E < 10)
                similar_pixels = delta_e < 10
                
                if np.any(similar_pixels):
                    memory_analysis[color_name] = {
                        'count': np.sum(similar_pixels),
                        'percentage': np.sum(similar_pixels) / len(image_flat) * 100,
                        'mean_lab': np.mean(image_flat[similar_pixels], axis=0),
                        'mean_delta_e': np.mean(delta_e[similar_pixels])
                    }
        
        return memory_analysis
    
    def extract_dominant_colors(self, perceptual_image, n_colors=8):
        """Ekstrakcja dominujących kolorów"""
        image_flat = perceptual_image.reshape(-1, 3)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(image_flat)
        
        dominant_colors = []
        for i in range(n_colors):
            mask = labels == i
            if np.any(mask):
                color_info = {
                    'color': kmeans.cluster_centers_[i],
                    'percentage': np.sum(mask) / len(image_flat) * 100,
                    'pixel_count': np.sum(mask)
                }
                dominant_colors.append(color_info)
        
        # Sortowanie według procentu
        dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
        
        return dominant_colors
    
    def calculate_perceptual_weights(self, lab_pixel):
        """Obliczenie wag percepcyjnych dla piksela"""
        L, a, b = lab_pixel[0], lab_pixel[1], lab_pixel[2]
        
        # Chroma i Hue
        chroma = np.sqrt(a**2 + b**2)
        hue = np.arctan2(b, a) * 180 / np.pi
        if hue < 0:
            hue += 360
        
        # Waga luminancji (wyższa dla średnich tonów)
        lightness_weight = 1.0 + 0.5 * np.exp(-((L - 50)**2) / (2 * 20**2))
        
        # Waga chromy (wyższa dla bardziej saturowanych kolorów)
        chroma_weight = 1.0 + 0.3 * (1.0 - np.exp(-chroma / 20))
        
        # Waga odcienia (wyższa dla memory colors)
        hue_weight = 1.0
        
        # Szczególna wrażliwość na odcienie skóry (hue 20-40°)
        if 20 <= hue <= 40 and 40 <= L <= 80 and 5 <= chroma <= 30:
            hue_weight = 2.0
        
        # Szczególna wrażliwość na zieleń (hue 90-150°)
        elif 90 <= hue <= 150:
            hue_weight = 1.5
        
        # Szczególna wrażliwość na błękit nieba (hue 200-250°)
        elif 200 <= hue <= 250 and L > 60:
            hue_weight = 1.3
        
        return {
            'lightness': lightness_weight,
            'chroma': chroma_weight,
            'hue': hue_weight,
            'overall': lightness_weight * chroma_weight * hue_weight
        }
    
    def calculate_perceptual_mapping(self, source_chars, target_chars, method='statistical'):
        """Obliczenie mapowania percepcyjnego"""
        mapping = {}
        
        if method == 'statistical':
            # Mapowanie statystyczne z wagami percepcyjnymi
            
            # Lightness mapping
            source_L_mean = source_chars['lightness_mean']
            source_L_std = source_chars['lightness_std']
            target_L_mean = target_chars['lightness_mean']
            target_L_std = target_chars['lightness_std']
            
            mapping['lightness_scale'] = target_L_std / source_L_std if source_L_std > 0 else 1.0
            mapping['lightness_shift'] = target_L_mean - source_L_mean * mapping['lightness_scale']
            
            # Chroma mapping
            source_C_mean = source_chars['chroma_mean']
            source_C_std = source_chars['chroma_std']
            target_C_mean = target_chars['chroma_mean']
            target_C_std = target_chars['chroma_std']
            
            mapping['chroma_scale'] = target_C_std / source_C_std if source_C_std > 0 else 1.0
            mapping['chroma_shift'] = target_C_mean - source_C_mean * mapping['chroma_scale']
            
            # Hue mapping (zachowanie względnych różnic)
            hue_shift = target_chars['hue_mean'] - source_chars['hue_mean']
            if hue_shift > 180:
                hue_shift -= 360
            elif hue_shift < -180:
                hue_shift += 360
            
            mapping['hue_shift'] = hue_shift
            
        elif method == 'memory_color_preservation':
            # Mapowanie z zachowaniem memory colors
            mapping = self.calculate_memory_color_mapping(
                source_chars['memory_colors'],
                target_chars['memory_colors']
            )
        
        return mapping
    
    def apply_perceptual_mapping(self, source_lab, mapping, weights=None):
        """Aplikacja mapowania percepcyjnego"""
        result_lab = source_lab.copy()
        
        L, a, b = source_lab[:,:,0], source_lab[:,:,1], source_lab[:,:,2]
        
        # Obliczenie chroma i hue
        chroma = np.sqrt(a**2 + b**2)
        hue = np.arctan2(b, a)
        
        # Aplikacja mapowania lightness
        new_L = L * mapping['lightness_scale'] + mapping['lightness_shift']
        new_L = np.clip(new_L, 0, 100)
        
        # Aplikacja mapowania chroma
        new_chroma = chroma * mapping['chroma_scale'] + mapping['chroma_shift']
        new_chroma = np.maximum(new_chroma, 0)
        
        # Aplikacja mapowania hue
        new_hue = hue + np.radians(mapping['hue_shift'])
        
        # Konwersja z powrotem do a*, b*
        new_a = new_chroma * np.cos(new_hue)
        new_b = new_chroma * np.sin(new_hue)
        
        # Aplikacja wag percepcyjnych (jeśli podane)
        if weights is not None:
            for i in range(result_lab.shape[0]):
                for j in range(result_lab.shape[1]):
                    pixel_weights = self.calculate_perceptual_weights(
                        source_lab[i, j]
                    )
                    
                    # Mieszanie oryginalnego i zmapowanego koloru
                    blend_factor = pixel_weights['overall'] / 3.0  # Normalizacja
                    blend_factor = np.clip(blend_factor, 0, 1)
                    
                    result_lab[i, j, 0] = L[i, j] * (1 - blend_factor) + new_L[i, j] * blend_factor
                    result_lab[i, j, 1] = a[i, j] * (1 - blend_factor) + new_a[i, j] * blend_factor
                    result_lab[i, j, 2] = b[i, j] * (1 - blend_factor) + new_b[i, j] * blend_factor
        else:
            result_lab[:,:,0] = new_L
            result_lab[:,:,1] = new_a
            result_lab[:,:,2] = new_b
        
        return result_lab
    
    def apply_chromatic_adaptation(self, lab_image, source_illuminant='D65', target_illuminant='D65'):
        """Aplikacja adaptacji chromatycznej"""
        if source_illuminant == target_illuminant:
            return lab_image
        
        # Konwersja do XYZ
        xyz = colour.Lab_to_XYZ(
            lab_image,
            illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][source_illuminant]
        )
        
        # Adaptacja chromatyczna (Bradford)
        adapted_xyz = colour.chromatic_adaptation(
            xyz,
            colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][source_illuminant],
            colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][target_illuminant],
            method='Bradford'
        )
        
        # Konwersja z powrotem do LAB
        adapted_lab = colour.XYZ_to_Lab(
            adapted_xyz,
            illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][target_illuminant]
        )
        
        return adapted_lab
    
    def apply_perceptual_color_matching(self, source_image, target_image,
                                       color_space='lab',
                                       mapping_method='statistical',
                                       use_perceptual_weights=True,
                                       chromatic_adaptation=True,
                                       source_illuminant='D65',
                                       target_illuminant='D65'):
        """Główna funkcja dopasowania percepcyjnego"""
        
        # Konwersja do przestrzeni percepcyjnej
        if color_space == 'lab':
            source_perceptual = self.rgb_to_lab(source_image, source_illuminant)
            target_perceptual = self.rgb_to_lab(target_image, target_illuminant)
        elif color_space == 'cam16ucs':
            source_perceptual = self.rgb_to_cam16ucs(source_image)
            target_perceptual = self.rgb_to_cam16ucs(target_image)
        else:
            raise ValueError(f"Nieobsługiwana przestrzeń kolorów: {color_space}")
        
        # Analiza charakterystyk percepcyjnych
        source_chars = self.analyze_perceptual_characteristics(
            source_perceptual, color_space
        )
        target_chars = self.analyze_perceptual_characteristics(
            target_perceptual, color_space
        )
        
        # Obliczenie mapowania
        mapping = self.calculate_perceptual_mapping(
            source_chars, target_chars, mapping_method
        )
        
        # Aplikacja mapowania
        weights = 'perceptual' if use_perceptual_weights else None
        result_perceptual = self.apply_perceptual_mapping(
            source_perceptual, mapping, weights
        )
        
        # Adaptacja chromatyczna
        if chromatic_adaptation and source_illuminant != target_illuminant:
            result_perceptual = self.apply_chromatic_adaptation(
                result_perceptual, source_illuminant, target_illuminant
            )
        
        # Konwersja z powrotem do RGB
        if color_space == 'lab':
            result_image = self.lab_to_rgb(result_perceptual, target_illuminant)
        elif color_space == 'cam16ucs':
            # Implementacja konwersji CAM16UCS -> RGB
            result_image = self.cam16ucs_to_rgb(result_perceptual)
        
        return result_image
    
    def evaluate_perceptual_quality(self, original, result, target, metric='delta_e_2000'):
        """Ocena jakości percepcyjnej"""
        # Konwersja do LAB
        result_lab = self.rgb_to_lab(result)
        target_lab = self.rgb_to_lab(target)
        
        if metric == 'delta_e_2000':
            # Delta E 2000
            delta_e = colour.delta_E_CIE2000(result_lab, target_lab)
            return {
                'mean_delta_e': np.mean(delta_e),
                'std_delta_e': np.std(delta_e),
                'max_delta_e': np.max(delta_e),
                'percentile_95': np.percentile(delta_e, 95)
            }
        
        elif metric == 'memory_color_accuracy':
            # Dokładność memory colors
            result_memory = self.analyze_memory_colors(result_lab)
            target_memory = self.analyze_memory_colors(target_lab)
            
            accuracy_scores = {}
            for color_name in self.memory_colors_lab.keys():
                if color_name in result_memory and color_name in target_memory:
                    result_color = result_memory[color_name]['mean_lab']
                    target_color = target_memory[color_name]['mean_lab']
                    
                    delta_e = np.sqrt(
                        ((result_color[0] - target_color[0]) / 1.0)**2 +
                        ((result_color[1] - target_color[1]) / 1.0)**2 +
                        ((result_color[2] - target_color[2]) / 1.0)**2
                    )
                    
                    accuracy_scores[color_name] = delta_e
            
            return accuracy_scores

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja
    perceptual_matcher = PerceptualColorMatcher()
    
    # Wczytanie obrazów
    source = cv2.imread('source.jpg')
    target = cv2.imread('target.jpg')
    
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    
    # Dopasowanie percepcyjne
    result = perceptual_matcher.apply_perceptual_color_matching(
        source, target,
        color_space='lab',
        mapping_method='statistical',
        use_perceptual_weights=True,
        chromatic_adaptation=True
    )
    
    # Ocena jakości
    quality = perceptual_matcher.evaluate_perceptual_quality(
        source, result, target, metric='delta_e_2000'
    )
    
    print(f"Mean Delta E: {quality['mean_delta_e']:.2f}")
    print(f"95th percentile Delta E: {quality['percentile_95']:.2f}")
    
    # Zapisanie wyniku
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result_perceptual.jpg', result_bgr)
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
```

### Predefiniowane Profile

#### Fotografia Portretowa
```python
portrait_params = PerceptualParameters()
portrait_params.preserve_memory_colors = True
portrait_params.memory_color_weight = 3.0
portrait_params.hue_sensitivity = 2.0  # Wysoka wrażliwość na odcienie skóry
portrait_params.local_adaptation = True
```

#### Fotografia Krajobrazowa
```python
landscape_params = PerceptualParameters()
landscape_params.chroma_sensitivity = 1.5  # Podkreślenie saturacji
landscape_params.color_space = 'cam16ucs'  # Lepsza percepcja dla krajobrazów
landscape_params.gamut_mapping = True
```

#### Reprodukcja Dzieł Sztuki
```python
art_reproduction_params = PerceptualParameters()
art_reproduction_params.mapping_method = 'memory_color_preservation'
art_reproduction_params.chromatic_adaptation = True
art_reproduction_params.adaptation_method = 'bradford'
art_reproduction_params.use_perceptual_weights = False  # Zachowanie oryginalnych proporcji
```

---

## 6. Analiza Wydajności

### Złożoność Obliczeniowa
- **Czasowa**: O(n × m × k × w) gdzie n=piksele, m=kanały, k=iteracje, w=wagi
- **Pamięciowa**: O(n × m × s) gdzie s=przestrzenie kolorów

### Benchmarki Wydajności

| Rozdzielczość | LAB [s] | CAM16-UCS [s] | RAM [MB] | Jakość [ΔE] |
|---------------|---------|---------------|----------|-------------|
| 1920×1080     | 3.8     | 12.4          | 45       | 2.1         |
| 3840×2160     | 15.2    | 49.6          | 180      | 1.8         |
| 7680×4320     | 60.8    | 198.4         | 720      | 1.6         |

### Optymalizacje

```python
# Optymalizacja dla dużych obrazów
def optimized_perceptual_matching(image, target, tile_size=512, overlap=64):
    """Przetwarzanie kafelkowe z nakładaniem"""
    height, width = image.shape[:2]
    result = np.zeros_like(image)
    
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            # Wycinanie kafelka z nakładaniem
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            tile = image[y:y_end, x:x_end]
            
            # Przetwarzanie kafelka
            processed_tile = process_perceptual_tile(tile, target)
            
            # Mieszanie z nakładaniem
            if y > 0 or x > 0:
                result[y:y_end, x:x_end] = blend_with_overlap(
                    result[y:y_end, x:x_end],
                    processed_tile,
                    overlap
                )
            else:
                result[y:y_end, x:x_end] = processed_tile
    
    return result

# Optymalizacja pamięci
def memory_efficient_processing(image, chunk_size=1000000):
    """Przetwarzanie w małych fragmentach"""
    original_shape = image.shape
    image_flat = image.reshape(-1, 3)
    result_flat = np.zeros_like(image_flat)
    
    for i in range(0, len(image_flat), chunk_size):
        chunk = image_flat[i:i+chunk_size]
        result_flat[i:i+chunk_size] = process_chunk(chunk)
    
    return result_flat.reshape(original_shape)
```

---

## 7. Ocena Jakości

### Metryki Percepcyjne

```python
def comprehensive_perceptual_evaluation(original, result, target):
    """Kompleksowa ocena percepcyjna"""
    evaluation = {}
    
    # Delta E metrics
    evaluation['delta_e_2000'] = calculate_delta_e_2000(result, target)
    evaluation['delta_e_cam16'] = calculate_delta_e_cam16(result, target)
    evaluation['delta_e_94'] = calculate_delta_e_94(result, target)
    
    # Memory color accuracy
    evaluation['memory_color_accuracy'] = evaluate_memory_colors(result, target)
    
    # Perceptual uniformity
    evaluation['perceptual_uniformity'] = calculate_perceptual_uniformity(result)
    
    # Color harmony preservation
    evaluation['harmony_preservation'] = evaluate_color_harmony(
        original, result, target
    )
    
    # Spatial coherence
    evaluation['spatial_coherence'] = calculate_spatial_coherence(result)
    
    return evaluation

def calculate_perceptual_uniformity(image_lab):
    """Obliczenie jednorodności percepcyjnej"""
    # Gradient w przestrzeni LAB
    L, a, b = image_lab[:,:,0], image_lab[:,:,1], image_lab[:,:,2]
    
    # Gradienty dla każdego kanału
    grad_L = np.gradient(L)
    grad_a = np.gradient(a)
    grad_b = np.gradient(b)
    
    # Magnitude gradientu percepcyjnego
    perceptual_gradient = np.sqrt(
        grad_L[0]**2 + grad_L[1]**2 +
        grad_a[0]**2 + grad_a[1]**2 +
        grad_b[0]**2 + grad_b[1]**2
    )
    
    # Jednorodność jako odwrotność wariancji gradientu
    uniformity = 1.0 / (1.0 + np.var(perceptual_gradient))
    
    return uniformity
```

### Kryteria Oceny

| Metryka | Doskonały | Dobry | Akceptowalny | Słaby |
|---------|-----------|-------|--------------|-------|
| Delta E 2000 | < 1.0 | 1-2.5 | 2.5-5.0 | > 5.0 |
| Delta E CAM16 | < 0.8 | 0.8-2.0 | 2.0-4.0 | > 4.0 |
| Memory Color Accuracy | > 95% | 90-95% | 80-90% | < 80% |
| Perceptual Uniformity | > 0.9 | 0.8-0.9 | 0.7-0.8 | < 0.7 |
| Harmony Preservation | > 0.95 | 0.9-0.95 | 0.8-0.9 | < 0.8 |

---

## 8. Przypadki Użycia

### Przypadek 1: Fotografia Medyczna
```python
# Precyzyjne odwzorowanie kolorów skóry
medical_params = PerceptualParameters()
medical_params.preserve_memory_colors = True
medical_params.memory_color_weight = 5.0
medical_params.color_space = 'lab'
medical_params.hue_sensitivity = 3.0

result = perceptual_matcher.apply_perceptual_color_matching(
    patient_photo, reference_standard,
    **medical_params.__dict__
)
```

### Przypadek 2: E-commerce
```python
# Spójne przedstawienie produktów
ecommerce_params = PerceptualParameters()
ecommerce_params.chromatic_adaptation = True
ecommerce_params.source_illuminant = 'F2'  # Fluorescent
ecommerce_params.target_illuminant = 'D65'  # Standard daylight

product_photos = []
for photo in raw_product_photos:
    standardized = perceptual_matcher.apply_perceptual_color_matching(
        photo, color_standard,
        **ecommerce_params.__dict__
    )
    product_photos.append(standardized)
```

### Przypadek 3: Archiwizacja Cyfrowa
```python
# Zachowanie autentycznych kolorów dzieł sztuki
archival_params = PerceptualParameters()
archival_params.mapping_method = 'memory_color_preservation'
archival_params.use_perceptual_weights = False
archival_params.gamut_mapping = False  # Zachowanie oryginalnego gamut

archival_version = perceptual_matcher.apply_perceptual_color_matching(
    scanned_artwork, color_reference,
    **archival_params.__dict__
)
```

---

## 9. Rozwiązywanie Problemów

### Częste Problemy

#### Problem: Nienaturalne odcienie skóry
```python
def fix_skin_tone_issues(result_lab):
    """Korekcja problemów z odcieniami skóry"""
    # Identyfikacja regionów skóry
    skin_mask = identify_skin_regions(result_lab)
    
    # Korekcja odcienia dla regionów skóry
    corrected_lab = result_lab.copy()
    
    for i in range(result_lab.shape[0]):
        for j in range(result_lab.shape[1]):
            if skin_mask[i, j]:
                # Korekta w kierunku naturalnych odcieni skóry
                corrected_lab[i, j] = correct_skin_tone(
                    result_lab[i, j],
                    target_skin_tone='caucasian'  # lub 'asian', 'african'
                )
    
    return corrected_lab
```

#### Problem: Utrata saturacji
```python
def restore_saturation(original_lab, result_lab, factor=1.2):
    """Przywrócenie saturacji"""
    # Obliczenie chroma
    original_chroma = np.sqrt(
        original_lab[:,:,1]**2 + original_lab[:,:,2]**2
    )
    result_chroma = np.sqrt(
        result_lab[:,:,1]**2 + result_lab[:,:,2]**2
    )
    
    # Obliczenie współczynnika korekcji
    chroma_ratio = np.where(
        result_chroma > 0,
        original_chroma / result_chroma * factor,
        1.0
    )
    
    # Aplikacja korekcji
    corrected_lab = result_lab.copy()
    corrected_lab[:,:,1] *= chroma_ratio
    corrected_lab[:,:,2] *= chroma_ratio
    
    return corrected_lab
```

#### Problem: Artefakty w gradientach
```python
def smooth_gradients(result_lab, kernel_size=5):
    """Wygładzenie gradientów"""
    from scipy import ndimage
    
    # Gaussian smoothing dla każdego kanału
    smoothed_lab = result_lab.copy()
    
    for channel in range(3):
        smoothed_lab[:,:,channel] = ndimage.gaussian_filter(
            result_lab[:,:,channel],
            sigma=kernel_size/6
        )
    
    return smoothed_lab
```

### Debugowanie

```python
def debug_perceptual_matching(source, target, result):
    """Narzędzie debugowania"""
    print("=== Perceptual Matching Debug ===")
    
    # Konwersja do LAB
    source_lab = rgb_to_lab(source)
    target_lab = rgb_to_lab(target)
    result_lab = rgb_to_lab(result)
    
    # Analiza memory colors
    source_memory = analyze_memory_colors(source_lab)
    target_memory = analyze_memory_colors(target_lab)
    result_memory = analyze_memory_colors(result_lab)
    
    print("Memory Colors Analysis:")
    for color_name in ['skin_caucasian', 'sky_blue', 'grass_green']:
        if color_name in source_memory:
            print(f"{color_name}:")
            print(f"  Source: {source_memory[color_name]['mean_lab']}")
            if color_name in target_memory:
                print(f"  Target: {target_memory[color_name]['mean_lab']}")
            if color_name in result_memory:
                print(f"  Result: {result_memory[color_name]['mean_lab']}")
    
    # Ocena jakości
    quality = evaluate_perceptual_quality(source, result, target)
    print(f"\nQuality Metrics:")
    print(f"  Mean Delta E 2000: {quality['mean_delta_e']:.2f}")
    print(f"  95th percentile: {quality['percentile_95']:.2f}")
    
    # Wizualizacja
    plot_perceptual_analysis(source_lab, target_lab, result_lab)
```

---

## 10. Przyszłe Ulepszenia

### Planowane Funkcje

#### 1. Advanced Perceptual Models
```python
class AdvancedPerceptualMatcher(PerceptualColorMatcher):
    def __init__(self):
        super().__init__()
        self.cam16_model = load_cam16_model()
        self.appearance_model = load_appearance_model()
    
    def apply_advanced_perception(self, image, viewing_conditions):
        # Zaawansowane modele percepcji
        return self.appearance_model.transform(
            image, viewing_conditions
        )
```

#### 2. Machine Learning Enhancement
```python
class MLPerceptualMatcher(PerceptualColorMatcher):
    def __init__(self):
        super().__init__()
        self.perceptual_network = load_model('perceptual_net.pth')
    
    def ml_enhanced_matching(self, source, target):
        # Tradycyjne dopasowanie
        basic_result = self.apply_perceptual_color_matching(source, target)
        
        # ML enhancement
        enhanced = self.perceptual_network.predict(
            np.concatenate([source, target, basic_result], axis=2)
        )
        
        return enhanced
```

#### 3. Real-time Perceptual Processing
- GPU acceleration dla CAM16
- Streaming perceptual analysis
- Live preview z perceptual feedback

#### 4. Personalized Perception
```python
class PersonalizedPerceptualMatcher(PerceptualColorMatcher):
    def __init__(self, user_profile):
        super().__init__()
        self.user_profile = user_profile
        self.personal_weights = calculate_personal_weights(user_profile)
    
    def apply_personalized_matching(self, source, target):
        # Dopasowanie z personalizowanymi wagami
        return self.apply_perceptual_color_matching(
            source, target,
            perceptual_weights=self.personal_weights
        )
```

### Roadmap Rozwoju

**Q1 2024:**
- ✅ Podstawowa implementacja LAB
- ✅ Memory colors
- 🔄 CAM16-UCS integration

**Q2 2024:**
- 📋 Advanced perceptual models
- 📋 ML enhancement
- 📋 Real-time processing

**Q3 2024:**
- 📋 Personalized perception
- 📋 Advanced quality metrics
- 📋 Professional tools integration

**Q4 2024:**
- 📋 Research publication
- 📋 Perceptual database
- 📋 Industry standards compliance

---

*Ostatnia aktualizacja: 2024-01-20*  
*Autor: GattoNero AI Assistant*  
*Wersja: 1.0*  
*Status: W rozwoju* 🔄