# Perceptual Color Matching Algorithm - Część 3 z 6: Implementacja Podstawowa

**Część 3 z 6: Implementacja Podstawowa**

---

## Nawigacja

**◀️ Poprzednia część**: [Pseudokod i Architektura Systemu](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-2of6.md)  
**▶️ Następna część**: [Parametry i Konfiguracja](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-4of6.md)  
**🏠 Powrót do**: [Spis Treści](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-0of6.md)

---

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

## 4. Implementacja Python - Część 1

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
        # Normalizacja
        if rgb_image.dtype == np.uint8:
            rgb_normalized = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb_image.astype(np.float32)
        
        # Konwersja do XYZ
        xyz = colour.RGB_to_XYZ(
            rgb_normalized,
            colour.RGB_COLOURSPACES['sRGB']
        )
        
        # Warunki obserwacji dla CAM16
        vc = self.viewing_conditions[viewing_conditions]
        cam16_vc = colour.CAM16_VIEWING_CONDITIONS['CIE 1931 2 Degree Standard Observer']
        cam16_vc.update(vc)
        
        # Konwersja do CAM16-UCS
        cam16ucs = colour.XYZ_to_CAM16UCS(
            xyz,
            cam16_vc
        )
        
        return cam16ucs
    
    def cam16ucs_to_rgb(self, cam16ucs_image, viewing_conditions='average'):
        """Konwersja CAM16-UCS do RGB"""
        # Warunki obserwacji
        vc = self.viewing_conditions[viewing_conditions]
        cam16_vc = colour.CAM16_VIEWING_CONDITIONS['CIE 1931 2 Degree Standard Observer']
        cam16_vc.update(vc)
        
        # Konwersja do XYZ
        xyz = colour.CAM16UCS_to_XYZ(
            cam16ucs_image,
            cam16_vc
        )
        
        # Konwersja do RGB
        rgb = colour.XYZ_to_RGB(
            xyz,
            colour.RGB_COLOURSPACES['sRGB']
        )
        
        # Ograniczenie i konwersja
        rgb_clipped = np.clip(rgb, 0, 1)
        return (rgb_clipped * 255).astype(np.uint8)
```

---

## Nawigacja

**◀️ Poprzednia część**: [Pseudokod i Architektura Systemu](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-2of6.md)  
**▶️ Następna część**: [Parametry i Konfiguracja](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-4of6.md)  
**🏠 Powrót do**: [Spis Treści](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-0of6.md)

---

*Ostatnia aktualizacja: 2024-01-20*  
*Autor: GattoNero AI Assistant*  
*Wersja: 1.0*  
*Status: Część 3 z 6 - Implementacja Podstawowa* ✅