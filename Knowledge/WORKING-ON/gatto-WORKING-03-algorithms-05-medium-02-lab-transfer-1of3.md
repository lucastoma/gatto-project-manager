# LAB Color Space Transfer - Część 1: Podstawy Teoretyczne (Wersja Poprawiona)

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 4-6 godzin | **Złożoność**: O(n)  
**ID Algorytmu**: `algorithm_05_lab_transfer` | **Numer API**: `5`

---

## Przegląd

LAB Color Space Transfer to zaawansowany algorytm dopasowania kolorów operujący w przestrzeni kolorów LAB (CIELAB). Algorytm wykorzystuje percepcyjnie jednolitą przestrzeń kolorów, gdzie odległości euklidesowe lepiej odpowiadają różnicom percepcyjnym między kolorami.

### Zastosowania
- Profesjonalna korekta kolorów
- Dopasowanie oświetlenia między zdjęciami
- Color grading w postprodukcji
- Normalizacja kolorów w seriach zdjęć

### Zalety
- ✅ Percepcyjnie dokładne dopasowanie
- ✅ Zachowuje naturalne przejścia kolorów
- ✅ Lepsze wyniki niż RGB
- ✅ Kontrola nad luminancją i chromatycznością

### Wady
- ❌ Wyższa złożoność obliczeniowa
- ❌ Wymaga konwersji przestrzeni kolorów
- ❌ Może być zbyt subtelny dla niektórych zastosowań
- ❌ Trudniejszy w implementacji

---

## Podstawy Teoretyczne

### Przestrzeń Kolorów LAB (CIELAB)

Przestrzeń LAB składa się z trzech składowych:
- **L*** (Lightness): Jasność (0-100)
- **a***: Oś zielony-czerwony (-128 do +127)
- **b***: Oś niebieski-żółty (-128 do +127)

### Właściwości Przestrzeni LAB

1. **Percepcyjna jednolitość**: Równe odległości w przestrzeni LAB odpowiadają podobnym różnicom percepcyjnym
2. **Niezależność od urządzenia**: Nie zależy od konkretnego monitora czy drukarki
3. **Separacja luminancji**: Kanał L* jest niezależny od chromatyczności
4. **Większy gamut**: Pokrywa wszystkie kolory widzialne przez człowieka

---

## Konwersja RGB ↔ LAB

Proces konwersji przebiega przez przestrzeń XYZ: RGB → XYZ → LAB i z powrotem LAB → XYZ → RGB. Poniżej znajdują się formuły i zoptymalizowana implementacja.

#### Krok 1: RGB → XYZ
```
// Normalizacja RGB (0-1)
R' = R / 255.0
G' = G / 255.0
B' = B / 255.0

// Gamma correction (sRGB)
if R' > 0.04045:
    R' = ((R' + 0.055) / 1.055)^2.4
else:
    R' = R' / 12.92

// Podobnie dla G' i B' 

// Transformacja do XYZ (sRGB matrix)
X = R' * 0.4124564 + G' * 0.3575761 + B' * 0.1804375
Y = R' * 0.2126729 + G' * 0.7151522 + B' * 0.0721750
Z = R' * 0.0193339 + G' * 0.1191920 + B' * 0.9503041
```

#### Krok 2: XYZ → LAB
```
// Normalizacja względem białego D65
Xn = 95.047
Yn = 100.000
Zn = 108.883

fx = f(X / Xn)
fy = f(Y / Yn)
fz = f(Z / Zn)

// Funkcja f(t)
if t > (6/29)^3:
    f(t) = t^(1/3)
else:
    f(t) = (1/3) * (29/6)^2 * t + 4/29

// Obliczenie LAB
L* = 116 * fy - 16
a* = 500 * (fx - fy)
b* = 200 * (fy - fz)
```

#### Krok 1: LAB → XYZ
```
fy = (L* + 16) / 116
fx = a* / 500 + fy
fz = fy - b* / 200

// Odwrotna funkcja f
if fx^3 > (6/29)^3:
    X = fx^3 * Xn
else:
    X = 3 * (6/29)^2 * (fx - 4/29) * Xn

// Podobnie dla Y i Z
```

#### Krok 2: XYZ → RGB
```
// Odwrotna transformacja (sRGB matrix)
R' = X *  3.2404542 + Y * -1.5371385 + Z * -0.4985314
G' = X * -0.9692660 + Y *  1.8760108 + Z *  0.0415560
B' = X *  0.0556434 + Y * -0.2040259 + Z *  1.0572252

// Odwrotna gamma correction
if R' > 0.0031308:
    R' = 1.055 * R'^(1/2.4) - 0.055
else:
    R' = 12.92 * R'

// Denormalizacja do 0-255
R = R' * 255
G = G' * 255
B = B' * 255
```

---

## Algorytm Transferu Kolorów w LAB

### Podstawowa Metoda: Statystyczny Transfer

Algorytm dopasowuje statystyki (średnią i odchylenie standardowe) każdego kanału LAB:

```
FUNCTION lab_color_transfer(source_image, target_image):
    // Konwertuj do LAB
    source_lab = rgb_to_lab(source_image)
    target_lab = rgb_to_lab(target_image)
    
    result_lab = copy(source_lab)
    
    FOR each channel in [L, a, b]:
        // Oblicz statystyki
        source_mean = mean(source_lab[channel])
        source_std = std(source_lab[channel])
        target_mean = mean(target_lab[channel])
        target_std = std(target_lab[channel])
        
        // Zastosuj transformację
        result_lab[channel] = (source_lab[channel] - source_mean) * (target_std / source_std) + target_mean
    
    // Konwertuj z powrotem do RGB
    result_rgb = lab_to_rgb(result_lab)
    
    RETURN result_rgb
```

### Zaawansowane Metody

#### 1. Selektywny Transfer Kanałów
```
// Transfer tylko chromatyczności (a*, b*)
result_lab[L] = source_lab[L]  // Zachowaj oryginalną jasność
result_lab[a] = transfer_channel(source_lab[a], target_lab[a])
result_lab[b] = transfer_channel(source_lab[b], target_lab[b])
```

#### 2. Adaptacyjny Transfer z Wagami
```
// Różne wagi dla różnych kanałów
weight_L = 0.8  // Mniejsza zmiana jasności
weight_a = 1.0  // Pełny transfer chromatyczności
weight_b = 1.0

result_lab[L] = source_lab[L] + weight_L * (transferred_L - source_lab[L])
```

#### 3. Lokalny Transfer LAB
```
// Transfer w regionach o podobnej jasności
FOR each luminance_range in [0-33, 34-66, 67-100]:
    mask = create_luminance_mask(source_lab[L], luminance_range)
    apply_transfer_to_region(result_lab, mask, target_stats)
```

---

## Metryki Jakości w Przestrzeni LAB

### Delta E - Odległość Percepcyjna

Delta E mierzy percepcyjną różnicę między kolorami:

```
// Delta E 1976 (CIE76)
ΔE*ab = √[(ΔL*)² + (Δa*)² + (Δb*)²]

// Delta E 1994 (CIE94)
ΔE*94 = √[(ΔL*/kL*SL)² + (ΔC*/kC*SC)² + (ΔH*/kH*SH)²]

// Delta E 2000 (CIEDE2000) - najbardziej dokładny
ΔE*00 = √[(ΔL'/kL*SL)² + (ΔC'/kC*SC)² + (ΔH'/kH*SH)² + RT*(ΔC'/kC*SC)*(ΔH'/kH*SH)]
```

### Interpretacja Delta E
- **0-1**: Różnica niezauważalna
- **1-2**: Różnica ledwo zauważalna
- **2-3**: Różnica zauważalna przy porównaniu
- **3-6**: Różnica wyraźnie zauważalna
- **6+**: Różnica bardzo duża

### Ocena Jakości Transferu
```python
def evaluate_lab_transfer_quality(source_lab, target_lab, result_lab):
    """
    Ocenia jakość transferu kolorów w przestrzeni LAB
    """
    metrics = {}
    
    # 1. Średnie Delta E między wynikiem a targetem
    delta_e_target = calculate_delta_e(result_lab, target_lab)
    metrics['mean_delta_e_target'] = np.mean(delta_e_target)
    
    # 2. Zachowanie struktury (korelacja z oryginałem)
    for channel in ['L', 'a', 'b']:
        correlation = np.corrcoef(
            source_lab[channel].flatten(), 
            result_lab[channel].flatten()
        )[0, 1]
        metrics[f'correlation_{channel}'] = correlation
    
    # 3. Dopasowanie statystyk do targetu
    for channel in ['L', 'a', 'b']:
        target_mean = np.mean(target_lab[channel])
        result_mean = np.mean(result_lab[channel])
        target_std = np.std(target_lab[channel])
        result_std = np.std(result_lab[channel])
        
        metrics[f'mean_diff_{channel}'] = abs(target_mean - result_mean)
        metrics[f'std_diff_{channel}'] = abs(target_std - result_std)
    
    return metrics
```

---

## Implementacja Podstawowych Funkcji (Wersja Zoptymalizowana)

Poniższy kod przedstawia zoptymalizowane, zwektoryzowane funkcje konwersji zgodne z architekturą GattoNero. Użycie operacji na tablicach NumPy (np.where) jest znacznie wydajniejsze niż podejście z np.vectorize.

```python
import numpy as np
from PIL import Image
from app.core.development_logger import get_logger
from app.core.performance_profiler import get_profiler

class LABColorConverter:
    """
    Klasa do konwersji kolorów RGB ↔ LAB zgodna z architekturą GattoNero
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.profiler = get_profiler()
        
        # Stałe używane w konwersjach
        self.ILLUMINANT_D65 = np.array([95.047, 100.000, 108.883])
        self.SRGB_TO_XYZ_MATRIX = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        self.XYZ_TO_SRGB_MATRIX = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])
        
        # Cache dla optymalizacji (z limitem)
        self._conversion_cache = {}
        self.MAX_CACHE_SIZE = 10

    def load_image_safely(self, image_path):
        """Bezpieczne ładowanie obrazów z różnych formatów"""
        try:
            image = Image.open(image_path)
            
            # Konwertuj RGBA do RGB z białym tłem
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
            return np.array(image)
        except Exception as e:
            self.logger.error(f"Błąd ładowania obrazu {image_path}: {e}")
            raise

    def validate_lab_ranges(self, lab_array):
        """Walidacja zakresów LAB z automatyczną korekcją"""
        L, a, b = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]
        
        corrections = []
        
        # Sprawdź i popraw zakresy
        if np.any(L < 0) or np.any(L > 100):
            lab_array[:, :, 0] = np.clip(L, 0, 100)
            corrections.append("L channel clipped to [0, 100]")
        
        if np.any(a < -128) or np.any(a > 127):
            lab_array[:, :, 1] = np.clip(a, -128, 127)
            corrections.append("a channel clipped to [-128, 127]")
        
        if np.any(b < -128) or np.any(b > 127):
            lab_array[:, :, 2] = np.clip(b, -128, 127)
            corrections.append("b channel clipped to [-128, 127]")
        
        if corrections:
            self.logger.warning(f"LAB corrections applied: {corrections}")
        
        return lab_array

    def rgb_to_lab(self, rgb_array):
        """
        Zoptymalizowana konwersja RGB -> LAB z walidacją i cache
        """
        with self.profiler.profile_operation("rgb_to_lab_conversion"):
            # Cache key based on shape and sample of data
            cache_key = (rgb_array.shape, hash(rgb_array.tobytes()[:1000]))
            
            if cache_key in self._conversion_cache:
                return self._conversion_cache[cache_key].copy()
            
            try:
                # Normalizacja i korekcja gamma
                rgb_norm = rgb_array.astype(np.float64) / 255.0
                mask = rgb_norm > 0.04045
                rgb_linear = np.where(mask,
                                     np.power((rgb_norm + 0.055) / 1.055, 2.4),
                                     rgb_norm / 12.92)
                
                # Transformacja do XYZ
                original_shape = rgb_linear.shape
                xyz = np.dot(rgb_linear.reshape(-1, 3), self.SRGB_TO_XYZ_MATRIX.T).reshape(original_shape)
                
                # Transformacja do LAB
                xyz_norm = xyz / self.ILLUMINANT_D65
                delta = 6.0 / 29.0
                f_xyz = np.where(xyz_norm > (delta ** 3),
                                np.power(xyz_norm, 1.0/3.0),
                                (xyz_norm / (3 * delta**2)) + (4.0/29.0))
                
                L = 116 * f_xyz[:, :, 1] - 16
                a = 500 * (f_xyz[:, :, 0] - f_xyz[:, :, 1])
                b = 200 * (f_xyz[:, :, 1] - f_xyz[:, :, 2])
                
                lab_array = np.stack([L, a, b], axis=2)
                
                # Walidacja zakresów
                lab_array = self.validate_lab_ranges(lab_array)
                
                # Zarządzanie cache
                if len(self._conversion_cache) >= self.MAX_CACHE_SIZE:
                    oldest_key = next(iter(self._conversion_cache))
                    del self._conversion_cache[oldest_key]
                
                self._conversion_cache[cache_key] = lab_array
                return lab_array
                
            except Exception as e:
                self.logger.error(f"Błąd konwersji RGB->LAB: {e}")
                raise

    def lab_to_rgb(self, lab_array):
        """
        Zoptymalizowana konwersja LAB -> RGB z walidacją
        """
        with self.profiler.profile_operation("lab_to_rgb_conversion"):
            try:
                L, a, b = lab_array[:, :, 0], lab_array[:, :, 1], lab_array[:, :, 2]
                
                # Transformacja do XYZ
                fy = (L + 16) / 116
                fx = a / 500 + fy
                fz = fy - b / 200
                
                delta = 6.0 / 29.0
                
                def f_inv(t):
                    return np.where(t > delta,
                                   np.power(t, 3),
                                   3 * delta**2 * (t - 4.0/29.0))
                
                xyz = np.stack([
                    f_inv(fx) * self.ILLUMINANT_D65[0],
                    f_inv(fy) * self.ILLUMINANT_D65[1],
                    f_inv(fz) * self.ILLUMINANT_D65[2]
                ], axis=2)
                
                # Transformacja do RGB
                original_shape = xyz.shape
                rgb_linear = np.dot(xyz.reshape(-1, 3), self.XYZ_TO_SRGB_MATRIX.T).reshape(original_shape)
                
                # Odwrotna korekcja gamma
                mask = rgb_linear > 0.0031308
                rgb_norm = np.where(mask,
                                   1.055 * np.power(np.abs(rgb_linear), 1.0/2.4) - 0.055,
                                   12.92 * rgb_linear)
                
                # Denormalizacja i obcięcie do zakresu
                rgb = np.clip(rgb_norm * 255, 0, 255).astype(np.uint8)
                
                return rgb
                
            except Exception as e:
                self.logger.error(f"Błąd konwersji LAB->RGB: {e}")
                raise

    def calculate_delta_e(self, lab1, lab2):
        """
        Oblicza Delta E między dwoma obrazami LAB przy użyciu miary CIEDE2000.
        Jest to percepcyjnie dokładniejsza miara niż Delta E 1976 (Euclidean).
        
        Wymaga: from skimage.color import deltaE_ciede2000
        """
        # Import na poziomie funkcji aby uniknąć zależności globalnych
        from skimage.color import deltaE_ciede2000
        
        # CIEDE2000 dla lepszej percepcyjnej dokładności
        # Musimy zadbać o kształt arrayów
        original_shape = lab1.shape[:2]  # Zachowaj oryginalny kształt
        
        # Przekształć do formatu wymaganego przez deltaE_ciede2000
        lab1_reshaped = lab1.reshape(-1, 3)
        lab2_reshaped = lab2.reshape(-1, 3)
        
        # Oblicz Delta E używając CIEDE2000
        delta_e = deltaE_ciede2000(lab1_reshaped, lab2_reshaped)
        
        # Przywróć oryginalny kształt
        return delta_e.reshape(original_shape)
```

---

## Walidacja Konwersji

### Test Roundtrip

Test "w obie strony" (RGB → LAB → RGB) pozwala zweryfikować, czy konwersje nie wprowadzają znaczących błędów.

```python
def test_rgb_lab_roundtrip():
    """
    Testuje, czy konwersja RGB -> LAB -> RGB zachowuje kolory.
    Używa zoptymalizowanych funkcji.
    """
    # Utwórz instancję konwertera
    converter = LABColorConverter()
    
    test_colors = np.array([
        [[255, 0, 0]],    # Czerwony
        [[0, 255, 0]],    # Zielony
        [[0, 0, 255]],    # Niebieski
        [[255, 255, 255]], # Biały
        [[0, 0, 0]],      # Czarny
        [[128, 128, 128]] # Szary
    ], dtype=np.uint8)
    
    # Konwersja roundtrip
    lab = converter.rgb_to_lab(test_colors)
    rgb_back = converter.lab_to_rgb(lab)
    
    # Sprawdzenie różnic
    diff = np.abs(test_colors.astype(float) - rgb_back.astype(float))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Maksymalna różnica w teście roundtrip: {max_diff:.2f}")
    print(f"Średnia różnica w teście roundtrip: {mean_diff:.2f}")
    
    # Błędy zaokrągleń mogą powodować niewielkie różnice.
    # Tolerancja na poziomie 2 jednostek na kanał jest akceptowalna.
    assert max_diff <= 2, f"Zbyt duża różnica w roundtrip: {max_diff}"
    
    return True

# Uruchomienie testu
if __name__ == "__main__":
    test_rgb_lab_roundtrip()
    print("✅ Test roundtrip RGB↔LAB przeszedł pomyślnie.")
```

---

## Podsumowanie Części 1

W tej części omówiliśmy:
- Podstawy teoretyczne przestrzeni kolorów LAB.
- Matematyczne formuły konwersji RGB ↔ LAB.
- Zoptymalizowaną implementację podstawowych funkcji konwersji.
- Metryki jakości (Delta E) do oceny różnic kolorów.
- Metodę walidacji poprawności konwersji za pomocą testu roundtrip.
- Integrację z systemem GattoNero zgodnie z ustalonymi zasadami.