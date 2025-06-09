# LAB Color Space Transfer - Część 1: Podstawy Teoretyczne

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 4-6 godzin | **Złożoność**: O(n)

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

### Konwersja RGB → LAB

Proces konwersji przebiega przez przestrzeń XYZ:

```
RGB → XYZ → LAB
```

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

### Konwersja LAB → RGB

Proces odwrotny:

```
LAB → XYZ → RGB
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
        result_lab[channel] = (source_lab[channel] - source_mean) * 
                             (target_std / source_std) + target_mean
    
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

## Implementacja Podstawowych Funkcji

### Konwersja RGB → LAB
```python
import numpy as np
from scipy import optimize

def rgb_to_xyz(rgb):
    """
    Konwertuje RGB do XYZ (sRGB)
    """
    # Normalizacja do 0-1
    rgb_norm = rgb / 255.0
    
    # Gamma correction
    def gamma_correct(c):
        if c > 0.04045:
            return np.power((c + 0.055) / 1.055, 2.4)
        else:
            return c / 12.92
    
    rgb_linear = np.vectorize(gamma_correct)(rgb_norm)
    
    # sRGB to XYZ matrix
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    # Reshape dla matrix multiplication
    original_shape = rgb_linear.shape
    rgb_reshaped = rgb_linear.reshape(-1, 3)
    
    # Transformacja
    xyz = np.dot(rgb_reshaped, matrix.T)
    
    return xyz.reshape(original_shape)

def xyz_to_lab(xyz):
    """
    Konwertuje XYZ do LAB
    """
    # Illuminant D65
    Xn, Yn, Zn = 95.047, 100.000, 108.883
    
    # Normalizacja
    x = xyz[:, :, 0] / Xn
    y = xyz[:, :, 1] / Yn
    z = xyz[:, :, 2] / Zn
    
    # Funkcja f(t)
    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta**3, 
                       np.power(t, 1.0/3.0),
                       (t / (3 * delta**2)) + (4.0/29.0))
    
    fx = f(x)
    fy = f(y)
    fz = f(z)
    
    # Obliczenie LAB
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.stack([L, a, b], axis=2)

def rgb_to_lab(rgb):
    """
    Bezpośrednia konwersja RGB → LAB
    """
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    return lab
```

### Konwersja LAB → RGB
```python
def lab_to_xyz(lab):
    """
    Konwertuje LAB do XYZ
    """
    # Illuminant D65
    Xn, Yn, Zn = 95.047, 100.000, 108.883
    
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    # Obliczenie f wartości
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    # Odwrotna funkcja f
    def f_inv(t):
        delta = 6.0 / 29.0
        return np.where(t > delta,
                       np.power(t, 3),
                       3 * delta**2 * (t - 4.0/29.0))
    
    x = f_inv(fx) * Xn
    y = f_inv(fy) * Yn
    z = f_inv(fz) * Zn
    
    return np.stack([x, y, z], axis=2)

def xyz_to_rgb(xyz):
    """
    Konwertuje XYZ do RGB (sRGB)
    """
    # XYZ to sRGB matrix
    matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    # Reshape dla matrix multiplication
    original_shape = xyz.shape
    xyz_reshaped = xyz.reshape(-1, 3)
    
    # Transformacja
    rgb_linear = np.dot(xyz_reshaped, matrix.T)
    rgb_linear = rgb_linear.reshape(original_shape)
    
    # Odwrotna gamma correction
    def gamma_correct_inv(c):
        return np.where(c > 0.0031308,
                       1.055 * np.power(c, 1.0/2.4) - 0.055,
                       12.92 * c)
    
    rgb_norm = gamma_correct_inv(rgb_linear)
    
    # Denormalizacja i clipping
    rgb = np.clip(rgb_norm * 255, 0, 255)
    
    return rgb.astype(np.uint8)

def lab_to_rgb(lab):
    """
    Bezpośrednia konwersja LAB → RGB
    """
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_rgb(xyz)
    return rgb
```

---

## Walidacja Konwersji

### Test Roundtrip
```python
def test_rgb_lab_roundtrip():
    """
    Test czy RGB → LAB → RGB zachowuje kolory
    """
    # Testowe kolory
    test_colors = np.array([
        [[255, 0, 0]],    # Czerwony
        [[0, 255, 0]],    # Zielony
        [[0, 0, 255]],    # Niebieski
        [[255, 255, 255]], # Biały
        [[0, 0, 0]],      # Czarny
        [[128, 128, 128]] # Szary
    ], dtype=np.uint8)
    
    # Konwersja roundtrip
    lab = rgb_to_lab(test_colors)
    rgb_back = lab_to_rgb(lab)
    
    # Sprawdź różnice
    diff = np.abs(test_colors.astype(float) - rgb_back.astype(float))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Maksymalna różnica: {max_diff}")
    print(f"Średnia różnica: {mean_diff}")
    
    # Akceptowalna różnica: ±2 (błąd zaokrąglenia)
    assert max_diff <= 2, f"Zbyt duża różnica w roundtrip: {max_diff}"
    
    return True

# Test
test_rgb_lab_roundtrip()
print("✅ Test roundtrip RGB↔LAB przeszedł pomyślnie")
```

---

## Podsumowanie Części 1

W tej części omówiliśmy:

1. **Podstawy teoretyczne** przestrzeni kolorów LAB
2. **Matematyczne formuły** konwersji RGB ↔ LAB
3. **Implementację** podstawowych funkcji konwersji
4. **Metryki jakości** (Delta E) w przestrzeni LAB
5. **Walidację** poprawności konwersji

### Co dalej?

**Część 2** będzie zawierać:
- Implementację algorytmów transferu kolorów
- Optymalizacje wydajności
- Zaawansowane techniki (lokalny transfer, adaptacyjne wagi)

**Część 3** będzie zawierać:
- Testy i benchmarki
- Przypadki użycia
- Rozwiązywanie problemów
- Integrację z systemem

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Część 1 - Podstawy teoretyczne