# ACES Color Space Transfer - Część 1of6: Teoria i Podstawy 🎓

> **Seria:** ACES Color Space Transfer  
> **Część:** 1 z 6 - Teoria i Podstawy  
> **Wymagania:** Brak (punkt startowy)  
> **Następna część:** [2of6 - Pseudokod i Architektura](gatto-WORKING-03-algorithms-08-advanced-01-aces-2of6.md)

---

## 1. Przegląd Algorytmu ACES

### Czym jest ACES?
ACES (Academy Color Encoding System) to standardowa przestrzeń kolorów opracowana przez Academy of Motion Picture Arts and Sciences. Jest to zaawansowany algorytm dopasowania kolorów, który zapewnia precyzyjne i spójne transfery kolorów między różnymi obrazami.

### Kluczowe Cechy ACES
- **Standardowa przestrzeń kolorów**: Międzynarodowy standard przemysłu filmowego
- **Szeroki gamut**: Pokrywa więcej kolorów niż sRGB czy Adobe RGB
- **Matematyczna precyzja**: Liniowe transformacje bez utraty informacji
- **HDR Ready**: Natywne wsparcie dla High Dynamic Range
- **Future-proof**: Zaprojektowane z myślą o przyszłych technologiach

### Zastosowania
- **Post-produkcja filmowa**: Dopasowanie kolorów między scenami
- **Fotografia komercyjna**: Spójność kolorów w seriach zdjęć
- **Archiwizacja cyfrowa**: Zachowanie autentycznych kolorów
- **Grading kolorów**: Profesjonalne korekcje tonalne
- **VFX i CGI**: Integracja elementów cyfrowych z materiałem filmowym

### Zalety vs Wady

**✅ Zalety:**
- Wysoka precyzja kolorów
- Standardowa przestrzeń kolorów
- Zachowanie szczegółów w cieniach i światłach
- Kompatybilność z workflow filmowym
- Matematycznie spójna transformacja

**❌ Wady:**
- Wysoka złożoność obliczeniowa
- Wymaga precyzyjnych profili kolorów
- Długi czas przetwarzania
- Duże wymagania pamięciowe
- Skomplikowana implementacja

---

## 2. Podstawy Teoretyczne Przestrzeni Kolorów

### Przestrzeń Kolorów ACES AP0

ACES wykorzystuje przestrzeń kolorów AP0 (ACES Primaries 0) z następującymi charakterystykami:

```
Primaries (CIE 1931 xy chromaticity):
- Red:   x=0.7347, y=0.2653
- Green: x=0.0000, y=1.0000  
- Blue:  x=0.0001, y=-0.0770
- White Point: x=0.32168, y=0.33767 (D60)

Gamma: Linear (1.0)
Bit Depth: 16-bit half-float
Range: [0, 65504]
Color Temperature: 6000K (D60)
```

### Porównanie z Innymi Przestrzeniami

| Przestrzeń | Gamut Coverage | Bit Depth | Gamma | White Point |
|------------|----------------|-----------|-------|-------------|
| sRGB       | 35.9%         | 8-bit     | 2.2   | D65 (6500K) |
| Adobe RGB  | 52.1%         | 8-bit     | 2.2   | D65 (6500K) |
| ProPhoto   | 90.0%         | 16-bit    | 1.8   | D50 (5000K) |
| **ACES AP0** | **100%+**   | **16-bit** | **1.0** | **D60 (6000K)** |

### Matematyczne Podstawy

#### Transformacja Chromatyczności
Przestrzeń ACES AP0 definiowana jest przez macierz transformacji:

```
ACES_AP0_Matrix = [
    [0.9525523959, 0.0000000000, 0.0000936786],
    [0.3439664498, 0.7281660966, -0.0721325464],
    [0.0000000000, 0.0000000000, 1.0088251844]
]
```

#### Illuminant D60
ACES używa illuminant D60 zamiast standardowego D65:

```python
# D60 Illuminant (ACES standard)
D60_xy = (0.32168, 0.33767)
D60_XYZ = (0.95265, 1.00000, 1.00882)

# D65 Illuminant (sRGB standard)
D65_xy = (0.31271, 0.32902)
D65_XYZ = (0.95047, 1.00000, 1.08883)
```

---

## 3. Transformacje Kolorów

### 3.1 sRGB → ACES AP0

Transformacja z sRGB do ACES AP0 wymaga dwóch kroków:

#### Krok 1: sRGB → Linear RGB
```python
def srgb_to_linear(srgb_value):
    if srgb_value <= 0.04045:
        return srgb_value / 12.92
    else:
        return pow((srgb_value + 0.055) / 1.055, 2.4)
```

#### Krok 2: Linear RGB → ACES AP0
```
Matrix sRGB_to_ACES = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
] × [
    [1.7166511, -0.3556708, -0.2533663],
    [-0.6666844, 1.6164812, 0.0157685],
    [0.0176399, -0.0427706, 0.9421031]
]

= [
    [0.6131, 0.0701, 0.0206],
    [0.3395, 0.9168, 0.1095],
    [0.0474, 0.0131, 0.8699]
]
```

### 3.2 ACES AP0 → sRGB

Transformacja odwrotna:

#### Krok 1: ACES AP0 → Linear RGB
```
Matrix ACES_to_sRGB = inverse(sRGB_to_ACES) = [
    [1.7050, -0.1302, -0.0240],
    [-0.6217, 1.1408, -0.1289],
    [-0.0833, -0.0106, 1.1529]
]
```

#### Krok 2: Linear RGB → sRGB
```python
def linear_to_srgb(linear_value):
    if linear_value <= 0.0031308:
        return linear_value * 12.92
    else:
        return 1.055 * pow(linear_value, 1.0/2.4) - 0.055
```

### 3.3 Chromatic Adaptation

Przejście między różnymi illuminantami (D65 ↔ D60):

```python
# Bradford Adaptation Matrix
Bradford_Matrix = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296]
]

# Adaptation D65 → D60
D65_to_D60 = calculate_adaptation_matrix(D65_XYZ, D60_XYZ)
```

---

## 4. Tone Mapping w ACES

### Reference Rendering Transform (RRT)

RRT to kluczowy element ACES, który przekształca scene-referred data na display-referred:

```python
def apply_rrt(aces_color):
    # RRT Parameters
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    # RRT Formula
    rrt_color = (aces_color * (a * aces_color + b)) / \
                (aces_color * (c * aces_color + d) + e)
    
    return rrt_color
```

### Output Device Transform (ODT)

ODT dostosowuje obraz do konkretnego urządzenia wyświetlającego:

```python
def apply_odt(rrt_color, device_profile):
    # Device-specific parameters
    gamma = device_profile.gamma
    white_point = device_profile.white_point
    primaries = device_profile.primaries
    
    # Transform to device space
    device_matrix = calculate_device_matrix(primaries, white_point)
    device_color = device_matrix @ rrt_color
    
    # Apply gamma correction
    output_color = pow(device_color, 1.0/gamma)
    
    return output_color
```

### Kompletny Pipeline ACES

```
Scene Linear → ACES AP0 → RRT → ODT → Display
     ↑            ↑        ↑     ↑       ↑
  Camera      Working   Tone  Device  Monitor
   Data       Space   Mapping Transform
```

---

## 5. Matematyczne Fundamenty

### 5.1 Macierze Transformacji

Wszystkie transformacje kolorów w ACES opierają się na mnożeniu macierzowym:

```
[R']   [M11 M12 M13] [R]
[G'] = [M21 M22 M23] [G]
[B']   [M31 M32 M33] [B]
```

### 5.2 Zachowanie Luminancji

Luminancja w ACES obliczana jest według wzoru:

```
Y = 0.2722287168 × R + 0.6740817658 × G + 0.0536895174 × B
```

### 5.3 Gamut Mapping

Dla kolorów poza gamutem ACES stosuje się:

```python
def gamut_compress(color, threshold=0.9):
    # Calculate color magnitude
    magnitude = sqrt(color.r² + color.g² + color.b²)
    
    if magnitude > threshold:
        # Soft compression
        compression_factor = threshold + \
                           (1 - threshold) * tanh((magnitude - threshold) / (1 - threshold))
        color *= compression_factor / magnitude
    
    return color
```

---

## 6. Praktyczne Implikacje

### 6.1 Wymagania Pamięciowe

```python
# Dla obrazu 4K (3840×2160)
width, height = 3840, 2160
channels = 3
bytes_per_channel = 2  # 16-bit half-float

total_memory = width × height × channels × bytes_per_channel
print(f"Pamięć: {total_memory / (1024**2):.1f} MB")  # ~95 MB
```

### 6.2 Precyzja Obliczeń

```python
# Rekomendowane typy danych
import numpy as np

# Dla obliczeń pośrednich
float32_precision = np.float32  # 7 cyfr znaczących
float64_precision = np.float64  # 15 cyfr znaczących

# Dla przechowywania ACES
half_float = np.float16  # 3-4 cyfry znaczące, zakres [0, 65504]
```

### 6.3 Optymalizacja Wydajności

```python
# Vectorized operations (NumPy)
def fast_aces_transform(image_array, transform_matrix):
    # Reshape for matrix multiplication
    pixels = image_array.reshape(-1, 3)
    
    # Single matrix operation for all pixels
    transformed = (transform_matrix @ pixels.T).T
    
    return transformed.reshape(image_array.shape)
```

---

## 7. Podsumowanie Teoretyczne

### Kluczowe Koncepty
1. **ACES AP0**: Najszersza przestrzeń kolorów z liniową charakterystyką
2. **RRT/ODT**: Dwuetapowy tone mapping dla różnych urządzeń
3. **D60 Illuminant**: Standardowy punkt bieli dla ACES
4. **16-bit Half-Float**: Optymalna precyzja dla obliczeń ACES
5. **Macierze Transformacji**: Matematyczna podstawa wszystkich konwersji

### Przygotowanie do Implementacji
Po zrozumieniu teorii, następne kroki to:
1. **Część 2of6**: Szczegółowy pseudokod algorytmów
2. **Część 3of6**: Implementacja klasy `ACESColorTransfer`
3. **Część 4of6**: Zaawansowane funkcje i optymalizacje

---

## 📚 Bibliografia i Dalsze Czytanie

- **ACES Documentation**: [ACESCentral.com](https://acescentral.com)
- **SMPTE Standards**: ST 2065-1, ST 2065-2, ST 2065-3
- **CIE Publications**: CIE 15:2004 (Colorimetry)
- **Research Papers**: "ACES Color Management" (Academy 2014)

---

**Następna część:** [2of6 - Pseudokod i Architektura](gatto-WORKING-03-algorithms-08-advanced-01-aces-2of6.md)  
**Powrót do:** [Spis treści](gatto-WORKING-03-algorithms-08-advanced-01-aces-0of6.md)

*Część 1of6 - Teoria i Podstawy | Wersja 1.0 | 2024-01-20*