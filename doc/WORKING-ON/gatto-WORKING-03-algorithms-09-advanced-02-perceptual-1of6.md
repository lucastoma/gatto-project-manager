# Perceptual Color Matching Algorithm - Teoria i Podstawy Percepcyjne

**Część 1 z 6: Teoria i Podstawy Percepcyjne**

---

## Spis Treści Części

1. [Wprowadzenie do Percepcji Kolorów](#wprowadzenie-do-percepcji-kolorów)
2. [Przestrzenie Kolorów Percepcyjnych](#przestrzenie-kolorów-percepcyjnych)
3. [Metryki Różnic Kolorów](#metryki-różnic-kolorów)
4. [Memory Colors - Kolory Pamięciowe](#memory-colors---kolory-pamięciowe)
5. [Modele Adaptacji Chromatycznej](#modele-adaptacji-chromatycznej)
6. [Funkcje Wagowe Percepcyjne](#funkcje-wagowe-percepcyjne)
7. [Warunki Obserwacji](#warunki-obserwacji)
8. [Podsumowanie Teoretyczne](#podsumowanie-teoretyczne)

---

## Wprowadzenie do Percepcji Kolorów

### Podstawy Fizjologiczne

Percepcja kolorów to złożony proces neurobiologiczny, który przekształca sygnały świetlne w subiektywne wrażenia kolorystyczne. Algorytm Perceptual Color Matching wykorzystuje matematyczne modele tego procesu.

#### 🧠 **System Wzrokowy Człowieka**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Światło   │───▶│  Receptory  │───▶│   Mózg      │───▶│  Percepcja  │
│   (Widmo)   │    │   (L,M,S)   │    │ (Przetwarz.)│    │   Koloru    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

#### 📊 **Charakterystyki Percepcyjne**

```python
class PerceptualCharacteristics:
    """Charakterystyki percepcyjne systemu wzrokowego"""
    
    def __init__(self):
        # Czułość spektralna receptorów
        self.L_cone_peak = 564  # nm (czerwone)
        self.M_cone_peak = 534  # nm (zielone)
        self.S_cone_peak = 420  # nm (niebieskie)
        
        # Progi percepcyjne
        self.just_noticeable_difference = {
            'lightness': 1.0,    # Delta L*
            'chroma': 1.0,       # Delta C*
            'hue': 1.0           # Delta H*
        }
        
        # Adaptacja chromatyczna
        self.adaptation_factors = {
            'complete': 1.0,     # Pełna adaptacja
            'partial': 0.7,      # Częściowa adaptacja
            'none': 0.0          # Brak adaptacji
        }

    def calculate_cone_responses(self, spectrum):
        """Obliczenie odpowiedzi receptorów L, M, S"""
        # Funkcje czułości spektralnej
        L_response = np.trapz(spectrum * self.L_sensitivity, dx=1)
        M_response = np.trapz(spectrum * self.M_sensitivity, dx=1)
        S_response = np.trapz(spectrum * self.S_sensitivity, dx=1)
        
        return np.array([L_response, M_response, S_response])
```

### Nieliniowość Percepcji

Percepcja kolorów jest nieliniowa - równe różnice fizyczne nie przekładają się na równe różnice percepcyjne.

#### 📈 **Prawo Webera-Fechnera**

```python
def weber_fechner_law(stimulus_intensity, weber_constant=0.01):
    """Prawo Webera-Fechnera dla percepcji"""
    # ΔI/I = k (gdzie k to stała Webera)
    perceived_intensity = np.log(stimulus_intensity / weber_constant)
    return perceived_intensity

def stevens_power_law(stimulus_intensity, exponent=0.33):
    """Prawo potęgowe Stevensa (bardziej dokładne)"""
    # Ψ = k * I^n
    perceived_intensity = np.power(stimulus_intensity, exponent)
    return perceived_intensity
```

---

## Przestrzenie Kolorów Percepcyjnych

### CIE LAB (L*a*b*)

#### 🎯 **Podstawy Teoretyczne**

Przestrzeń CIE LAB została zaprojektowana jako percepcyjnie jednorodna - równe odległości euklidesowe odpowiadają równym różnicom percepcyjnym.

```python
class CIELABColorSpace:
    """Implementacja przestrzeni kolorów CIE LAB"""
    
    def __init__(self, illuminant='D65', observer='2'):
        self.illuminant = illuminant
        self.observer = observer
        
        # Punkt białego dla illuminanta D65
        self.white_point = {
            'D65': {'X': 95.047, 'Y': 100.000, 'Z': 108.883},
            'D50': {'X': 96.422, 'Y': 100.000, 'Z': 82.521},
            'A': {'X': 109.850, 'Y': 100.000, 'Z': 35.585}
        }
    
    def xyz_to_lab(self, xyz):
        """Konwersja XYZ → LAB"""
        # Normalizacja względem punktu białego
        wp = self.white_point[self.illuminant]
        x_norm = xyz[..., 0] / wp['X']
        y_norm = xyz[..., 1] / wp['Y']
        z_norm = xyz[..., 2] / wp['Z']
        
        # Funkcja f(t)
        def f(t):
            delta = 6/29
            return np.where(
                t > delta**3,
                np.cbrt(t),
                t / (3 * delta**2) + 4/29
            )
        
        fx = f(x_norm)
        fy = f(y_norm)
        fz = f(z_norm)
        
        # Obliczenie L*, a*, b*
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.stack([L, a, b], axis=-1)
    
    def lab_to_xyz(self, lab):
        """Konwersja LAB → XYZ"""
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        
        # Obliczenie f(x), f(y), f(z)
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        # Funkcja odwrotna f^(-1)(t)
        def f_inv(t):
            delta = 6/29
            return np.where(
                t > delta,
                t**3,
                3 * delta**2 * (t - 4/29)
            )
        
        # Denormalizacja
        wp = self.white_point[self.illuminant]
        x = f_inv(fx) * wp['X']
        y = f_inv(fy) * wp['Y']
        z = f_inv(fz) * wp['Z']
        
        return np.stack([x, y, z], axis=-1)
    
    def calculate_chroma_hue(self, lab):
        """Obliczenie chroma i hue z LAB"""
        a, b = lab[..., 1], lab[..., 2]
        
        chroma = np.sqrt(a**2 + b**2)
        hue = np.arctan2(b, a) * 180 / np.pi
        
        # Normalizacja hue do [0, 360)
        hue = np.where(hue < 0, hue + 360, hue)
        
        return chroma, hue
```

#### 📊 **Charakterystyki LAB**

| Składowa | Zakres | Znaczenie | Percepcyjna Interpretacja |
|----------|--------|-----------|---------------------------|
| L* | 0-100 | Lightness | Jasność (0=czarny, 100=biały) |
| a* | -128 do +127 | Green-Red | Zielony ← → Czerwony |
| b* | -128 do +127 | Blue-Yellow | Niebieski ← → Żółty |

### CAM16-UCS (Uniform Color Space)

#### 🧠 **Zaawansowany Model Percepcji**

CAM16-UCS to najnowszy model percepcji kolorów, uwzględniający warunki obserwacji.

```python
class CAM16UCSColorSpace:
    """Implementacja przestrzeni CAM16-UCS"""
    
    def __init__(self, viewing_conditions=None):
        if viewing_conditions is None:
            # Standardowe warunki obserwacji
            self.viewing_conditions = {
                'white_point': [95.047, 100.0, 108.883],  # D65
                'adapting_luminance': 64,  # cd/m²
                'background_luminance': 20,  # % białego
                'surround': 'average',  # 'average', 'dim', 'dark'
                'discounting': False
            }
        else:
            self.viewing_conditions = viewing_conditions
    
    def xyz_to_cam16(self, xyz):
        """Konwersja XYZ → CAM16"""
        # Implementacja pełnego modelu CAM16
        # (uproszczona wersja dla demonstracji)
        
        # 1. Adaptacja chromatyczna (CAT16)
        xyz_adapted = self._chromatic_adaptation(xyz)
        
        # 2. Konwersja do przestrzeni LMS
        lms = self._xyz_to_lms(xyz_adapted)
        
        # 3. Nieliniowa kompresja
        lms_compressed = self._nonlinear_compression(lms)
        
        # 4. Obliczenie atrybutów percepcyjnych
        lightness = self._calculate_lightness(lms_compressed)
        chroma = self._calculate_chroma(lms_compressed)
        hue = self._calculate_hue(lms_compressed)
        
        return {
            'J': lightness,  # Lightness
            'C': chroma,     # Chroma
            'h': hue,        # Hue
            'Q': lightness * np.sqrt(self.viewing_conditions['adapting_luminance'] / 64),  # Brightness
            'M': chroma * np.sqrt(self.viewing_conditions['adapting_luminance'] / 64),     # Colorfulness
            's': 100 * np.sqrt(chroma / lightness) if lightness > 0 else 0                # Saturation
        }
    
    def cam16_to_ucs(self, cam16):
        """Konwersja CAM16 → UCS (Uniform Color Space)"""
        J, C, h = cam16['J'], cam16['C'], cam16['h']
        
        # Konwersja do przestrzeni jednorodnej
        # Formuły UCS dla CAM16
        M = C  # Dla uproszczenia
        
        # Współrzędne kartezjańskie w UCS
        a_ucs = M * np.cos(np.radians(h))
        b_ucs = M * np.sin(np.radians(h))
        
        return np.stack([J, a_ucs, b_ucs], axis=-1)
    
    def _chromatic_adaptation(self, xyz):
        """Adaptacja chromatyczna CAT16"""
        # Macierz transformacji CAT16
        cat16_matrix = np.array([
            [0.401288, 0.650173, -0.051461],
            [-0.250268, 1.204414, 0.045854],
            [-0.002079, 0.048952, 0.953127]
        ])
        
        # Transformacja do przestrzeni CAT16
        lms_cat16 = np.dot(xyz, cat16_matrix.T)
        
        # Adaptacja (uproszczona)
        adapted_lms = lms_cat16 * self._adaptation_factors()
        
        # Transformacja z powrotem do XYZ
        adapted_xyz = np.dot(adapted_lms, np.linalg.inv(cat16_matrix).T)
        
        return adapted_xyz
```

#### 🔄 **Porównanie LAB vs CAM16-UCS**

| Aspekt | CIE LAB | CAM16-UCS |
|--------|---------|----------|
| **Jednorodność** | Dobra | Bardzo dobra |
| **Warunki obserwacji** | Stałe | Konfigurowalne |
| **Złożoność** | Niska | Wysoka |
| **Dokładność** | Standardowa | Najwyższa |
| **Wydajność** | Szybka | Wolniejsza |
| **Zastosowania** | Ogólne | Specjalistyczne |

---

## Metryki Różnic Kolorów

### Delta E 2000 (ΔE₀₀)

#### 🎯 **Najdokładniejsza Metryka**

Delta E 2000 to obecnie najdokładniejsza metryka różnic kolorów, uwzględniająca nieliniowości percepcji.

```python
class DeltaE2000Calculator:
    """Kalkulator Delta E 2000"""
    
    def __init__(self):
        # Parametry wagowe
        self.kL = 1.0  # Waga lightness
        self.kC = 1.0  # Waga chroma
        self.kH = 1.0  # Waga hue
    
    def calculate_delta_e_2000(self, lab1, lab2):
        """Obliczenie Delta E 2000"""
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
        
        # 1. Obliczenie C* i h*
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        
        C_mean = (C1 + C2) / 2
        
        # 2. Korekcja a* (G factor)
        G = 0.5 * (1 - np.sqrt(C_mean**7 / (C_mean**7 + 25**7)))
        
        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2
        
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        
        h1_prime = np.arctan2(b1, a1_prime) * 180 / np.pi
        h2_prime = np.arctan2(b2, a2_prime) * 180 / np.pi
        
        # Normalizacja hue do [0, 360)
        h1_prime = np.where(h1_prime < 0, h1_prime + 360, h1_prime)
        h2_prime = np.where(h2_prime < 0, h2_prime + 360, h2_prime)
        
        # 3. Obliczenie różnic
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime
        
        # Delta h' (uwzględnienie cykliczności)
        delta_h_prime = self._calculate_delta_h_prime(h1_prime, h2_prime, C1_prime, C2_prime)
        delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2))
        
        # 4. Średnie wartości
        L_mean = (L1 + L2) / 2
        C_prime_mean = (C1_prime + C2_prime) / 2
        h_prime_mean = self._calculate_h_prime_mean(h1_prime, h2_prime, C1_prime, C2_prime)
        
        # 5. Funkcje wagowe
        T = (1 - 0.17 * np.cos(np.radians(h_prime_mean - 30)) +
             0.24 * np.cos(np.radians(2 * h_prime_mean)) +
             0.32 * np.cos(np.radians(3 * h_prime_mean + 6)) -
             0.20 * np.cos(np.radians(4 * h_prime_mean - 63)))
        
        delta_theta = 30 * np.exp(-((h_prime_mean - 275) / 25)**2)
        
        RC = 2 * np.sqrt(C_prime_mean**7 / (C_prime_mean**7 + 25**7))
        
        SL = 1 + (0.015 * (L_mean - 50)**2) / np.sqrt(20 + (L_mean - 50)**2)
        SC = 1 + 0.045 * C_prime_mean
        SH = 1 + 0.015 * C_prime_mean * T
        
        RT = -np.sin(2 * np.radians(delta_theta)) * RC
        
        # 6. Delta E 2000
        delta_e_2000 = np.sqrt(
            (delta_L_prime / (self.kL * SL))**2 +
            (delta_C_prime / (self.kC * SC))**2 +
            (delta_H_prime / (self.kH * SH))**2 +
            RT * (delta_C_prime / (self.kC * SC)) * (delta_H_prime / (self.kH * SH))
        )
        
        return delta_e_2000
    
    def _calculate_delta_h_prime(self, h1, h2, C1, C2):
        """Obliczenie delta h' z uwzględnieniem cykliczności"""
        # Jeśli którykolwiek chroma = 0, delta h' = 0
        mask = (C1 * C2) == 0
        
        delta_h = h2 - h1
        
        # Uwzględnienie cykliczności (360°)
        delta_h = np.where(np.abs(delta_h) <= 180, delta_h,
                          np.where(delta_h > 180, delta_h - 360, delta_h + 360))
        
        return np.where(mask, 0, delta_h)
    
    def _calculate_h_prime_mean(self, h1, h2, C1, C2):
        """Obliczenie średniego hue z uwzględnieniem cykliczności"""
        # Jeśli którykolwiek chroma = 0, używamy drugiego hue
        mask_C1_zero = C1 == 0
        mask_C2_zero = C2 == 0
        
        h_mean = (h1 + h2) / 2
        
        # Korekcja dla różnic > 180°
        diff = np.abs(h1 - h2)
        h_mean = np.where(
            diff > 180,
            np.where(h_mean < 180, h_mean + 180, h_mean - 180),
            h_mean
        )
        
        # Obsługa przypadków zerowego chroma
        h_mean = np.where(mask_C1_zero, h2, h_mean)
        h_mean = np.where(mask_C2_zero, h1, h_mean)
        
        return h_mean
```

#### 📊 **Interpretacja Delta E 2000**

| Wartość ΔE₀₀ | Percepcyjna Różnica | Zastosowanie |
|--------------|-------------------|-------------|
| 0.0 - 1.0 | Niezauważalna | Kalibracja premium |
| 1.0 - 2.3 | Ledwo zauważalna | Profesjonalne |
| 2.3 - 5.0 | Zauważalna | Komercyjne |
| 5.0 - 10.0 | Wyraźna | Podstawowe |
| > 10.0 | Bardzo wyraźna | Nieakceptowalne |

### Delta E CAM16

```python
def calculate_delta_e_cam16(cam16_1, cam16_2):
    """Obliczenie Delta E w przestrzeni CAM16-UCS"""
    # Konwersja do UCS
    ucs1 = cam16_to_ucs(cam16_1)
    ucs2 = cam16_to_ucs(cam16_2)
    
    # Euklidesowa odległość w przestrzeni UCS
    delta_e_cam16 = np.sqrt(
        (ucs2[..., 0] - ucs1[..., 0])**2 +  # Delta J'
        (ucs2[..., 1] - ucs1[..., 1])**2 +  # Delta a'
        (ucs2[..., 2] - ucs1[..., 2])**2    # Delta b'
    )
    
    return delta_e_cam16
```

---

## Memory Colors - Kolory Pamięciowe

### Koncepcja Memory Colors

#### 🧠 **Definicja**
Memory colors to kolory obiektów, które mamy zapamiętane i które wpływają na naszą percepcję. Są to kolory "idealne" w naszej pamięci.

```python
class MemoryColors:
    """Baza danych memory colors"""
    
    def __init__(self):
        # Memory colors w przestrzeni LAB
        self.memory_colors_lab = {
            # Odcienie skóry
            'skin_caucasian': {'L': 70, 'a': 15, 'b': 25, 'tolerance': 10},
            'skin_asian': {'L': 65, 'a': 12, 'b': 20, 'tolerance': 8},
            'skin_african': {'L': 45, 'a': 8, 'b': 15, 'tolerance': 12},
            
            # Kolory naturalne
            'sky_blue': {'L': 70, 'a': -5, 'b': -25, 'tolerance': 15},
            'grass_green': {'L': 50, 'a': -40, 'b': 35, 'tolerance': 20},
            'ocean_blue': {'L': 45, 'a': 5, 'b': -35, 'tolerance': 18},
            
            # Kolory żywności
            'apple_red': {'L': 45, 'a': 55, 'b': 35, 'tolerance': 12},
            'banana_yellow': {'L': 85, 'a': -5, 'b': 75, 'tolerance': 15},
            'orange_fruit': {'L': 65, 'a': 35, 'b': 65, 'tolerance': 10},
            
            # Kolory neutralne
            'white_paper': {'L': 95, 'a': 0, 'b': 2, 'tolerance': 5},
            'black_text': {'L': 15, 'a': 0, 'b': 0, 'tolerance': 8},
            'gray_neutral': {'L': 50, 'a': 0, 'b': 0, 'tolerance': 3}
        }
    
    def identify_memory_colors(self, lab_image, threshold=15):
        """Identyfikacja memory colors w obrazie"""
        identified_colors = {}
        
        for color_name, color_data in self.memory_colors_lab.items():
            # Obliczenie odległości do memory color
            target_lab = np.array([color_data['L'], color_data['a'], color_data['b']])
            
            distances = np.sqrt(
                ((lab_image[..., 0] - target_lab[0]) / 1.0)**2 +  # L* waga 1.0
                ((lab_image[..., 1] - target_lab[1]) / 1.0)**2 +  # a* waga 1.0
                ((lab_image[..., 2] - target_lab[2]) / 1.0)**2    # b* waga 1.0
            )
            
            # Maska pikseli należących do memory color
            mask = distances < color_data['tolerance']
            
            if np.any(mask):
                # Statystyki memory color w obrazie
                pixels = lab_image[mask]
                identified_colors[color_name] = {
                    'mask': mask,
                    'pixel_count': np.sum(mask),
                    'percentage': np.sum(mask) / lab_image.size * 100,
                    'mean_lab': np.mean(pixels, axis=0),
                    'std_lab': np.std(pixels, axis=0),
                    'target_lab': target_lab,
                    'mean_distance': np.mean(distances[mask])
                }
        
        return identified_colors
    
    def calculate_memory_color_weights(self, lab_image):
        """Obliczenie wag dla memory colors"""
        weights = np.ones(lab_image.shape[:2], dtype=np.float32)
        
        identified = self.identify_memory_colors(lab_image)
        
        for color_name, color_info in identified.items():
            # Wyższa waga dla memory colors
            if color_name.startswith('skin_'):
                weights[color_info['mask']] = 3.0  # Najwyższa waga dla skóry
            elif color_name in ['sky_blue', 'grass_green']:
                weights[color_info['mask']] = 2.0  # Wysoka waga dla natury
            else:
                weights[color_info['mask']] = 1.5  # Standardowa waga
        
        return weights
    
    def preserve_memory_colors(self, source_lab, target_lab, result_lab, strength=0.8):
        """Zachowanie memory colors w wyniku"""
        preserved_result = result_lab.copy()
        
        # Identyfikacja memory colors w źródle i celu
        source_memory = self.identify_memory_colors(source_lab)
        target_memory = self.identify_memory_colors(target_lab)
        
        for color_name in source_memory.keys():
            if color_name in target_memory:
                source_mask = source_memory[color_name]['mask']
                target_mean = target_memory[color_name]['mean_lab']
                
                # Mieszanie z target memory color
                preserved_result[source_mask] = (
                    (1 - strength) * result_lab[source_mask] +
                    strength * target_mean
                )
        
        return preserved_result
```

#### 🎨 **Wizualizacja Memory Colors**

```python
def visualize_memory_colors(lab_image, memory_colors):
    """Wizualizacja zidentyfikowanych memory colors"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Oryginalny obraz
    rgb_image = lab_to_rgb(lab_image)
    axes[0].imshow(rgb_image)
    axes[0].set_title('Oryginalny obraz')
    axes[0].axis('off')
    
    # Maski memory colors
    identified = memory_colors.identify_memory_colors(lab_image)
    
    for i, (color_name, color_info) in enumerate(identified.items()):
        if i < 5:  # Maksymalnie 5 memory colors
            mask_viz = np.zeros_like(rgb_image)
            mask_viz[color_info['mask']] = [1, 1, 1]  # Białe piksele
            
            axes[i+1].imshow(mask_viz)
            axes[i+1].set_title(f'{color_name}\n{color_info["percentage"]:.1f}%')
            axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## Modele Adaptacji Chromatycznej

### Bradford Transform

#### 🔄 **Najdokładniejszy Model**

```python
class BradfordAdaptation:
    """Model adaptacji chromatycznej Bradford"""
    
    def __init__(self):
        # Macierz transformacji Bradford
        self.bradford_matrix = np.array([
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296]
        ])
        
        self.bradford_matrix_inv = np.linalg.inv(self.bradford_matrix)
    
    def adapt_white_point(self, xyz, source_white, target_white):
        """Adaptacja do nowego punktu białego"""
        # Konwersja punktów białych do przestrzeni Bradford
        source_lms = np.dot(source_white, self.bradford_matrix.T)
        target_lms = np.dot(target_white, self.bradford_matrix.T)
        
        # Obliczenie współczynników adaptacji
        adaptation_matrix = np.diag(target_lms / source_lms)
        
        # Pełna macierz transformacji
        full_transform = np.dot(
            self.bradford_matrix_inv,
            np.dot(adaptation_matrix, self.bradford_matrix)
        )
        
        # Aplikacja transformacji
        adapted_xyz = np.dot(xyz, full_transform.T)
        
        return adapted_xyz
    
    def calculate_adaptation_degree(self, viewing_time_minutes):
        """Obliczenie stopnia adaptacji w czasie"""
        # Eksponencjalny model adaptacji
        # Pełna adaptacja po ~10 minutach
        max_adaptation = 1.0
        time_constant = 3.0  # minuty
        
        adaptation_degree = max_adaptation * (
            1 - np.exp(-viewing_time_minutes / time_constant)
        )
        
        return np.clip(adaptation_degree, 0, 1)
```

### Von Kries Adaptation

```python
class VonKriesAdaptation:
    """Klasyczny model Von Kries"""
    
    def __init__(self):
        # Macierz transformacji do przestrzeni LMS
        self.lms_matrix = np.array([
            [0.4002, 0.7075, -0.0807],
            [-0.2280, 1.1500, 0.0612],
            [0.0000, 0.0000, 0.9184]
        ])
        
        self.lms_matrix_inv = np.linalg.inv(self.lms_matrix)
    
    def adapt_simple(self, xyz, source_white, target_white):
        """Prosta adaptacja Von Kries"""
        # Konwersja do LMS
        xyz_lms = np.dot(xyz, self.lms_matrix.T)
        source_lms = np.dot(source_white, self.lms_matrix.T)
        target_lms = np.dot(target_white, self.lms_matrix.T)
        
        # Skalowanie każdego kanału
        adapted_lms = xyz_lms * (target_lms / source_lms)
        
        # Konwersja z powrotem do XYZ
        adapted_xyz = np.dot(adapted_lms, self.lms_matrix_inv.T)
        
        return adapted_xyz
```

---

## Funkcje Wagowe Percepcyjne

### Wagi Przestrzenne

```python
class PerceptualWeights:
    """Obliczanie wag percepcyjnych"""
    
    def __init__(self):
        self.memory_colors = MemoryColors()
    
    def calculate_spatial_weights(self, lab_image, method='gradient'):
        """Obliczenie wag przestrzennych"""
        if method == 'gradient':
            return self._gradient_weights(lab_image)
        elif method == 'saliency':
            return self._saliency_weights(lab_image)
        elif method == 'edge':
            return self._edge_weights(lab_image)
        else:
            return np.ones(lab_image.shape[:2])
    
    def _gradient_weights(self, lab_image):
        """Wagi oparte na gradiencie"""
        from scipy import ndimage
        
        # Gradient dla każdego kanału LAB
        grad_L = ndimage.sobel(lab_image[..., 0])
        grad_a = ndimage.sobel(lab_image[..., 1])
        grad_b = ndimage.sobel(lab_image[..., 2])
        
        # Magnitude gradientu
        gradient_magnitude = np.sqrt(grad_L**2 + grad_a**2 + grad_b**2)
        
        # Normalizacja do [0, 1]
        weights = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
        
        # Inwersja - mniejsze wagi dla dużych gradientów
        weights = 1.0 - weights
        
        return weights
    
    def _saliency_weights(self, lab_image):
        """Wagi oparte na saliency"""
        # Uproszczony model saliency
        L = lab_image[..., 0]
        
        # Kontrast lokalny
        from scipy import ndimage
        local_mean = ndimage.uniform_filter(L, size=15)
        local_contrast = np.abs(L - local_mean)
        
        # Normalizacja
        weights = local_contrast / (np.max(local_contrast) + 1e-8)
        
        return weights
    
    def calculate_perceptual_weights(self, lab_pixel):
        """Obliczenie wag percepcyjnych dla pojedynczego piksela"""
        L, a, b = lab_pixel[0], lab_pixel[1], lab_pixel[2]
        
        # Waga lightness (wyższa dla średnich wartości)
        lightness_weight = 1.0 - np.abs(L - 50) / 50
        lightness_weight = np.clip(lightness_weight, 0.3, 1.0)
        
        # Waga chroma (wyższa dla nasyconych kolorów)
        chroma = np.sqrt(a**2 + b**2)
        chroma_weight = np.tanh(chroma / 30)  # Saturacja
        
        # Waga hue (specjalne traktowanie dla krytycznych odcieni)
        hue = np.arctan2(b, a) * 180 / np.pi
        if hue < 0:
            hue += 360
        
        # Wyższa waga dla odcieni skóry (10-50°)
        if 10 <= hue <= 50:
            hue_weight = 2.0
        # Wyższa waga dla nieba (200-250°)
        elif 200 <= hue <= 250:
            hue_weight = 1.5
        else:
            hue_weight = 1.0
        
        return {
            'lightness': lightness_weight,
            'chroma': chroma_weight,
            'hue': hue_weight,
            'overall': (lightness_weight + chroma_weight + hue_weight) / 3
        }
```

---

## Warunki Obserwacji

### Standardowe Warunki

```python
class ViewingConditions:
    """Definicja warunków obserwacji"""
    
    def __init__(self):
        self.standard_conditions = {
            'D65_2deg': {
                'illuminant': 'D65',
                'observer': '2',
                'adapting_luminance': 64,  # cd/m²
                'background': 20,  # % reflectance
                'surround': 'average'
            },
            'D50_2deg': {
                'illuminant': 'D50',
                'observer': '2',
                'adapting_luminance': 64,
                'background': 20,
                'surround': 'average'
            },
            'tungsten': {
                'illuminant': 'A',
                'observer': '2',
                'adapting_luminance': 32,
                'background': 20,
                'surround': 'dim'
            }
        }
    
    def get_condition(self, name):
        """Pobranie predefiniowanych warunków"""
        return self.standard_conditions.get(name, self.standard_conditions['D65_2deg'])
    
    def calculate_adaptation_factor(self, current_luminance, target_luminance):
        """Obliczenie współczynnika adaptacji luminancji"""
        # Model Hunt-Pointer-Estevez
        ratio = target_luminance / current_luminance
        adaptation_factor = np.power(ratio, 0.42)
        
        return adaptation_factor
```

---

## Podsumowanie Teoretyczne

### Kluczowe Koncepcje

#### 🎯 **Percepcyjna Jednorodność**
- Równe odległości = równe różnice percepcyjne
- Podstawa dla dokładnych metryk
- Kluczowa dla naturalnych rezultatów

#### 🧠 **Memory Colors**
- Wpływ pamięci na percepcję
- Krytyczne dla naturalności
- Wymagają specjalnego traktowania

#### 🔄 **Adaptacja Chromatyczna**
- Dostosowanie do warunków oświetlenia
- Bradford jako standard przemysłowy
- Kluczowa dla spójności kolorów

#### ⚖️ **Wagi Percepcyjne**
- Różne regiony wymagają różnego traktowania
- Skóra, niebo, zieleń - priorytetowe
- Adaptacja do kontekstu obrazu

### Praktyczne Implikacje

```python
def theoretical_summary_example():
    """Przykład zastosowania teorii w praktyce"""
    
    # 1. Wybór przestrzeni kolorów
    color_space = 'lab'  # Dla szybkości
    # color_space = 'cam16ucs'  # Dla dokładności
    
    # 2. Metryka oceny
    metric = 'delta_e_2000'  # Standard przemysłowy
    
    # 3. Memory colors
    preserve_memory = True  # Zawsze zalecane
    memory_weight = 2.0     # Podwójna waga
    
    # 4. Adaptacja chromatyczna
    chromatic_adaptation = True
    adaptation_method = 'bradford'  # Najdokładniejszy
    
    # 5. Wagi percepcyjne
    use_perceptual_weights = True
    spatial_weighting = 'gradient'  # Lub 'saliency'
    
    return {
        'color_space': color_space,
        'metric': metric,
        'preserve_memory': preserve_memory,
        'memory_weight': memory_weight,
        'chromatic_adaptation': chromatic_adaptation,
        'adaptation_method': adaptation_method,
        'use_perceptual_weights': use_perceptual_weights,
        'spatial_weighting': spatial_weighting
    }
```

### Wybór Parametrów

| Zastosowanie | Przestrzeń | Metryka | Memory Colors | Adaptacja |
|--------------|------------|---------|---------------|----------|
| **Fotografia portretowa** | LAB | ΔE₀₀ | Wysoka waga | Bradford |
| **Krajobraz** | CAM16-UCS | ΔE_CAM16 | Średnia waga | Bradford |
| **Medycyna** | LAB | ΔE₀₀ | Bardzo wysoka | Bradford |
| **E-commerce** | LAB | ΔE₀₀ | Wysoka waga | Bradford |
| **Sztuka** | CAM16-UCS | ΔE_CAM16 | Zachowanie | Von Kries |

---

## Nawigacja

**◀️ Poprzednia część**: [Spis Treści i Wprowadzenie](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-0of6.md)  
**▶️ Następna część**: [Pseudokod i Architektura Systemu](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-2of6.md)  
**🏠 Powrót do**: [Spis Treści](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-0of6.md#spis-treści---kompletna-dokumentacja)

---

*Ostatnia aktualizacja: 2024-01-20*  
*Autor: GattoNero AI Assistant*  
*Wersja: 1.0*  
*Status: Dokumentacja kompletna* ✅