# Delta E Color Distance - Część 2: Podstawowa Implementacja

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 3-4 godziny | **Złożoność**: O(n)

---

## Kompletna Implementacja DeltaECalculator

### Klasa Główna

```python
import math
import numpy as np
from typing import Tuple, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

class DeltaEMethod(Enum):
    """Dostępne metody obliczania Delta E"""
    CIE76 = "cie76"
    CIE94 = "cie94"
    CIEDE2000 = "ciede2000"
    CMC = "cmc"

@dataclass
class DeltaEResult:
    """Wynik obliczenia Delta E"""
    value: float
    method: DeltaEMethod
    color1_lab: Tuple[float, float, float]
    color2_lab: Tuple[float, float, float]
    interpretation: str
    
    def __str__(self):
        return f"ΔE {self.method.value.upper()}: {self.value:.2f} ({self.interpretation})"

class DeltaECalculator:
    """Kalkulator różnic kolorów Delta E"""
    
    def __init__(self, method: DeltaEMethod = DeltaEMethod.CIEDE2000):
        """
        Inicjalizuje kalkulator Delta E
        
        Args:
            method: Metoda obliczania Delta E
        """
        self.method = method
        self._setup_parameters()
    
    def _setup_parameters(self):
        """Ustawia parametry dla różnych metod"""
        # Parametry CIE94
        self.cie94_params = {
            'graphics': {'kL': 1.0, 'kC': 1.0, 'kH': 1.0, 'K1': 0.045, 'K2': 0.015},
            'textiles': {'kL': 2.0, 'kC': 1.0, 'kH': 1.0, 'K1': 0.048, 'K2': 0.014}
        }
        
        # Parametry CIEDE2000
        self.ciede2000_params = {
            'kL': 1.0,
            'kC': 1.0,
            'kH': 1.0
        }
        
        # Parametry CMC
        self.cmc_params = {
            'l': 2.0,  # lightness factor
            'c': 1.0   # chroma factor
        }
    
    def calculate(self, color1: Tuple[float, float, float], 
                 color2: Tuple[float, float, float],
                 return_details: bool = False) -> Union[float, DeltaEResult]:
        """
        Oblicza Delta E między dwoma kolorami
        
        Args:
            color1: Pierwszy kolor (L*, a*, b*)
            color2: Drugi kolor (L*, a*, b*)
            return_details: Czy zwrócić szczegółowe informacje
        
        Returns:
            Wartość Delta E lub obiekt DeltaEResult
        """
        # Walidacja danych wejściowych
        self._validate_lab_color(color1, "color1")
        self._validate_lab_color(color2, "color2")
        
        # Obliczenie Delta E
        if self.method == DeltaEMethod.CIE76:
            delta_e = self._delta_e_76(color1, color2)
        elif self.method == DeltaEMethod.CIE94:
            delta_e = self._delta_e_94(color1, color2)
        elif self.method == DeltaEMethod.CIEDE2000:
            delta_e = self._delta_e_2000(color1, color2)
        elif self.method == DeltaEMethod.CMC:
            delta_e = self._delta_e_cmc(color1, color2)
        else:
            raise ValueError(f"Nieznana metoda: {self.method}")
        
        if return_details:
            interpretation = self._interpret_delta_e(delta_e)
            return DeltaEResult(
                value=delta_e,
                method=self.method,
                color1_lab=color1,
                color2_lab=color2,
                interpretation=interpretation
            )
        
        return delta_e
    
    def _validate_lab_color(self, color: Tuple[float, float, float], name: str):
        """Waliduje kolor LAB"""
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise ValueError(f"{name} musi być tuple/list z 3 elementami")
        
        L, a, b = color
        
        if not (0 <= L <= 100):
            raise ValueError(f"{name}: L* musi być w zakresie [0, 100], otrzymano {L}")
        
        if not (-128 <= a <= 127):
            raise ValueError(f"{name}: a* musi być w zakresie [-128, 127], otrzymano {a}")
        
        if not (-128 <= b <= 127):
            raise ValueError(f"{name}: b* musi być w zakresie [-128, 127], otrzymano {b}")
    
    def _interpret_delta_e(self, delta_e: float) -> str:
        """Interpretuje wartość Delta E"""
        if delta_e < 1:
            return "Niewidoczna różnica"
        elif delta_e < 2:
            return "Ledwo widoczna różnica"
        elif delta_e < 3.5:
            return "Widoczna przy porównaniu"
        elif delta_e < 5:
            return "Wyraźnie widoczna różnica"
        elif delta_e < 10:
            return "Znacząca różnica"
        else:
            return "Bardzo duża różnica"
    
    def _delta_e_76(self, color1: Tuple[float, float, float], 
                   color2: Tuple[float, float, float]) -> float:
        """Delta E 76 - prosta odległość euklidesowa"""
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        
        dL = L2 - L1
        da = a2 - a1
        db = b2 - b1
        
        return math.sqrt(dL*dL + da*da + db*db)
    
    def _delta_e_94(self, color1: Tuple[float, float, float], 
                   color2: Tuple[float, float, float],
                   application: str = 'graphics') -> float:
        """Delta E 94 - z wagami dla różnych składowych"""
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        
        # Parametry dla aplikacji
        params = self.cie94_params[application]
        kL, kC, kH = params['kL'], params['kC'], params['kH']
        K1, K2 = params['K1'], params['K2']
        
        # Różnice
        dL = L1 - L2
        da = a1 - a2
        db = b1 - b2
        
        # Chromatyczność
        C1 = math.sqrt(a1*a1 + b1*b1)
        C2 = math.sqrt(a2*a2 + b2*b2)
        dC = C1 - C2
        
        # Odcień
        dH_squared = da*da + db*db - dC*dC
        dH = math.sqrt(max(0, dH_squared))
        
        # Funkcje wagowe
        SL = 1.0
        SC = 1 + K1 * C1
        SH = 1 + K2 * C1
        
        # Delta E 94
        term1 = (dL / (kL * SL))**2
        term2 = (dC / (kC * SC))**2
        term3 = (dH / (kH * SH))**2
        
        return math.sqrt(term1 + term2 + term3)
    
    def _delta_e_2000(self, color1: Tuple[float, float, float], 
                     color2: Tuple[float, float, float]) -> float:
        """Delta E 2000 - najbardziej zaawansowana formuła"""
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        
        # Parametry
        kL = self.ciede2000_params['kL']
        kC = self.ciede2000_params['kC']
        kH = self.ciede2000_params['kH']
        
        # Średnie wartości
        L_mean = (L1 + L2) / 2
        
        # Chromatyczność
        C1 = math.sqrt(a1*a1 + b1*b1)
        C2 = math.sqrt(a2*a2 + b2*b2)
        C_mean = (C1 + C2) / 2
        
        # Korekcja a* dla neutralnych kolorów
        G = 0.5 * (1 - math.sqrt(C_mean**7 / (C_mean**7 + 25**7)))
        
        a1_prime = a1 * (1 + G)
        a2_prime = a2 * (1 + G)
        
        # Nowe wartości chromatyczności
        C1_prime = math.sqrt(a1_prime*a1_prime + b1*b1)
        C2_prime = math.sqrt(a2_prime*a2_prime + b2*b2)
        C_prime_mean = (C1_prime + C2_prime) / 2
        
        # Odcienie
        h1_prime = math.atan2(b1, a1_prime) * 180 / math.pi
        h2_prime = math.atan2(b2, a2_prime) * 180 / math.pi
        
        if h1_prime < 0:
            h1_prime += 360
        if h2_prime < 0:
            h2_prime += 360
        
        # Różnica odcienia
        dh_prime = h2_prime - h1_prime
        if abs(dh_prime) > 180:
            if h2_prime > h1_prime:
                dh_prime -= 360
            else:
                dh_prime += 360
        
        # Średni odcień
        if abs(h1_prime - h2_prime) > 180:
            H_prime_mean = (h1_prime + h2_prime + 360) / 2
        else:
            H_prime_mean = (h1_prime + h2_prime) / 2
        
        # Różnice
        dL_prime = L2 - L1
        dC_prime = C2_prime - C1_prime
        dH_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(dh_prime / 2))
        
        # Funkcje wagowe
        T = (1 - 0.17 * math.cos(math.radians(H_prime_mean - 30)) +
             0.24 * math.cos(math.radians(2 * H_prime_mean)) +
             0.32 * math.cos(math.radians(3 * H_prime_mean + 6)) -
             0.20 * math.cos(math.radians(4 * H_prime_mean - 63)))
        
        dTheta = 30 * math.exp(-((H_prime_mean - 275) / 25)**2)
        
        RC = 2 * math.sqrt(C_prime_mean**7 / (C_prime_mean**7 + 25**7))
        
        SL = 1 + (0.015 * (L_mean - 50)**2) / math.sqrt(20 + (L_mean - 50)**2)
        SC = 1 + 0.045 * C_prime_mean
        SH = 1 + 0.015 * C_prime_mean * T
        
        RT = -math.sin(math.radians(2 * dTheta)) * RC
        
        # Delta E 2000
        term1 = (dL_prime / (kL * SL))**2
        term2 = (dC_prime / (kC * SC))**2
        term3 = (dH_prime / (kH * SH))**2
        term4 = RT * (dC_prime / (kC * SC)) * (dH_prime / (kH * SH))
        
        return math.sqrt(term1 + term2 + term3 + term4)
    
    def _delta_e_cmc(self, color1: Tuple[float, float, float], 
                    color2: Tuple[float, float, float]) -> float:
        """Delta E CMC - dla przemysłu tekstylnego"""
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        
        # Różnice
        dL = L1 - L2
        da = a1 - a2
        db = b1 - b2
        
        # Chromatyczność i odcień
        C1 = math.sqrt(a1*a1 + b1*b1)
        C2 = math.sqrt(a2*a2 + b2*b2)
        dC = C1 - C2
        
        dH_squared = da*da + db*db - dC*dC
        dH = math.sqrt(max(0, dH_squared))
        
        # Odcień w stopniach
        H1 = math.atan2(b1, a1) * 180 / math.pi
        if H1 < 0:
            H1 += 360
        
        # Funkcje wagowe CMC
        F = math.sqrt(C1**4 / (C1**4 + 1900))
        
        if 164 <= H1 <= 345:
            T = 0.56 + abs(0.2 * math.cos(math.radians(H1 + 168)))
        else:
            T = 0.36 + abs(0.4 * math.cos(math.radians(H1 + 35)))
        
        if L1 < 16:
            SL = 0.511
        else:
            SL = 0.040975 * L1 / (1 + 0.01765 * L1)
        
        SC = (0.0638 * C1) / (1 + 0.0131 * C1) + 0.638
        SH = SC * (F * T + 1 - F)
        
        # Parametry CMC
        l = self.cmc_params['l']
        c = self.cmc_params['c']
        
        # Delta E CMC
        term1 = (dL / (l * SL))**2
        term2 = (dC / (c * SC))**2
        term3 = (dH / SH)**2
        
        return math.sqrt(term1 + term2 + term3)
```

---

## Konwersje Kolorów

### Klasa ColorConverter

```python
class ColorConverter:
    """Klasa do konwersji między przestrzeniami kolorów"""
    
    # Illuminant D65 (standard daylight)
    ILLUMINANT_D65 = (0.95047, 1.00000, 1.08883)
    
    @staticmethod
    def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Konwertuje RGB do LAB przez przestrzeń XYZ
        
        Args:
            rgb: Kolor RGB (0-255)
        
        Returns:
            Kolor LAB (L*: 0-100, a*: -128-127, b*: -128-127)
        """
        # Walidacja RGB
        r, g, b = rgb
        if not all(0 <= c <= 255 for c in [r, g, b]):
            raise ValueError(f"RGB musi być w zakresie [0, 255], otrzymano {rgb}")
        
        # Normalizacja do [0, 1]
        r, g, b = [x / 255.0 for x in rgb]
        
        # Korekcja gamma (sRGB)
        r = ColorConverter._gamma_correct(r)
        g = ColorConverter._gamma_correct(g)
        b = ColorConverter._gamma_correct(b)
        
        # RGB → XYZ (macierz sRGB)
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # Normalizacja do illuminant D65
        x = x / ColorConverter.ILLUMINANT_D65[0]
        y = y / ColorConverter.ILLUMINANT_D65[1]
        z = z / ColorConverter.ILLUMINANT_D65[2]
        
        # XYZ → LAB
        fx = ColorConverter._lab_f(x)
        fy = ColorConverter._lab_f(y)
        fz = ColorConverter._lab_f(z)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b_val = 200 * (fy - fz)
        
        return (L, a, b_val)
    
    @staticmethod
    def lab_to_rgb(lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        Konwertuje LAB do RGB przez przestrzeń XYZ
        
        Args:
            lab: Kolor LAB
        
        Returns:
            Kolor RGB (0-255)
        """
        L, a, b = lab
        
        # LAB → XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        x = ColorConverter.ILLUMINANT_D65[0] * ColorConverter._lab_f_inv(fx)
        y = ColorConverter.ILLUMINANT_D65[1] * ColorConverter._lab_f_inv(fy)
        z = ColorConverter.ILLUMINANT_D65[2] * ColorConverter._lab_f_inv(fz)
        
        # XYZ → RGB (macierz odwrotna sRGB)
        r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
        g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
        b_val = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
        
        # Korekcja gamma odwrotna
        r = ColorConverter._gamma_correct_inv(r)
        g = ColorConverter._gamma_correct_inv(g)
        b_val = ColorConverter._gamma_correct_inv(b_val)
        
        # Ograniczenie do [0, 1] i konwersja do [0, 255]
        r = max(0, min(1, r)) * 255
        g = max(0, min(1, g)) * 255
        b_val = max(0, min(1, b_val)) * 255
        
        return (int(round(r)), int(round(g)), int(round(b_val)))
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """
        Konwertuje kolor HEX do RGB
        
        Args:
            hex_color: Kolor w formacie #RRGGBB, RRGGBB, #RGB lub RGB
        
        Returns:
            Kolor RGB
        """
        hex_color = hex_color.lstrip('#')
        
        if len(hex_color) == 3:
            # Format RGB → RRGGBB
            hex_color = ''.join([c*2 for c in hex_color])
        elif len(hex_color) != 6:
            raise ValueError(f"Nieprawidłowy format HEX: {hex_color}")
        
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            raise ValueError(f"Nieprawidłowy format HEX: {hex_color}")
    
    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """
        Konwertuje RGB do HEX
        
        Args:
            rgb: Kolor RGB
        
        Returns:
            Kolor w formacie #RRGGBB
        """
        r, g, b = rgb
        if not all(0 <= c <= 255 for c in [r, g, b]):
            raise ValueError(f"RGB musi być w zakresie [0, 255], otrzymano {rgb}")
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def _gamma_correct(value: float) -> float:
        """Korekcja gamma dla sRGB"""
        if value <= 0.04045:
            return value / 12.92
        else:
            return pow((value + 0.055) / 1.055, 2.4)
    
    @staticmethod
    def _gamma_correct_inv(value: float) -> float:
        """Odwrotna korekcja gamma dla sRGB"""
        if value <= 0.0031308:
            return 12.92 * value
        else:
            return 1.055 * pow(value, 1/2.4) - 0.055
    
    @staticmethod
    def _lab_f(t: float) -> float:
        """Funkcja f dla konwersji XYZ → LAB"""
        if t > (6/29)**3:
            return pow(t, 1/3)
        else:
            return (1/3) * (29/6)**2 * t + 4/29
    
    @staticmethod
    def _lab_f_inv(t: float) -> float:
        """Odwrotna funkcja f dla konwersji LAB → XYZ"""
        if t > 6/29:
            return t**3
        else:
            return 3 * (6/29)**2 * (t - 4/29)
```

---

## Przykłady Użycia

### Podstawowe Operacje

```python
# Inicjalizacja
calculator = DeltaECalculator(DeltaEMethod.CIEDE2000)
converter = ColorConverter()

# Kolory RGB
color1_rgb = (255, 0, 0)    # Czerwony
color2_rgb = (255, 50, 50)  # Jasno-czerwony

# Konwersja do LAB
color1_lab = converter.rgb_to_lab(color1_rgb)
color2_lab = converter.rgb_to_lab(color2_rgb)

print(f"Kolor 1: RGB{color1_rgb} → LAB{color1_lab}")
print(f"Kolor 2: RGB{color2_rgb} → LAB{color2_lab}")

# Obliczenie Delta E
result = calculator.calculate(color1_lab, color2_lab, return_details=True)
print(f"\n{result}")

# Konwersja z powrotem do RGB
back_to_rgb1 = converter.lab_to_rgb(color1_lab)
back_to_rgb2 = converter.lab_to_rgb(color2_lab)
print(f"\nSprawdzenie konwersji:")
print(f"RGB1: {color1_rgb} → LAB → RGB: {back_to_rgb1}")
print(f"RGB2: {color2_rgb} → LAB → RGB: {back_to_rgb2}")
```

### Porównanie Metod

```python
def compare_methods(color1_lab, color2_lab):
    """Porównuje różne metody Delta E"""
    methods = [DeltaEMethod.CIE76, DeltaEMethod.CIE94, 
               DeltaEMethod.CIEDE2000, DeltaEMethod.CMC]
    
    print("Porównanie metod Delta E:")
    print("-" * 40)
    
    results = {}
    for method in methods:
        calc = DeltaECalculator(method)
        delta_e = calc.calculate(color1_lab, color2_lab)
        interpretation = calc._interpret_delta_e(delta_e)
        results[method] = delta_e
        
        print(f"{method.value.upper():>10}: {delta_e:6.3f} - {interpretation}")
    
    return results

# Przykład użycia
color1 = (50, 20, -10)  # Szaro-różowy
color2 = (55, 25, -5)   # Jaśniejszy szaro-różowy

results = compare_methods(color1, color2)
```

### Konwersje z HEX

```python
def analyze_hex_colors(hex1: str, hex2: str):
    """Analizuje różnicę między kolorami HEX"""
    converter = ColorConverter()
    calculator = DeltaECalculator(DeltaEMethod.CIEDE2000)
    
    # Konwersje
    rgb1 = converter.hex_to_rgb(hex1)
    rgb2 = converter.hex_to_rgb(hex2)
    
    lab1 = converter.rgb_to_lab(rgb1)
    lab2 = converter.rgb_to_lab(rgb2)
    
    # Analiza
    result = calculator.calculate(lab1, lab2, return_details=True)
    
    print(f"Analiza kolorów:")
    print(f"Kolor 1: {hex1} → RGB{rgb1} → LAB({lab1[0]:.1f}, {lab1[1]:.1f}, {lab1[2]:.1f})")
    print(f"Kolor 2: {hex2} → RGB{rgb2} → LAB({lab2[0]:.1f}, {lab2[1]:.1f}, {lab2[2]:.1f})")
    print(f"\n{result}")
    
    return result

# Przykłady
print("=== Analiza kolorów czerwonych ===")
analyze_hex_colors("#FF0000", "#FF3333")

print("\n=== Analiza kolorów niebieskich ===")
analyze_hex_colors("#0000FF", "#3333FF")

print("\n=== Analiza kolorów zielonych ===")
analyze_hex_colors("#00FF00", "#33FF33")
```

---

## Obsługa Błędów i Walidacja

### Klasa Wyjątków

```python
class DeltaEError(Exception):
    """Bazowa klasa wyjątków dla Delta E"""
    pass

class InvalidColorError(DeltaEError):
    """Błąd nieprawidłowego koloru"""
    pass

class ConversionError(DeltaEError):
    """Błąd konwersji kolorów"""
    pass

class MethodError(DeltaEError):
    """Błąd metody obliczania"""
    pass
```

### Rozszerzona Walidacja

```python
class ValidatedDeltaECalculator(DeltaECalculator):
    """Kalkulator Delta E z rozszerzoną walidacją"""
    
    def calculate(self, color1: Tuple[float, float, float], 
                 color2: Tuple[float, float, float],
                 return_details: bool = False) -> Union[float, DeltaEResult]:
        """
        Oblicza Delta E z rozszerzoną walidacją
        """
        try:
            # Walidacja podstawowa
            self._validate_lab_color(color1, "color1")
            self._validate_lab_color(color2, "color2")
            
            # Sprawdzenie czy kolory nie są identyczne
            if color1 == color2:
                if return_details:
                    return DeltaEResult(
                        value=0.0,
                        method=self.method,
                        color1_lab=color1,
                        color2_lab=color2,
                        interpretation="Identyczne kolory"
                    )
                return 0.0
            
            # Obliczenie Delta E
            result = super().calculate(color1, color2, return_details)
            
            # Sprawdzenie czy wynik jest prawidłowy
            delta_e_value = result.value if return_details else result
            if delta_e_value < 0 or math.isnan(delta_e_value) or math.isinf(delta_e_value):
                raise MethodError(f"Nieprawidłowy wynik Delta E: {delta_e_value}")
            
            return result
            
        except Exception as e:
            if isinstance(e, DeltaEError):
                raise
            else:
                raise MethodError(f"Błąd podczas obliczania Delta E: {str(e)}")
    
    def _validate_lab_color(self, color: Tuple[float, float, float], name: str):
        """Rozszerzona walidacja koloru LAB"""
        try:
            super()._validate_lab_color(color, name)
        except ValueError as e:
            raise InvalidColorError(str(e))
        
        # Dodatkowe sprawdzenia
        L, a, b = color
        
        # Sprawdzenie czy wartości nie są NaN lub inf
        if any(math.isnan(x) or math.isinf(x) for x in [L, a, b]):
            raise InvalidColorError(f"{name}: Wartości nie mogą być NaN lub inf")
        
        # Ostrzeżenie dla nietypowych wartości
        if L > 95:  # Bardzo jasne
            print(f"Ostrzeżenie: {name} ma bardzo wysoką jasność L*={L}")
        elif L < 5:  # Bardzo ciemne
            print(f"Ostrzeżenie: {name} ma bardzo niską jasność L*={L}")
        
        if abs(a) > 100 or abs(b) > 100:  # Bardzo nasycone
            print(f"Ostrzeżenie: {name} ma bardzo wysokie nasycenie a*={a}, b*={b}")
```

### Przykład Obsługi Błędów

```python
def safe_delta_e_calculation(color1, color2):
    """Bezpieczne obliczanie Delta E z obsługą błędów"""
    calculator = ValidatedDeltaECalculator(DeltaEMethod.CIEDE2000)
    
    try:
        result = calculator.calculate(color1, color2, return_details=True)
        print(f"Sukces: {result}")
        return result
        
    except InvalidColorError as e:
        print(f"Błąd koloru: {e}")
        return None
        
    except MethodError as e:
        print(f"Błąd metody: {e}")
        return None
        
    except DeltaEError as e:
        print(f"Błąd Delta E: {e}")
        return None
        
    except Exception as e:
        print(f"Nieoczekiwany błąd: {e}")
        return None

# Przykłady testów
print("=== Test prawidłowych kolorów ===")
safe_delta_e_calculation((50, 20, -10), (55, 25, -5))

print("\n=== Test nieprawidłowych kolorów ===")
safe_delta_e_calculation((150, 20, -10), (55, 25, -5))  # L* > 100
safe_delta_e_calculation((50, 200, -10), (55, 25, -5))  # a* > 127
safe_delta_e_calculation((50, 20, float('nan')), (55, 25, -5))  # NaN
```

---

## Podsumowanie Części 2

W tej części zaimplementowaliśmy:

1. **Kompletną klasę DeltaECalculator** z obsługą wszystkich głównych metod
2. **Precyzyjne konwersje kolorów** RGB ↔ LAB ↔ HEX
3. **Walidację danych wejściowych** i obsługę błędów
4. **Szczegółowe wyniki** z interpretacją
5. **Przykłady praktyczne** użycia

### Kluczowe Cechy

✅ **Kompletność**: Wszystkie główne metody Delta E  
✅ **Dokładność**: Precyzyjne implementacje formuł  
✅ **Niezawodność**: Walidacja i obsługa błędów  
✅ **Elastyczność**: Konfigurowalne parametry  
✅ **Użyteczność**: Czytelne wyniki i interpretacje  

### Co dalej?

**Część 3** będzie zawierać:
- Szczegółową implementację CIE76
- Optymalizacje wydajności
- Testy jednostkowe
- Benchmarki porównawcze

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Część 2 - Podstawowa implementacja