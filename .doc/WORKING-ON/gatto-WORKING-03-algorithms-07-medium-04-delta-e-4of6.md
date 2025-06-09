# Delta E Color Distance - Część 4: CIE94 - Zaawansowana Implementacja

## 🟡 Poziom: Medium
**Trudność**: Średnia-Wysoka | **Czas implementacji**: 3-4 godziny | **Złożoność**: O(1)

---

## Wprowadzenie do CIE94

### Historia i Rozwój

Delta E CIE94 zostało wprowadzone w 1994 roku jako ulepszenie CIE76, aby lepiej odzwierciedlać percepcję ludzkiego oka. Główne innowacje:

- **Funkcje wagowe** dla różnych składowych
- **Parametry aplikacyjne** (grafika vs tekstylia)
- **Uwzględnienie chromatyczności** w obliczeniach
- **Lepsza korelacja** z oceną wizualną

### Kluczowe Różnice względem CIE76

| Aspekt | CIE76 | CIE94 |
|--------|-------|-------|
| Formuła | Prosta odległość euklidesowa | Ważona odległość z funkcjami korekcyjnymi |
| Chromatyczność | Ignorowana | Uwzględniana przez C* i H* |
| Parametry aplikacyjne | Brak | Grafika (kG=1) vs Tekstylia (kG=2) |
| Złożoność obliczeniowa | O(1) - bardzo prosta | O(1) - umiarkowana |
| Dokładność percepcyjna | Niska | Średnia-wysoka |

---

## Matematyczne Podstawy CIE94

### Formuła Główna

```
ΔE*94 = √[(ΔL*/kL·SL)² + (ΔC*/kC·SC)² + (ΔH*/kH·SH)²]
```

### Składowe i Parametry

#### 1. Różnice Podstawowe
```
ΔL* = L₂* - L₁*                    (różnica jasności)
Δa* = a₂* - a₁*                    (różnica a*)
Δb* = b₂* - b₁*                    (różnica b*)
```

#### 2. Chromatyczność i Odcień
```
C₁* = √(a₁*² + b₁*²)               (chromatyczność koloru 1)
C₂* = √(a₂*² + b₂*²)               (chromatyczność koloru 2)
ΔC* = C₂* - C₁*                    (różnica chromatyczności)

ΔH* = √(Δa*² + Δb*² - ΔC*²)        (różnica odcienia)
```

#### 3. Funkcje Wagowe
```
SL = 1                             (waga jasności - stała)
SC = 1 + K₁·C₁*                    (waga chromatyczności)
SH = 1 + K₂·C₁*                    (waga odcienia)
```

#### 4. Parametry Aplikacyjne

**Grafika (Graphic Arts)**:
```
kL = kC = kH = 1
K₁ = 0.045
K₂ = 0.015
```

**Tekstylia (Textiles)**:
```
kL = 2, kC = kH = 1
K₁ = 0.048
K₂ = 0.014
```

---

## Implementacja CIE94

### Enumeracje i Struktury

```python
import math
import numpy as np
from enum import Enum
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

class CIE94Application(Enum):
    """Typy aplikacji dla CIE94"""
    GRAPHIC_ARTS = "graphic_arts"
    TEXTILES = "textiles"
    CUSTOM = "custom"

@dataclass
class CIE94Parameters:
    """Parametry dla obliczenia CIE94"""
    kL: float  # Waga jasności
    kC: float  # Waga chromatyczności
    kH: float  # Waga odcienia
    K1: float  # Parametr funkcji SC
    K2: float  # Parametr funkcji SH
    application: CIE94Application
    
    @classmethod
    def graphic_arts(cls) -> 'CIE94Parameters':
        """Parametry dla grafiki"""
        return cls(
            kL=1.0, kC=1.0, kH=1.0,
            K1=0.045, K2=0.015,
            application=CIE94Application.GRAPHIC_ARTS
        )
    
    @classmethod
    def textiles(cls) -> 'CIE94Parameters':
        """Parametry dla tekstyliów"""
        return cls(
            kL=2.0, kC=1.0, kH=1.0,
            K1=0.048, K2=0.014,
            application=CIE94Application.TEXTILES
        )
    
    @classmethod
    def custom(cls, kL: float, kC: float, kH: float, 
              K1: float, K2: float) -> 'CIE94Parameters':
        """Parametry niestandardowe"""
        return cls(
            kL=kL, kC=kC, kH=kH,
            K1=K1, K2=K2,
            application=CIE94Application.CUSTOM
        )

@dataclass
class CIE94Result:
    """Wynik obliczenia Delta E CIE94"""
    delta_e: float
    delta_l: float
    delta_c: float
    delta_h: float
    c1_star: float
    c2_star: float
    sl: float
    sc: float
    sh: float
    color1_lab: Tuple[float, float, float]
    color2_lab: Tuple[float, float, float]
    parameters: CIE94Parameters
    interpretation: str
    
    def __str__(self):
        return f"ΔE*94: {self.delta_e:.3f} ({self.interpretation})"
    
    def detailed_breakdown(self) -> str:
        """Zwraca szczegółowy rozkład różnic"""
        return f"""Delta E CIE94 Breakdown ({self.parameters.application.value}):
├─ ΔE*94: {self.delta_e:.3f}
├─ ΔL*: {self.delta_l:.3f} (jasność, SL={self.sl:.3f})
├─ ΔC*: {self.delta_c:.3f} (chromatyczność, SC={self.sc:.3f})
├─ ΔH*: {self.delta_h:.3f} (odcień, SH={self.sh:.3f})
├─ C₁*: {self.c1_star:.3f}, C₂*: {self.c2_star:.3f}
└─ Interpretacja: {self.interpretation}"""
    
    def component_contributions(self) -> dict:
        """Oblicza wkład poszczególnych składowych"""
        # Składowe ważone
        l_weighted = (self.delta_l / (self.parameters.kL * self.sl))**2
        c_weighted = (self.delta_c / (self.parameters.kC * self.sc))**2
        h_weighted = (self.delta_h / (self.parameters.kH * self.sh))**2
        
        total = l_weighted + c_weighted + h_weighted
        
        if total > 0:
            return {
                'L*': (l_weighted / total) * 100,
                'C*': (c_weighted / total) * 100,
                'H*': (h_weighted / total) * 100
            }
        else:
            return {'L*': 0, 'C*': 0, 'H*': 0}
```

### Główna Klasa Kalkulatora

```python
class CIE94Calculator:
    """Kalkulator Delta E CIE94"""
    
    def __init__(self, application: CIE94Application = CIE94Application.GRAPHIC_ARTS,
                 custom_params: Optional[CIE94Parameters] = None):
        """
        Inicjalizuje kalkulator CIE94
        
        Args:
            application: Typ aplikacji (grafika/tekstylia)
            custom_params: Niestandardowe parametry (opcjonalne)
        """
        if custom_params is not None:
            self.parameters = custom_params
        elif application == CIE94Application.GRAPHIC_ARTS:
            self.parameters = CIE94Parameters.graphic_arts()
        elif application == CIE94Application.TEXTILES:
            self.parameters = CIE94Parameters.textiles()
        else:
            raise ValueError(f"Nieobsługiwany typ aplikacji: {application}")
        
        self.method_name = "CIE94"
        self.year_introduced = 1994
    
    def calculate(self, color1: Tuple[float, float, float], 
                 color2: Tuple[float, float, float],
                 return_details: bool = False) -> Union[float, CIE94Result]:
        """
        Oblicza Delta E CIE94 między dwoma kolorami LAB
        
        Args:
            color1: Pierwszy kolor (L*, a*, b*)
            color2: Drugi kolor (L*, a*, b*)
            return_details: Czy zwrócić szczegółowe informacje
        
        Returns:
            Wartość Delta E lub obiekt CIE94Result
        """
        # Walidacja
        self._validate_lab_color(color1, "color1")
        self._validate_lab_color(color2, "color2")
        
        # Rozpakowanie kolorów
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        
        # Podstawowe różnice
        delta_l = L2 - L1
        delta_a = a2 - a1
        delta_b = b2 - b1
        
        # Chromatyczność
        c1_star = math.sqrt(a1**2 + b1**2)
        c2_star = math.sqrt(a2**2 + b2**2)
        delta_c = c2_star - c1_star
        
        # Różnica odcienia (ΔH*)
        delta_h_squared = delta_a**2 + delta_b**2 - delta_c**2
        delta_h = math.sqrt(max(0, delta_h_squared))  # Zabezpieczenie przed ujemną wartością
        
        # Funkcje wagowe
        sl = 1.0  # SL jest zawsze 1 w CIE94
        sc = 1.0 + self.parameters.K1 * c1_star
        sh = 1.0 + self.parameters.K2 * c1_star
        
        # Składowe ważone
        l_component = (delta_l / (self.parameters.kL * sl))**2
        c_component = (delta_c / (self.parameters.kC * sc))**2
        h_component = (delta_h / (self.parameters.kH * sh))**2
        
        # Delta E CIE94
        delta_e = math.sqrt(l_component + c_component + h_component)
        
        if return_details:
            interpretation = self._interpret_delta_e(delta_e)
            return CIE94Result(
                delta_e=delta_e,
                delta_l=delta_l,
                delta_c=delta_c,
                delta_h=delta_h,
                c1_star=c1_star,
                c2_star=c2_star,
                sl=sl,
                sc=sc,
                sh=sh,
                color1_lab=color1,
                color2_lab=color2,
                parameters=self.parameters,
                interpretation=interpretation
            )
        
        return delta_e
    
    def calculate_components(self, color1: Tuple[float, float, float], 
                           color2: Tuple[float, float, float]) -> Tuple[float, float, float, float, float]:
        """
        Oblicza wszystkie składowe CIE94
        
        Returns:
            Tuple (delta_e, delta_l, delta_c, delta_h, c1_star)
        """
        result = self.calculate(color1, color2, return_details=True)
        return (result.delta_e, result.delta_l, result.delta_c, 
                result.delta_h, result.c1_star)
    
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
        
        # Sprawdzenie NaN/inf
        if any(math.isnan(x) or math.isinf(x) for x in [L, a, b]):
            raise ValueError(f"{name}: Wartości nie mogą być NaN lub inf")
    
    def _interpret_delta_e(self, delta_e: float) -> str:
        """Interpretuje wartość Delta E CIE94"""
        if delta_e < 1:
            return "Niewidoczna różnica"
        elif delta_e < 2:
            return "Ledwo widoczna różnica"
        elif delta_e < 3:
            return "Widoczna przy porównaniu"
        elif delta_e < 6:
            return "Wyraźnie widoczna różnica"
        elif delta_e < 12:
            return "Znacząca różnica"
        else:
            return "Bardzo duża różnica"
```

---

## Optymalizacje Wydajności

### Wektoryzowana Implementacja

```python
class OptimizedCIE94Calculator:
    """Zoptymalizowana wersja kalkulatora CIE94"""
    
    def __init__(self, parameters: CIE94Parameters = None):
        if parameters is None:
            self.parameters = CIE94Parameters.graphic_arts()
        else:
            self.parameters = parameters
        
        self.base_calculator = CIE94Calculator(
            custom_params=self.parameters
        )
    
    def calculate_batch(self, colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """
        Oblicza Delta E CIE94 dla batch'a kolorów
        
        Args:
            colors1: Array kolorów LAB (N, 3)
            colors2: Array kolorów LAB (N, 3)
        
        Returns:
            Array wartości Delta E (N,)
        """
        # Walidacja wymiarów
        if colors1.shape != colors2.shape:
            raise ValueError("Arrays muszą mieć te same wymiary")
        
        if colors1.shape[1] != 3:
            raise ValueError("Kolory muszą mieć 3 składowe (L*, a*, b*)")
        
        # Rozpakowanie składowych
        L1, a1, b1 = colors1[:, 0], colors1[:, 1], colors1[:, 2]
        L2, a2, b2 = colors2[:, 0], colors2[:, 1], colors2[:, 2]
        
        # Podstawowe różnice
        delta_l = L2 - L1
        delta_a = a2 - a1
        delta_b = b2 - b1
        
        # Chromatyczność
        c1_star = np.sqrt(a1**2 + b1**2)
        c2_star = np.sqrt(a2**2 + b2**2)
        delta_c = c2_star - c1_star
        
        # Różnica odcienia
        delta_h_squared = delta_a**2 + delta_b**2 - delta_c**2
        delta_h = np.sqrt(np.maximum(0, delta_h_squared))
        
        # Funkcje wagowe
        sl = 1.0
        sc = 1.0 + self.parameters.K1 * c1_star
        sh = 1.0 + self.parameters.K2 * c1_star
        
        # Składowe ważone
        l_component = (delta_l / (self.parameters.kL * sl))**2
        c_component = (delta_c / (self.parameters.kC * sc))**2
        h_component = (delta_h / (self.parameters.kH * sh))**2
        
        # Delta E CIE94
        delta_e_values = np.sqrt(l_component + c_component + h_component)
        
        return delta_e_values
    
    def calculate_detailed_batch(self, colors1: np.ndarray, 
                               colors2: np.ndarray) -> dict:
        """
        Oblicza szczegółowe informacje dla batch'a kolorów
        
        Returns:
            Słownik z wszystkimi składowymi
        """
        # Podstawowe obliczenia (jak w calculate_batch)
        L1, a1, b1 = colors1[:, 0], colors1[:, 1], colors1[:, 2]
        L2, a2, b2 = colors2[:, 0], colors2[:, 1], colors2[:, 2]
        
        delta_l = L2 - L1
        delta_a = a2 - a1
        delta_b = b2 - b1
        
        c1_star = np.sqrt(a1**2 + b1**2)
        c2_star = np.sqrt(a2**2 + b2**2)
        delta_c = c2_star - c1_star
        
        delta_h_squared = delta_a**2 + delta_b**2 - delta_c**2
        delta_h = np.sqrt(np.maximum(0, delta_h_squared))
        
        sl = np.ones_like(c1_star)
        sc = 1.0 + self.parameters.K1 * c1_star
        sh = 1.0 + self.parameters.K2 * c1_star
        
        l_component = (delta_l / (self.parameters.kL * sl))**2
        c_component = (delta_c / (self.parameters.kC * sc))**2
        h_component = (delta_h / (self.parameters.kH * sh))**2
        
        delta_e_values = np.sqrt(l_component + c_component + h_component)
        
        return {
            'delta_e': delta_e_values,
            'delta_l': delta_l,
            'delta_c': delta_c,
            'delta_h': delta_h,
            'c1_star': c1_star,
            'c2_star': c2_star,
            'sl': sl,
            'sc': sc,
            'sh': sh,
            'l_component': l_component,
            'c_component': c_component,
            'h_component': h_component
        }
```

### Implementacja z Numba

```python
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

class NumbaCIE94Calculator:
    """Kalkulator CIE94 z przyspieszeniem Numba"""
    
    def __init__(self, parameters: CIE94Parameters = None):
        if parameters is None:
            self.parameters = CIE94Parameters.graphic_arts()
        else:
            self.parameters = parameters
        
        self.numba_available = NUMBA_AVAILABLE
        if not NUMBA_AVAILABLE:
            print("Ostrzeżenie: Numba niedostępna, używam standardowej implementacji")
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_batch_numba(colors1: np.ndarray, colors2: np.ndarray,
                              kL: float, kC: float, kH: float,
                              K1: float, K2: float) -> np.ndarray:
        """
        Numba-zoptymalizowane obliczenie batch Delta E CIE94
        """
        n = colors1.shape[0]
        results = np.zeros(n)
        
        for i in prange(n):
            # Rozpakowanie kolorów
            L1, a1, b1 = colors1[i, 0], colors1[i, 1], colors1[i, 2]
            L2, a2, b2 = colors2[i, 0], colors2[i, 1], colors2[i, 2]
            
            # Podstawowe różnice
            delta_l = L2 - L1
            delta_a = a2 - a1
            delta_b = b2 - b1
            
            # Chromatyczność
            c1_star = math.sqrt(a1*a1 + b1*b1)
            c2_star = math.sqrt(a2*a2 + b2*b2)
            delta_c = c2_star - c1_star
            
            # Różnica odcienia
            delta_h_squared = delta_a*delta_a + delta_b*delta_b - delta_c*delta_c
            delta_h = math.sqrt(max(0.0, delta_h_squared))
            
            # Funkcje wagowe
            sl = 1.0
            sc = 1.0 + K1 * c1_star
            sh = 1.0 + K2 * c1_star
            
            # Składowe ważone
            l_component = (delta_l / (kL * sl))**2
            c_component = (delta_c / (kC * sc))**2
            h_component = (delta_h / (kH * sh))**2
            
            # Delta E CIE94
            results[i] = math.sqrt(l_component + c_component + h_component)
        
        return results
    
    def calculate_batch(self, colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """
        Oblicza Delta E CIE94 dla batch'a kolorów z przyspieszeniem Numba
        """
        if self.numba_available:
            return self._calculate_batch_numba(
                colors1, colors2,
                self.parameters.kL, self.parameters.kC, self.parameters.kH,
                self.parameters.K1, self.parameters.K2
            )
        else:
            # Fallback do NumPy
            calc = OptimizedCIE94Calculator(self.parameters)
            return calc.calculate_batch(colors1, colors2)
```

---

## Analiza Porównawcza

### Porównanie CIE76 vs CIE94

```python
class CIE76vsCIE94Analyzer:
    """Analizator porównujący CIE76 i CIE94"""
    
    def __init__(self):
        from delta_e_calculator import DeltaECalculator, DeltaEMethod
        
        self.cie76_calc = DeltaECalculator(DeltaEMethod.CIE76)
        self.cie94_graphic_calc = CIE94Calculator(CIE94Application.GRAPHIC_ARTS)
        self.cie94_textile_calc = CIE94Calculator(CIE94Application.TEXTILES)
    
    def compare_methods(self, color1: Tuple[float, float, float], 
                       color2: Tuple[float, float, float]) -> dict:
        """
        Porównuje różne metody Delta E dla pary kolorów
        """
        # Obliczenia
        cie76_result = self.cie76_calc.calculate(color1, color2)
        cie94_graphic = self.cie94_graphic_calc.calculate(color1, color2)
        cie94_textile = self.cie94_textile_calc.calculate(color1, color2)
        
        # Szczegółowe wyniki CIE94
        cie94_graphic_detailed = self.cie94_graphic_calc.calculate(color1, color2, return_details=True)
        cie94_textile_detailed = self.cie94_textile_calc.calculate(color1, color2, return_details=True)
        
        return {
            'colors': {
                'color1': color1,
                'color2': color2
            },
            'results': {
                'CIE76': cie76_result,
                'CIE94_graphic': cie94_graphic,
                'CIE94_textile': cie94_textile
            },
            'differences': {
                'CIE94_graphic_vs_CIE76': abs(cie94_graphic - cie76_result),
                'CIE94_textile_vs_CIE76': abs(cie94_textile - cie76_result),
                'CIE94_textile_vs_graphic': abs(cie94_textile - cie94_graphic)
            },
            'detailed_cie94': {
                'graphic': cie94_graphic_detailed,
                'textile': cie94_textile_detailed
            },
            'analysis': self._analyze_differences(cie76_result, cie94_graphic, cie94_textile)
        }
    
    def _analyze_differences(self, cie76: float, cie94_g: float, cie94_t: float) -> dict:
        """
        Analizuje różnice między metodami
        """
        # Która metoda daje największą/najmniejszą wartość
        values = {'CIE76': cie76, 'CIE94_graphic': cie94_g, 'CIE94_textile': cie94_t}
        
        min_method = min(values, key=values.get)
        max_method = max(values, key=values.get)
        
        # Względne różnice
        relative_diff_g = ((cie94_g - cie76) / cie76 * 100) if cie76 > 0 else 0
        relative_diff_t = ((cie94_t - cie76) / cie76 * 100) if cie76 > 0 else 0
        
        return {
            'min_method': min_method,
            'max_method': max_method,
            'min_value': values[min_method],
            'max_value': values[max_method],
            'range': values[max_method] - values[min_method],
            'relative_differences': {
                'CIE94_graphic_vs_CIE76_percent': relative_diff_g,
                'CIE94_textile_vs_CIE76_percent': relative_diff_t
            },
            'interpretation': self._interpret_method_differences(relative_diff_g, relative_diff_t)
        }
    
    def _interpret_method_differences(self, rel_diff_g: float, rel_diff_t: float) -> str:
        """
        Interpretuje różnice między metodami
        """
        if abs(rel_diff_g) < 5 and abs(rel_diff_t) < 5:
            return "Metody dają podobne wyniki"
        elif abs(rel_diff_g) > 20 or abs(rel_diff_t) > 20:
            return "Znaczące różnice między metodami"
        else:
            return "Umiarkowane różnice między metodami"
    
    def batch_comparison(self, colors1: np.ndarray, colors2: np.ndarray) -> dict:
        """
        Porównuje metody dla batch'a kolorów
        """
        # Obliczenia batch
        cie76_results = []
        for i in range(len(colors1)):
            result = self.cie76_calc.calculate(tuple(colors1[i]), tuple(colors2[i]))
            cie76_results.append(result)
        cie76_results = np.array(cie76_results)
        
        cie94_graphic_calc = OptimizedCIE94Calculator(CIE94Parameters.graphic_arts())
        cie94_textile_calc = OptimizedCIE94Calculator(CIE94Parameters.textiles())
        
        cie94_graphic_results = cie94_graphic_calc.calculate_batch(colors1, colors2)
        cie94_textile_results = cie94_textile_calc.calculate_batch(colors1, colors2)
        
        # Statystyki
        return {
            'count': len(colors1),
            'statistics': {
                'CIE76': {
                    'mean': np.mean(cie76_results),
                    'std': np.std(cie76_results),
                    'min': np.min(cie76_results),
                    'max': np.max(cie76_results)
                },
                'CIE94_graphic': {
                    'mean': np.mean(cie94_graphic_results),
                    'std': np.std(cie94_graphic_results),
                    'min': np.min(cie94_graphic_results),
                    'max': np.max(cie94_graphic_results)
                },
                'CIE94_textile': {
                    'mean': np.mean(cie94_textile_results),
                    'std': np.std(cie94_textile_results),
                    'min': np.min(cie94_textile_results),
                    'max': np.max(cie94_textile_results)
                }
            },
            'correlations': {
                'CIE76_vs_CIE94_graphic': np.corrcoef(cie76_results, cie94_graphic_results)[0, 1],
                'CIE76_vs_CIE94_textile': np.corrcoef(cie76_results, cie94_textile_results)[0, 1],
                'CIE94_graphic_vs_textile': np.corrcoef(cie94_graphic_results, cie94_textile_results)[0, 1]
            },
            'mean_absolute_differences': {
                'CIE94_graphic_vs_CIE76': np.mean(np.abs(cie94_graphic_results - cie76_results)),
                'CIE94_textile_vs_CIE76': np.mean(np.abs(cie94_textile_results - cie76_results)),
                'CIE94_textile_vs_graphic': np.mean(np.abs(cie94_textile_results - cie94_graphic_results))
            }
        }
```

---

## Przykłady Praktyczne

### Demonstracja Podstawowa

```python
def demonstrate_cie94_basic():
    """Demonstracja podstawowego użycia CIE94"""
    print("=== Demonstracja CIE94 ===")
    
    # Kolory testowe
    red = (53.24, 80.09, 67.20)      # Czerwony
    orange = (74.93, 23.93, 78.95)   # Pomarańczowy
    
    print(f"Kolor 1 (czerwony): LAB{red}")
    print(f"Kolor 2 (pomarańczowy): LAB{orange}")
    print("-" * 50)
    
    # Porównanie aplikacji
    graphic_calc = CIE94Calculator(CIE94Application.GRAPHIC_ARTS)
    textile_calc = CIE94Calculator(CIE94Application.TEXTILES)
    
    # Obliczenia
    graphic_result = graphic_calc.calculate(red, orange, return_details=True)
    textile_result = textile_calc.calculate(red, orange, return_details=True)
    
    print("GRAFIKA (Graphic Arts):")
    print(graphic_result.detailed_breakdown())
    print(f"\nWkład składowych:")
    graphic_contrib = graphic_result.component_contributions()
    for component, value in graphic_contrib.items():
        print(f"  {component}: {value:.1f}%")
    
    print("\n" + "="*50)
    print("TEKSTYLIA (Textiles):")
    print(textile_result.detailed_breakdown())
    print(f"\nWkład składowych:")
    textile_contrib = textile_result.component_contributions()
    for component, value in textile_contrib.items():
        print(f"  {component}: {value:.1f}%")
    
    # Porównanie
    diff = abs(graphic_result.delta_e - textile_result.delta_e)
    print(f"\n=== PORÓWNANIE ===")
    print(f"Różnica Grafika-Tekstylia: {diff:.3f}")
    print(f"Względna różnica: {(diff/graphic_result.delta_e)*100:.1f}%")

# demonstrate_cie94_basic()
```

### Analiza Wrażliwości

```python
def analyze_cie94_sensitivity():
    """Analizuje wrażliwość CIE94 na różne typy zmian kolorów"""
    print("=== Analiza Wrażliwości CIE94 ===")
    
    analyzer = CIE76vsCIE94Analyzer()
    
    # Test cases - różne typy zmian
    test_cases = [
        {
            'name': 'Zmiana jasności',
            'color1': (50, 0, 0),
            'color2': (55, 0, 0)
        },
        {
            'name': 'Zmiana chromatyczności (a*)',
            'color1': (50, 20, 0),
            'color2': (50, 25, 0)
        },
        {
            'name': 'Zmiana chromatyczności (b*)',
            'color1': (50, 0, 20),
            'color2': (50, 0, 25)
        },
        {
            'name': 'Zmiana odcienia',
            'color1': (50, 20, 20),
            'color2': (50, 20, 25)
        },
        {
            'name': 'Kolory o wysokiej chromatyczności',
            'color1': (50, 60, 60),
            'color2': (55, 65, 65)
        },
        {
            'name': 'Kolory o niskiej chromatyczności',
            'color1': (50, 5, 5),
            'color2': (55, 10, 10)
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        comparison = analyzer.compare_methods(test_case['color1'], test_case['color2'])
        
        print(f"CIE76:           {comparison['results']['CIE76']:.3f}")
        print(f"CIE94 (grafika): {comparison['results']['CIE94_graphic']:.3f}")
        print(f"CIE94 (tekstyl): {comparison['results']['CIE94_textile']:.3f}")
        
        print(f"Interpretacja: {comparison['analysis']['interpretation']}")
        
        # Analiza funkcji wagowych dla CIE94 grafika
        cie94_detail = comparison['detailed_cie94']['graphic']
        print(f"Funkcje wagowe: SL={cie94_detail.sl:.3f}, SC={cie94_detail.sc:.3f}, SH={cie94_detail.sh:.3f}")

# analyze_cie94_sensitivity()
```

### Benchmark Wydajności

```python
import time

def benchmark_cie94_performance():
    """Benchmark wydajności różnych implementacji CIE94"""
    # Generowanie danych testowych
    np.random.seed(42)
    n_colors = 5000
    colors1 = np.random.rand(n_colors, 3) * [100, 255, 255] - [0, 128, 128]
    colors2 = np.random.rand(n_colors, 3) * [100, 255, 255] - [0, 128, 128]
    
    print(f"=== Benchmark CIE94 ({n_colors} par kolorów) ===")
    
    # 1. Podstawowa implementacja (pętla)
    basic_calc = CIE94Calculator(CIE94Application.GRAPHIC_ARTS)
    start_time = time.time()
    
    basic_results = []
    for i in range(n_colors):
        result = basic_calc.calculate(tuple(colors1[i]), tuple(colors2[i]))
        basic_results.append(result)
    
    basic_time = time.time() - start_time
    print(f"Podstawowa (pętla):     {basic_time:.3f}s")
    
    # 2. Zoptymalizowana NumPy
    optimized_calc = OptimizedCIE94Calculator(CIE94Parameters.graphic_arts())
    start_time = time.time()
    
    optimized_results = optimized_calc.calculate_batch(colors1, colors2)
    
    optimized_time = time.time() - start_time
    print(f"Zoptymalizowana NumPy:  {optimized_time:.3f}s")
    
    # 3. Numba (jeśli dostępna)
    if NUMBA_AVAILABLE:
        numba_calc = NumbaCIE94Calculator(CIE94Parameters.graphic_arts())
        
        # Pierwsze uruchomienie (kompilacja)
        _ = numba_calc.calculate_batch(colors1[:100], colors2[:100])
        
        start_time = time.time()
        numba_results = numba_calc.calculate_batch(colors1, colors2)
        numba_time = time.time() - start_time
        
        print(f"Numba:                  {numba_time:.3f}s")
        
        # Przyspieszenie
        print(f"\n=== Przyspieszenie ===")
        print(f"NumPy vs podstawowa:    {basic_time/optimized_time:.1f}x")
        print(f"Numba vs podstawowa:    {basic_time/numba_time:.1f}x")
        print(f"Numba vs NumPy:         {optimized_time/numba_time:.1f}x")
    else:
        print(f"\n=== Przyspieszenie ===")
        print(f"NumPy vs podstawowa:    {basic_time/optimized_time:.1f}x")
    
    # Weryfikacja zgodności wyników
    basic_array = np.array(basic_results)
    diff_basic_optimized = np.max(np.abs(basic_array - optimized_results))
    print(f"\n=== Zgodność wyników ===")
    print(f"Max różnica podstawowa-NumPy: {diff_basic_optimized:.10f}")
    
    if NUMBA_AVAILABLE:
        diff_optimized_numba = np.max(np.abs(optimized_results - numba_results))
        print(f"Max różnica NumPy-Numba:      {diff_optimized_numba:.10f}")

# benchmark_cie94_performance()
```

---

## Zastosowania Praktyczne

### Wybór Parametrów Aplikacyjnych

```python
def application_parameter_guide():
    """Przewodnik wyboru parametrów aplikacyjnych"""
    print("=== Przewodnik Parametrów CIE94 ===")
    
    applications = {
        "Grafika (Graphic Arts)": {
            "parameters": CIE94Parameters.graphic_arts(),
            "use_cases": [
                "Druk offsetowy",
                "Grafika cyfrowa",
                "Kontrola jakości druku",
                "Dopasowywanie kolorów na monitorze",
                "Prepress workflow"
            ],
            "characteristics": [
                "kL = kC = kH = 1 (równe wagi)",
                "K1 = 0.045, K2 = 0.015",
                "Zbalansowane podejście do wszystkich składowych"
            ]
        },
        "Tekstylia (Textiles)": {
            "parameters": CIE94Parameters.textiles(),
            "use_cases": [
                "Przemysł tekstylny",
                "Dopasowywanie barwników",
                "Kontrola jakości tkanin",
                "Ocena trwałości kolorów",
                "Standardy przemysłowe"
            ],
            "characteristics": [
                "kL = 2 (podwójna waga jasności)",
                "kC = kH = 1",
                "K1 = 0.048, K2 = 0.014",
                "Większa tolerancja na zmiany jasności"
            ]
        }
    }
    
    for app_name, app_info in applications.items():
        print(f"\n=== {app_name} ===")
        
        params = app_info["parameters"]
        print(f"Parametry: kL={params.kL}, kC={params.kC}, kH={params.kH}")
        print(f"Stałe: K1={params.K1}, K2={params.K2}")
        
        print("\nCharakterystyka:")
        for char in app_info["characteristics"]:
            print(f"  • {char}")
        
        print("\nZastosowania:")
        for use_case in app_info["use_cases"]:
            print(f"  • {use_case}")
    
    # Przykład niestandardowych parametrów
    print("\n=== Parametry Niestandardowe ===")
    print("Możesz utworzyć własne parametry dla specjalnych zastosowań:")
    print("")
    print("# Przykład: Większa wrażliwość na odcień")
    print("custom_params = CIE94Parameters.custom(")
    print("    kL=1.0, kC=1.0, kH=0.5,  # Mniejsza waga odcienia")
    print("    K1=0.045, K2=0.015")
    print(")")
    print("")
    print("# Przykład: Większa tolerancja na jasność")
    print("custom_params = CIE94Parameters.custom(")
    print("    kL=3.0, kC=1.0, kH=1.0,  # Większa waga jasności")
    print("    K1=0.050, K2=0.020")
    print(")")

# application_parameter_guide()
```

---

## Podsumowanie Części 4

W tej części szczegółowo omówiliśmy:

1. **Matematyczne podstawy CIE94** - funkcje wagowe i parametry aplikacyjne
2. **Implementację kompletną** - klasa CIE94Calculator z obsługą różnych aplikacji
3. **Optymalizacje wydajności** - NumPy i Numba dla batch processing
4. **Analizę porównawczą** - CIE76 vs CIE94
5. **Zastosowania praktyczne** - wybór parametrów dla różnych branż

### Kluczowe Cechy CIE94

✅ **Lepsza percepcja**: Uwzględnia chromatyczność i odcień  
✅ **Parametry aplikacyjne**: Grafika vs tekstylia  
✅ **Funkcje wagowe**: SC i SH zależne od chromatyczności  
✅ **Kompatybilność**: Wsteczna zgodność z CIE76  
❌ **Złożoność**: Bardziej skomplikowane obliczenia  
❌ **Ograniczenia**: Wciąż nie idealna dla wszystkich przypadków  

### Kiedy Używać CIE94

- **Grafika i druk** - parametry graphic arts
- **Przemysł tekstylny** - parametry textiles
- **Gdy CIE76 jest niewystarczające** - lepsza percepcja
- **Aplikacje wymagające** średniej dokładności

### Co dalej?

**Część 5** będzie zawierać:
- Szczegółową implementację CIEDE2000
- Najbardziej zaawansowaną formułę Delta E
- Kompleksowe funkcje korekcyjne
- Porównanie wszystkich metod

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Część 4 - CIE94 zaawansowana implementacja