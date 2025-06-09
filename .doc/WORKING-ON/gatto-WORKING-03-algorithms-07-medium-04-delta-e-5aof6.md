# Delta E Color Distance - Część 5a: CIEDE2000 - Implementacja Podstawowa

## 🔴 Poziom: Advanced
**Trudność**: Wysoka | **Czas implementacji**: 4-6 godzin | **Złożoność**: O(1) - skomplikowana

---

## Wprowadzenie do CIEDE2000

### Historia i Znaczenie

CIEDE2000 (Delta E 2000) to najnowsza i najbardziej zaawansowana formuła do obliczania różnic kolorów, wprowadzona przez CIE w 2001 roku. Jest to kulminacja dziesięcioleci badań nad percepcją kolorów przez człowieka.

### Kluczowe Innowacje CIEDE2000

- **Funkcje korekcyjne** dla wszystkich składowych (L*, C*, H*)
- **Interakcja między składowymi** - uwzględnienie wzajemnych wpływów
- **Korekcja obrotu** (RT) dla niebieskich kolorów
- **Najlepsza korelacja** z oceną wizualną człowieka
- **Kompleksowe funkcje wagowe** dostosowane do percepcji

### Porównanie z Poprzednimi Metodami

| Aspekt | CIE76 | CIE94 | CIEDE2000 |
|--------|-------|-------|----------|
| Rok wprowadzenia | 1976 | 1994 | 2001 |
| Złożoność formuły | Bardzo prosta | Średnia | Bardzo wysoka |
| Funkcje wagowe | Brak | Podstawowe | Zaawansowane |
| Korekcja obrotu | Brak | Brak | Tak (RT) |
| Interakcja składowych | Brak | Ograniczona | Pełna |
| Dokładność percepcyjna | Niska | Średnia | Najwyższa |
| Czas obliczeń | Najszybszy | Szybki | Najwolniejszy |

---

## Matematyczne Podstawy CIEDE2000

### Formuła Główna

```
ΔE₀₀ = √[(ΔL'/kL·SL)² + (ΔC'/kC·SC)² + (ΔH'/kH·SH)² + RT·(ΔC'/kC·SC)·(ΔH'/kH·SH)]
```

### Kluczowe Składowe

#### 1. Przekształcenia Wstępne

**Średnia jasność:**
```
L̄ = (L₁* + L₂*) / 2
```

**Korekcja a* (uwzględnienie chromatyczności):**
```
C̄* = (C₁* + C₂*) / 2
G = 0.5 × (1 - √(C̄*⁷ / (C̄*⁷ + 25⁷)))
a₁' = (1 + G) × a₁*
a₂' = (1 + G) × a₂*
```

**Nowe wartości chromatyczności i odcienia:**
```
C₁' = √(a₁'² + b₁*²)
C₂' = √(a₂'² + b₂*²)
h₁' = atan2(b₁*, a₁') × 180/π
h₂' = atan2(b₂*, a₂') × 180/π
```

#### 2. Różnice Podstawowe

```
ΔL' = L₂* - L₁*
ΔC' = C₂' - C₁'
Δh' = h₂' - h₁' (z uwzględnieniem cykliczności)
ΔH' = 2 × √(C₁' × C₂') × sin(Δh'/2 × π/180)
```

#### 3. Średnie Wartości

```
L̄' = (L₁* + L₂*) / 2
C̄' = (C₁' + C₂') / 2
h̄' = (h₁' + h₂') / 2 (z uwzględnieniem cykliczności)
```

#### 4. Funkcje Wagowe

**SL (waga jasności):**
```
SL = 1 + (0.015 × (L̄' - 50)²) / √(20 + (L̄' - 50)²)
```

**SC (waga chromatyczności):**
```
SC = 1 + 0.045 × C̄'
```

**SH (waga odcienia):**
```
T = 1 - 0.17×cos(h̄'-30°) + 0.24×cos(2×h̄') + 0.32×cos(3×h̄'+6°) - 0.20×cos(4×h̄'-63°)
SH = 1 + 0.015 × C̄' × T
```

#### 5. Funkcja Korekcji Obrotu (RT)

```
Δθ = 30 × exp(-((h̄' - 275)/25)²)
RC = 2 × √(C̄'⁷ / (C̄'⁷ + 25⁷))
RT = -sin(2 × Δθ × π/180) × RC
```

---

## Implementacja CIEDE2000

### Struktury Danych

```python
import math
import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

@dataclass
class CIEDE2000Parameters:
    """Parametry dla obliczenia CIEDE2000"""
    kL: float = 1.0  # Waga jasności
    kC: float = 1.0  # Waga chromatyczności
    kH: float = 1.0  # Waga odcienia
    
    @classmethod
    def standard(cls) -> 'CIEDE2000Parameters':
        """Standardowe parametry CIEDE2000"""
        return cls(kL=1.0, kC=1.0, kH=1.0)
    
    @classmethod
    def custom(cls, kL: float, kC: float, kH: float) -> 'CIEDE2000Parameters':
        """Niestandardowe parametry"""
        return cls(kL=kL, kC=kC, kH=kH)

@dataclass
class CIEDE2000IntermediateValues:
    """Wartości pośrednie obliczeń CIEDE2000"""
    # Wartości wejściowe
    L1: float
    a1: float
    b1: float
    L2: float
    a2: float
    b2: float
    
    # Średnie wartości
    L_bar: float
    C_bar_initial: float
    
    # Korekcja G
    G: float
    
    # Skorygowane wartości a'
    a1_prime: float
    a2_prime: float
    
    # Nowe chromatyczności i odcienie
    C1_prime: float
    C2_prime: float
    h1_prime: float
    h2_prime: float
    
    # Różnice
    delta_L_prime: float
    delta_C_prime: float
    delta_h_prime: float
    delta_H_prime: float
    
    # Średnie skorygowane
    L_bar_prime: float
    C_bar_prime: float
    h_bar_prime: float
    
    # Funkcje wagowe
    SL: float
    SC: float
    SH: float
    T: float
    
    # Korekcja obrotu
    delta_theta: float
    RC: float
    RT: float
    
    # Składowe finalne
    L_component: float
    C_component: float
    H_component: float
    interaction_term: float
    
    # Wynik końcowy
    delta_E: float

@dataclass
class CIEDE2000Result:
    """Wynik obliczenia Delta E CIEDE2000"""
    delta_e: float
    color1_lab: Tuple[float, float, float]
    color2_lab: Tuple[float, float, float]
    parameters: CIEDE2000Parameters
    intermediate_values: Optional[CIEDE2000IntermediateValues]
    interpretation: str
    
    def __str__(self):
        return f"ΔE₀₀: {self.delta_e:.3f} ({self.interpretation})"
    
    def detailed_breakdown(self) -> str:
        """Zwraca szczegółowy rozkład różnic"""
        if self.intermediate_values is None:
            return f"ΔE₀₀: {self.delta_e:.3f} - {self.interpretation}"
        
        iv = self.intermediate_values
        return f"""CIEDE2000 Detailed Breakdown:
├─ ΔE₀₀: {self.delta_e:.3f}
├─ Składowe:
│  ├─ L': {iv.delta_L_prime:.3f} (SL={iv.SL:.3f})
│  ├─ C': {iv.delta_C_prime:.3f} (SC={iv.SC:.3f})
│  ├─ H': {iv.delta_H_prime:.3f} (SH={iv.SH:.3f})
│  └─ RT: {iv.RT:.3f} (korekcja obrotu)
├─ Funkcje wagowe:
│  ├─ T: {iv.T:.3f}
│  ├─ G: {iv.G:.3f}
│  └─ RC: {iv.RC:.3f}
├─ Składowe ważone:
│  ├─ L-component: {iv.L_component:.3f}
│  ├─ C-component: {iv.C_component:.3f}
│  ├─ H-component: {iv.H_component:.3f}
│  └─ Interaction: {iv.interaction_term:.3f}
└─ Interpretacja: {self.interpretation}"""
    
    def component_contributions(self) -> dict:
        """Oblicza wkład poszczególnych składowych"""
        if self.intermediate_values is None:
            return {'L*': 0, 'C*': 0, 'H*': 0, 'Interaction': 0}
        
        iv = self.intermediate_values
        total_squared = (iv.L_component**2 + iv.C_component**2 + 
                        iv.H_component**2 + abs(iv.interaction_term))
        
        if total_squared > 0:
            return {
                'L*': (iv.L_component**2 / total_squared) * 100,
                'C*': (iv.C_component**2 / total_squared) * 100,
                'H*': (iv.H_component**2 / total_squared) * 100,
                'Interaction': (abs(iv.interaction_term) / total_squared) * 100
            }
        else:
            return {'L*': 0, 'C*': 0, 'H*': 0, 'Interaction': 0}
```

### Główna Klasa Kalkulatora

```python
class CIEDE2000Calculator:
    """Kalkulator Delta E CIEDE2000"""
    
    def __init__(self, parameters: Optional[CIEDE2000Parameters] = None):
        """
        Inicjalizuje kalkulator CIEDE2000
        
        Args:
            parameters: Parametry CIEDE2000 (domyślnie standardowe)
        """
        self.parameters = parameters or CIEDE2000Parameters.standard()
        self.method_name = "CIEDE2000"
        self.year_introduced = 2001
        self.description = "Najbardziej zaawansowana formuła Delta E z korekcją obrotu"
    
    def calculate(self, color1: Tuple[float, float, float], 
                 color2: Tuple[float, float, float],
                 return_details: bool = False,
                 include_intermediate: bool = False) -> Union[float, CIEDE2000Result]:
        """
        Oblicza Delta E CIEDE2000 między dwoma kolorami LAB
        
        Args:
            color1: Pierwszy kolor (L*, a*, b*)
            color2: Drugi kolor (L*, a*, b*)
            return_details: Czy zwrócić szczegółowe informacje
            include_intermediate: Czy dołączyć wartości pośrednie
        
        Returns:
            Wartość Delta E lub obiekt CIEDE2000Result
        """
        # Walidacja
        self._validate_lab_color(color1, "color1")
        self._validate_lab_color(color2, "color2")
        
        # Rozpakowanie kolorów
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        
        # Krok 1: Obliczenie średniej jasności
        L_bar = (L1 + L2) / 2.0
        
        # Krok 2: Obliczenie początkowej chromatyczności
        C1_initial = math.sqrt(a1**2 + b1**2)
        C2_initial = math.sqrt(a2**2 + b2**2)
        C_bar_initial = (C1_initial + C2_initial) / 2.0
        
        # Krok 3: Obliczenie G (korekcja a*)
        C_bar_7 = C_bar_initial**7
        G = 0.5 * (1 - math.sqrt(C_bar_7 / (C_bar_7 + 25**7)))
        
        # Krok 4: Skorygowane wartości a'
        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2
        
        # Krok 5: Nowe chromatyczności
        C1_prime = math.sqrt(a1_prime**2 + b1**2)
        C2_prime = math.sqrt(a2_prime**2 + b2**2)
        
        # Krok 6: Nowe odcienie (w stopniach)
        h1_prime = self._calculate_hue_angle(a1_prime, b1)
        h2_prime = self._calculate_hue_angle(a2_prime, b2)
        
        # Krok 7: Różnice
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime
        delta_h_prime = self._calculate_hue_difference(h1_prime, h2_prime, C1_prime, C2_prime)
        delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))
        
        # Krok 8: Średnie wartości
        L_bar_prime = (L1 + L2) / 2.0
        C_bar_prime = (C1_prime + C2_prime) / 2.0
        h_bar_prime = self._calculate_mean_hue(h1_prime, h2_prime, C1_prime, C2_prime)
        
        # Krok 9: Funkcje wagowe
        SL = self._calculate_SL(L_bar_prime)
        SC = self._calculate_SC(C_bar_prime)
        T = self._calculate_T(h_bar_prime)
        SH = self._calculate_SH(C_bar_prime, T)
        
        # Krok 10: Korekcja obrotu
        delta_theta = 30 * math.exp(-((h_bar_prime - 275) / 25)**2)
        C_bar_prime_7 = C_bar_prime**7
        RC = 2 * math.sqrt(C_bar_prime_7 / (C_bar_prime_7 + 25**7))
        RT = -math.sin(math.radians(2 * delta_theta)) * RC
        
        # Krok 11: Składowe ważone
        L_component = delta_L_prime / (self.parameters.kL * SL)
        C_component = delta_C_prime / (self.parameters.kC * SC)
        H_component = delta_H_prime / (self.parameters.kH * SH)
        interaction_term = RT * C_component * H_component
        
        # Krok 12: Delta E CIEDE2000
        delta_e = math.sqrt(L_component**2 + C_component**2 + H_component**2 + interaction_term)
        
        if return_details:
            interpretation = self._interpret_delta_e(delta_e)
            
            intermediate_values = None
            if include_intermediate:
                intermediate_values = CIEDE2000IntermediateValues(
                    L1=L1, a1=a1, b1=b1, L2=L2, a2=a2, b2=b2,
                    L_bar=L_bar, C_bar_initial=C_bar_initial,
                    G=G, a1_prime=a1_prime, a2_prime=a2_prime,
                    C1_prime=C1_prime, C2_prime=C2_prime,
                    h1_prime=h1_prime, h2_prime=h2_prime,
                    delta_L_prime=delta_L_prime, delta_C_prime=delta_C_prime,
                    delta_h_prime=delta_h_prime, delta_H_prime=delta_H_prime,
                    L_bar_prime=L_bar_prime, C_bar_prime=C_bar_prime,
                    h_bar_prime=h_bar_prime, SL=SL, SC=SC, SH=SH, T=T,
                    delta_theta=delta_theta, RC=RC, RT=RT,
                    L_component=L_component, C_component=C_component,
                    H_component=H_component, interaction_term=interaction_term,
                    delta_E=delta_e
                )
            
            return CIEDE2000Result(
                delta_e=delta_e,
                color1_lab=color1,
                color2_lab=color2,
                parameters=self.parameters,
                intermediate_values=intermediate_values,
                interpretation=interpretation
            )
        
        return delta_e
    
    def _calculate_hue_angle(self, a_prime: float, b: float) -> float:
        """
        Oblicza kąt odcienia w stopniach [0, 360)
        """
        if a_prime == 0 and b == 0:
            return 0.0
        
        hue_rad = math.atan2(b, a_prime)
        hue_deg = math.degrees(hue_rad)
        
        # Normalizacja do zakresu [0, 360)
        if hue_deg < 0:
            hue_deg += 360
        
        return hue_deg
    
    def _calculate_hue_difference(self, h1: float, h2: float, C1: float, C2: float) -> float:
        """
        Oblicza różnicę odcienia z uwzględnieniem cykliczności
        """
        if C1 * C2 == 0:
            return 0.0
        
        delta_h = h2 - h1
        
        if abs(delta_h) <= 180:
            return delta_h
        elif delta_h > 180:
            return delta_h - 360
        else:
            return delta_h + 360
    
    def _calculate_mean_hue(self, h1: float, h2: float, C1: float, C2: float) -> float:
        """
        Oblicza średni odcień z uwzględnieniem cykliczności
        """
        if C1 * C2 == 0:
            return h1 + h2
        
        if abs(h1 - h2) <= 180:
            return (h1 + h2) / 2.0
        elif (h1 + h2) < 360:
            return (h1 + h2 + 360) / 2.0
        else:
            return (h1 + h2 - 360) / 2.0
    
    def _calculate_SL(self, L_bar_prime: float) -> float:
        """
        Oblicza funkcję wagową SL dla jasności
        """
        return 1 + (0.015 * (L_bar_prime - 50)**2) / math.sqrt(20 + (L_bar_prime - 50)**2)
    
    def _calculate_SC(self, C_bar_prime: float) -> float:
        """
        Oblicza funkcję wagową SC dla chromatyczności
        """
        return 1 + 0.045 * C_bar_prime
    
    def _calculate_T(self, h_bar_prime: float) -> float:
        """
        Oblicza funkcję T dla odcienia
        """
        h_rad = math.radians(h_bar_prime)
        return (1 - 0.17 * math.cos(h_rad - math.radians(30)) +
                0.24 * math.cos(2 * h_rad) +
                0.32 * math.cos(3 * h_rad + math.radians(6)) -
                0.20 * math.cos(4 * h_rad - math.radians(63)))
    
    def _calculate_SH(self, C_bar_prime: float, T: float) -> float:
        """
        Oblicza funkcję wagową SH dla odcienia
        """
        return 1 + 0.015 * C_bar_prime * T
    
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
        """Interpretuje wartość Delta E CIEDE2000"""
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

## Przykłady Podstawowe

### Demonstracja Użycia

```python
def demonstrate_ciede2000_basic():
    """Demonstracja podstawowego użycia CIEDE2000"""
    print("=== Demonstracja CIEDE2000 ===")
    
    # Kolory testowe
    red = (53.24, 80.09, 67.20)      # Czerwony
    orange = (74.93, 23.93, 78.95)   # Pomarańczowy
    
    print(f"Kolor 1 (czerwony): LAB{red}")
    print(f"Kolor 2 (pomarańczowy): LAB{orange}")
    print("-" * 50)
    
    # Kalkulator CIEDE2000
    calculator = CIEDE2000Calculator()
    
    # Proste obliczenie
    delta_e = calculator.calculate(red, orange)
    print(f"CIEDE2000: {delta_e:.3f}")
    
    # Szczegółowe informacje
    result = calculator.calculate(red, orange, return_details=True, include_intermediate=True)
    print(f"\n{result.detailed_breakdown()}")
    
    # Wkład składowych
    contributions = result.component_contributions()
    print(f"\n=== Wkład składowych ===")
    for component, value in contributions.items():
        print(f"{component}: {value:.1f}%")

# demonstrate_ciede2000_basic()
```

### Porównanie z Innymi Metodami

```python
def compare_all_delta_e_methods():
    """Porównuje wszystkie metody Delta E"""
    from delta_e_calculator import DeltaECalculator, DeltaEMethod
    
    # Kolory testowe
    test_colors = [
        ((50, 0, 0), (55, 0, 0), "Zmiana jasności"),
        ((50, 20, 0), (50, 25, 0), "Zmiana a*"),
        ((50, 0, 20), (50, 0, 25), "Zmiana b*"),
        ((50, 20, 20), (50, 25, 25), "Zmiana chromatyczności"),
        ((30, 0, -50), (30, 0, -45), "Niebieskie (ciemne)"),
        ((70, 0, -50), (70, 0, -45), "Niebieskie (jasne)"),
        ((50, 60, 60), (55, 65, 65), "Wysoka chromatyczność"),
        ((50, 5, 5), (55, 10, 10), "Niska chromatyczność")
    ]
    
    print("=== Porównanie Wszystkich Metod Delta E ===")
    
    # Kalkulatory
    cie76_calc = DeltaECalculator(DeltaEMethod.CIE76)
    cie94_graphic_calc = CIE94Calculator(CIE94Application.GRAPHIC_ARTS)
    cie94_textile_calc = CIE94Calculator(CIE94Application.TEXTILES)
    ciede2000_calc = CIEDE2000Calculator()
    
    for color1, color2, description in test_colors:
        print(f"\n--- {description} ---")
        print(f"Kolory: {color1} → {color2}")
        
        # Obliczenia
        cie76_result = cie76_calc.calculate(color1, color2)
        cie94_graphic = cie94_graphic_calc.calculate(color1, color2)
        cie94_textile = cie94_textile_calc.calculate(color1, color2)
        ciede2000_result = ciede2000_calc.calculate(color1, color2)
        
        print(f"CIE76:           {cie76_result:.3f}")
        print(f"CIE94 (grafika): {cie94_graphic:.3f}")
        print(f"CIE94 (tekstyl): {cie94_textile:.3f}")
        print(f"CIEDE2000:       {ciede2000_result:.3f}")
        
        # Analiza różnic
        methods = {
            'CIE76': cie76_result,
            'CIE94_G': cie94_graphic,
            'CIE94_T': cie94_textile,
            'CIEDE2000': ciede2000_result
        }
        
        min_method = min(methods, key=methods.get)
        max_method = max(methods, key=methods.get)
        range_value = methods[max_method] - methods[min_method]
        
        print(f"Zakres: {range_value:.3f} ({min_method} → {max_method})")

# compare_all_delta_e_methods()
```

---

## Analiza Funkcji Korekcyjnych

### Wizualizacja Funkcji Wagowych

```python
import matplotlib.pyplot as plt

def visualize_ciede2000_weighting_functions():
    """Wizualizuje funkcje wagowe CIEDE2000"""
    calculator = CIEDE2000Calculator()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Funkcja SL (jasność)
    L_values = np.linspace(0, 100, 1000)
    SL_values = [calculator._calculate_SL(L) for L in L_values]
    
    axes[0, 0].plot(L_values, SL_values, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('L* (Jasność)')
    axes[0, 0].set_ylabel('SL (Waga jasności)')
    axes[0, 0].set_title('Funkcja Wagowa SL')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='SL = 1')
    axes[0, 0].legend()
    
    # 2. Funkcja SC (chromatyczność)
    C_values = np.linspace(0, 100, 1000)
    SC_values = [calculator._calculate_SC(C) for C in C_values]
    
    axes[0, 1].plot(C_values, SC_values, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('C* (Chromatyczność)')
    axes[0, 1].set_ylabel('SC (Waga chromatyczności)')
    axes[0, 1].set_title('Funkcja Wagowa SC')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Funkcja T (odcień)
    h_values = np.linspace(0, 360, 1000)
    T_values = [calculator._calculate_T(h) for h in h_values]
    
    axes[1, 0].plot(h_values, T_values, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('h* (Odcień, stopnie)')
    axes[1, 0].set_ylabel('T (Funkcja odcienia)')
    axes[1, 0].set_title('Funkcja T dla Odcienia')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 360)
    
    # 4. Funkcja RT (korekcja obrotu)
    h_values_rt = np.linspace(200, 350, 1000)
    C_test = 50  # Stała chromatyczność dla testu
    RT_values = []
    
    for h in h_values_rt:
        delta_theta = 30 * math.exp(-((h - 275) / 25)**2)
        C_7 = C_test**7
        RC = 2 * math.sqrt(C_7 / (C_7 + 25**7))
        RT = -math.sin(math.radians(2 * delta_theta)) * RC
        RT_values.append(RT)
    
    axes[1, 1].plot(h_values_rt, RT_values, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('h* (Odcień, stopnie)')
    axes[1, 1].set_ylabel('RT (Korekcja obrotu)')
    axes[1, 1].set_title(f'Funkcja Korekcji Obrotu RT (C*={C_test})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# visualize_ciede2000_weighting_functions()
```

### Analiza Korekcji G

```python
def analyze_g_correction():
    """Analizuje wpływ korekcji G na wartości a*"""
    print("=== Analiza Korekcji G ===")
    
    # Test różnych poziomów chromatyczności
    test_cases = [
        (0, 0, "Szary (brak chromatyczności)"),
        (10, 10, "Niska chromatyczność"),
        (30, 30, "Średnia chromatyczność"),
        (60, 60, "Wysoka chromatyczność"),
        (100, 100, "Bardzo wysoka chromatyczność")
    ]
    
    calculator = CIEDE2000Calculator()
    
    for a, b, description in test_cases:
        # Obliczenie chromatyczności
        C = math.sqrt(a**2 + b**2)
        
        # Obliczenie G dla pary kolorów o tej samej chromatyczności
        C_bar = C  # Średnia = C dla identycznych kolorów
        C_bar_7 = C_bar**7
        G = 0.5 * (1 - math.sqrt(C_bar_7 / (C_bar_7 + 25**7)))
        
        # Skorygowane a'
        a_prime = (1 + G) * a
        
        print(f"\n{description}:")
        print(f"  Oryginalne: a*={a}, b*={b}, C*={C:.2f}")
        print(f"  Korekcja G: {G:.4f}")
        print(f"  Skorygowane: a'={a_prime:.2f} (zmiana: {((a_prime-a)/a*100) if a != 0 else 0:.1f}%)")
        
        # Wpływ na chromatyczność
        C_prime = math.sqrt(a_prime**2 + b**2)
        print(f"  Nowa chromatyczność C': {C_prime:.2f} (zmiana: {((C_prime-C)/C*100) if C != 0 else 0:.1f}%)")

# analyze_g_correction()
```

---

## Podsumowanie Części 5a

W tej części omówiliśmy:

1. **Teoretyczne podstawy CIEDE2000** - najbardziej zaawansowanej formuły Delta E
2. **Kompleksną implementację** - wszystkie kroki algorytmu
3. **Struktury danych** - szczegółowe przechowywanie wyników
4. **Funkcje korekcyjne** - G, SL, SC, SH, T, RT
5. **Przykłady podstawowe** - demonstracja użycia

### Kluczowe Cechy CIEDE2000

✅ **Najwyższa dokładność** - najlepsza korelacja z percepcją  
✅ **Kompleksowe korekcje** - wszystkie aspekty percepcji  
✅ **Korekcja obrotu** - specjalne traktowanie niebieskich  
✅ **Funkcje wagowe** - dostosowane do ludzkiego oka  
❌ **Złożoność** - najbardziej skomplikowana implementacja  
❌ **Wydajność** - najwolniejsza w obliczeniach  

### Co dalej?

**Część 5b** będzie zawierać:
- Optymalizacje wydajności (NumPy, Numba)
- Batch processing dla dużych zbiorów danych
- Zaawansowane analizy i porównania
- Praktyczne zastosowania i case studies
- Benchmarki wydajności wszystkich metod

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Część 5a - CIEDE2000 implementacja podstawowa