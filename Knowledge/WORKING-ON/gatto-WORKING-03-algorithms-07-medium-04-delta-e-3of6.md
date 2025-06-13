# Delta E Color Distance - Część 3: CIE76 - Szczegółowa Implementacja

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 2-3 godziny | **Złożoność**: O(1)

---

## Wprowadzenie do CIE76

### Historia i Kontekst

Delta E CIE76 (znane również jako Delta E*ab) to pierwsza standardowa formuła do obliczania różnic kolorów w przestrzeni LAB, wprowadzona przez Międzynarodową Komisję Oświetleniową (CIE) w 1976 roku.

### Charakterystyka CIE76

- **Prostota**: Najprostsza implementacja - odległość euklidesowa
- **Szybkość**: Najszybsza w obliczeniach
- **Ograniczenia**: Nie uwzględnia percepcji ludzkiego oka
- **Zastosowania**: Podstawowe porównania, szybkie obliczenia

---

## Matematyczne Podstawy

### Formuła CIE76

Delta E CIE76 to prosta odległość euklidesowa w przestrzeni LAB:

```
ΔE*ab = √[(ΔL*)² + (Δa*)² + (Δb*)²]
```

Gdzie:
- `ΔL* = L₂* - L₁*` (różnica jasności)
- `Δa* = a₂* - a₁*` (różnica w osi zielony-czerwony)
- `Δb* = b₂* - b₁*` (różnica w osi niebieski-żółty)

### Interpretacja Geometryczna

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_cie76_distance():
    """Wizualizuje odległość CIE76 w przestrzeni LAB"""
    fig = plt.figure(figsize=(12, 5))
    
    # Wykres 3D
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Przykładowe kolory
    color1 = [50, 20, -10]  # L*, a*, b*
    color2 = [55, 25, -5]
    
    # Punkty w przestrzeni LAB
    ax1.scatter(*color1, color='red', s=100, label='Kolor 1')
    ax1.scatter(*color2, color='blue', s=100, label='Kolor 2')
    
    # Linia łącząca (odległość)
    ax1.plot([color1[0], color2[0]], 
             [color1[1], color2[1]], 
             [color1[2], color2[2]], 
             'k--', linewidth=2, label='Odległość CIE76')
    
    ax1.set_xlabel('L* (Jasność)')
    ax1.set_ylabel('a* (Zielony-Czerwony)')
    ax1.set_zlabel('b* (Niebieski-Żółty)')
    ax1.set_title('Odległość CIE76 w przestrzeni LAB')
    ax1.legend()
    
    # Wykres 2D - rzut na płaszczyznę a*b*
    ax2 = fig.add_subplot(122)
    ax2.scatter(color1[1], color1[2], color='red', s=100, label='Kolor 1')
    ax2.scatter(color2[1], color2[2], color='blue', s=100, label='Kolor 2')
    ax2.plot([color1[1], color2[1]], [color1[2], color2[2]], 
             'k--', linewidth=2, label='Projekcja odległości')
    
    ax2.set_xlabel('a* (Zielony-Czerwony)')
    ax2.set_ylabel('b* (Niebieski-Żółty)')
    ax2.set_title('Rzut na płaszczyznę chromatyczności')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Uruchomienie wizualizacji
# visualize_cie76_distance()
```

---

## Implementacja CIE76

### Podstawowa Klasa

```python
import math
import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

@dataclass
class CIE76Result:
    """Wynik obliczenia Delta E CIE76"""
    delta_e: float
    delta_l: float
    delta_a: float
    delta_b: float
    color1_lab: Tuple[float, float, float]
    color2_lab: Tuple[float, float, float]
    interpretation: str
    
    def __str__(self):
        return f"ΔE*ab: {self.delta_e:.3f} ({self.interpretation})"
    
    def detailed_breakdown(self) -> str:
        """Zwraca szczegółowy rozkład różnic"""
        return f"""Delta E CIE76 Breakdown:
├─ ΔE*ab: {self.delta_e:.3f}
├─ ΔL*: {self.delta_l:.3f} (jasność)
├─ Δa*: {self.delta_a:.3f} (zielony-czerwony)
├─ Δb*: {self.delta_b:.3f} (niebieski-żółty)
└─ Interpretacja: {self.interpretation}"""

class CIE76Calculator:
    """Specjalizowany kalkulator Delta E CIE76"""
    
    def __init__(self):
        """Inicjalizuje kalkulator CIE76"""
        self.method_name = "CIE76"
        self.year_introduced = 1976
        self.description = "Prosta odległość euklidesowa w przestrzeni LAB"
    
    def calculate(self, color1: Tuple[float, float, float], 
                 color2: Tuple[float, float, float],
                 return_details: bool = False) -> Union[float, CIE76Result]:
        """
        Oblicza Delta E CIE76 między dwoma kolorami LAB
        
        Args:
            color1: Pierwszy kolor (L*, a*, b*)
            color2: Drugi kolor (L*, a*, b*)
            return_details: Czy zwrócić szczegółowe informacje
        
        Returns:
            Wartość Delta E lub obiekt CIE76Result
        """
        # Walidacja
        self._validate_lab_color(color1, "color1")
        self._validate_lab_color(color2, "color2")
        
        # Obliczenie różnic
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        
        delta_l = L2 - L1
        delta_a = a2 - a1
        delta_b = b2 - b1
        
        # Delta E CIE76
        delta_e = math.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
        
        if return_details:
            interpretation = self._interpret_delta_e(delta_e)
            return CIE76Result(
                delta_e=delta_e,
                delta_l=delta_l,
                delta_a=delta_a,
                delta_b=delta_b,
                color1_lab=color1,
                color2_lab=color2,
                interpretation=interpretation
            )
        
        return delta_e
    
    def calculate_components(self, color1: Tuple[float, float, float], 
                           color2: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """
        Oblicza składowe Delta E CIE76
        
        Returns:
            Tuple (delta_e, delta_l, delta_a, delta_b)
        """
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        
        delta_l = L2 - L1
        delta_a = a2 - a1
        delta_b = b2 - b1
        delta_e = math.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
        
        return delta_e, delta_l, delta_a, delta_b
    
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
        """Interpretuje wartość Delta E CIE76"""
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
```

---

## Optymalizacje Wydajności

### Wektoryzowana Implementacja

```python
class OptimizedCIE76Calculator:
    """Zoptymalizowana wersja kalkulatora CIE76"""
    
    def __init__(self):
        self.base_calculator = CIE76Calculator()
    
    def calculate_batch(self, colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """
        Oblicza Delta E CIE76 dla batch'a kolorów
        
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
        
        # Wektoryzowane obliczenie różnic
        differences = colors2 - colors1
        
        # Delta E CIE76 = sqrt(sum(differences^2, axis=1))
        delta_e_values = np.sqrt(np.sum(differences**2, axis=1))
        
        return delta_e_values
    
    def calculate_matrix(self, colors: np.ndarray) -> np.ndarray:
        """
        Oblicza macierz Delta E między wszystkimi parami kolorów
        
        Args:
            colors: Array kolorów LAB (N, 3)
        
        Returns:
            Macierz Delta E (N, N)
        """
        n_colors = colors.shape[0]
        delta_e_matrix = np.zeros((n_colors, n_colors))
        
        for i in range(n_colors):
            for j in range(i, n_colors):
                if i == j:
                    delta_e_matrix[i, j] = 0.0
                else:
                    delta_e = self.base_calculator.calculate(colors[i], colors[j])
                    delta_e_matrix[i, j] = delta_e
                    delta_e_matrix[j, i] = delta_e  # Symetria
        
        return delta_e_matrix
    
    def find_closest_colors(self, target_color: np.ndarray, 
                          palette: np.ndarray, 
                          n_closest: int = 5) -> List[Tuple[int, float]]:
        """
        Znajduje najbliższe kolory w palecie
        
        Args:
            target_color: Kolor docelowy (3,)
            palette: Paleta kolorów (N, 3)
            n_closest: Liczba najbliższych kolorów
        
        Returns:
            Lista (indeks, delta_e) posortowana według odległości
        """
        # Rozszerzenie target_color do (1, 3)
        target_expanded = target_color.reshape(1, -1)
        
        # Obliczenie Delta E dla wszystkich kolorów w palecie
        delta_e_values = self.calculate_batch(
            np.repeat(target_expanded, len(palette), axis=0),
            palette
        )
        
        # Sortowanie według odległości
        sorted_indices = np.argsort(delta_e_values)
        
        # Zwrócenie n najbliższych
        closest = [(int(idx), float(delta_e_values[idx])) 
                  for idx in sorted_indices[:n_closest]]
        
        return closest
```

### Implementacja z Numba

```python
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(x):
        return range(x)

class NumbaCIE76Calculator:
    """Kalkulator CIE76 z przyspieszeniem Numba"""
    
    def __init__(self):
        self.numba_available = NUMBA_AVAILABLE
        if not NUMBA_AVAILABLE:
            print("Ostrzeżenie: Numba niedostępna, używam standardowej implementacji")
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_batch_numba(colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """
        Numba-zoptymalizowane obliczenie batch Delta E CIE76
        """
        n = colors1.shape[0]
        results = np.zeros(n)
        
        for i in prange(n):
            dL = colors2[i, 0] - colors1[i, 0]
            da = colors2[i, 1] - colors1[i, 1]
            db = colors2[i, 2] - colors1[i, 2]
            results[i] = math.sqrt(dL*dL + da*da + db*db)
        
        return results
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_matrix_numba(colors: np.ndarray) -> np.ndarray:
        """
        Numba-zoptymalizowane obliczenie macierzy Delta E
        """
        n = colors.shape[0]
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dL = colors[j, 0] - colors[i, 0]
                da = colors[j, 1] - colors[i, 1]
                db = colors[j, 2] - colors[i, 2]
                delta_e = math.sqrt(dL*dL + da*da + db*db)
                matrix[i, j] = delta_e
                matrix[j, i] = delta_e
        
        return matrix
    
    def calculate_batch(self, colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """
        Oblicza Delta E CIE76 dla batch'a kolorów z przyspieszeniem Numba
        """
        if self.numba_available:
            return self._calculate_batch_numba(colors1, colors2)
        else:
            # Fallback do NumPy
            differences = colors2 - colors1
            return np.sqrt(np.sum(differences**2, axis=1))
    
    def calculate_matrix(self, colors: np.ndarray) -> np.ndarray:
        """
        Oblicza macierz Delta E z przyspieszeniem Numba
        """
        if self.numba_available:
            return self._calculate_matrix_numba(colors)
        else:
            # Fallback do standardowej implementacji
            calc = OptimizedCIE76Calculator()
            return calc.calculate_matrix(colors)
```

---

## Analiza Składowych

### Dekompozycja Delta E

```python
class CIE76ComponentAnalyzer:
    """Analizator składowych Delta E CIE76"""
    
    def __init__(self):
        self.calculator = CIE76Calculator()
    
    def analyze_components(self, color1: Tuple[float, float, float], 
                         color2: Tuple[float, float, float]) -> dict:
        """
        Analizuje wkład poszczególnych składowych do Delta E
        
        Returns:
            Słownik z analizą składowych
        """
        delta_e, delta_l, delta_a, delta_b = self.calculator.calculate_components(color1, color2)
        
        # Kwadrat składowych (wkład do sumy)
        l_squared = delta_l**2
        a_squared = delta_a**2
        b_squared = delta_b**2
        total_squared = l_squared + a_squared + b_squared
        
        # Procentowy wkład każdej składowej
        l_contribution = (l_squared / total_squared) * 100 if total_squared > 0 else 0
        a_contribution = (a_squared / total_squared) * 100 if total_squared > 0 else 0
        b_contribution = (b_squared / total_squared) * 100 if total_squared > 0 else 0
        
        # Dominująca składowa
        contributions = {'L*': l_contribution, 'a*': a_contribution, 'b*': b_contribution}
        dominant_component = max(contributions, key=contributions.get)
        
        return {
            'delta_e': delta_e,
            'components': {
                'delta_l': delta_l,
                'delta_a': delta_a,
                'delta_b': delta_b
            },
            'contributions_percent': {
                'L*': l_contribution,
                'a*': a_contribution,
                'b*': b_contribution
            },
            'dominant_component': dominant_component,
            'dominant_value': contributions[dominant_component],
            'interpretation': self._interpret_components(contributions)
        }
    
    def _interpret_components(self, contributions: dict) -> str:
        """
        Interpretuje dominację składowych
        """
        max_contrib = max(contributions.values())
        
        if max_contrib > 70:
            return "Jedna składowa dominuje"
        elif max_contrib > 50:
            return "Jedna składowa przeważa"
        elif max_contrib > 40:
            return "Umiarkowana dominacja"
        else:
            return "Równomierne rozłożenie składowych"
    
    def compare_component_sensitivity(self, base_color: Tuple[float, float, float],
                                   step_size: float = 1.0) -> dict:
        """
        Porównuje wrażliwość na zmiany w różnych składowych
        
        Args:
            base_color: Kolor bazowy
            step_size: Rozmiar kroku zmiany
        
        Returns:
            Analiza wrażliwości składowych
        """
        L, a, b = base_color
        
        # Zmiany w poszczególnych składowych
        color_l_plus = (L + step_size, a, b)
        color_a_plus = (L, a + step_size, b)
        color_b_plus = (L, a, b + step_size)
        
        # Obliczenie Delta E dla każdej zmiany
        delta_e_l = self.calculator.calculate(base_color, color_l_plus)
        delta_e_a = self.calculator.calculate(base_color, color_a_plus)
        delta_e_b = self.calculator.calculate(base_color, color_b_plus)
        
        return {
            'step_size': step_size,
            'sensitivity': {
                'L*': delta_e_l,
                'a*': delta_e_a,
                'b*': delta_e_b
            },
            'most_sensitive': max(['L*', 'a*', 'b*'], 
                                key=lambda x: {'L*': delta_e_l, 'a*': delta_e_a, 'b*': delta_e_b}[x]),
            'sensitivity_ratio': {
                'L*/a*': delta_e_l / delta_e_a if delta_e_a > 0 else float('inf'),
                'L*/b*': delta_e_l / delta_e_b if delta_e_b > 0 else float('inf'),
                'a*/b*': delta_e_a / delta_e_b if delta_e_b > 0 else float('inf')
            }
        }
```

---

## Przykłady Praktyczne

### Podstawowe Użycie

```python
def demonstrate_cie76_basic():
    """Demonstracja podstawowego użycia CIE76"""
    calculator = CIE76Calculator()
    
    # Przykładowe kolory
    red = (53.24, 80.09, 67.20)      # Czerwony
    orange = (74.93, 23.93, 78.95)   # Pomarańczowy
    
    print("=== Podstawowe obliczenie CIE76 ===")
    
    # Proste obliczenie
    delta_e = calculator.calculate(red, orange)
    print(f"Delta E CIE76: {delta_e:.3f}")
    
    # Szczegółowe informacje
    result = calculator.calculate(red, orange, return_details=True)
    print(f"\n{result.detailed_breakdown()}")
    
    # Analiza składowych
    analyzer = CIE76ComponentAnalyzer()
    analysis = analyzer.analyze_components(red, orange)
    
    print(f"\n=== Analiza składowych ===")
    print(f"Dominująca składowa: {analysis['dominant_component']} ({analysis['dominant_value']:.1f}%)")
    print(f"Interpretacja: {analysis['interpretation']}")
    
    for component, contribution in analysis['contributions_percent'].items():
        print(f"{component}: {contribution:.1f}%")

# demonstrate_cie76_basic()
```

### Porównanie z Innymi Metodami

```python
def compare_cie76_with_others():
    """Porównuje CIE76 z innymi metodami Delta E"""
    from delta_e_calculator import DeltaECalculator, DeltaEMethod
    
    # Kolory testowe
    color1 = (50, 20, -10)
    color2 = (55, 25, -5)
    
    print("=== Porównanie metod Delta E ===")
    print(f"Kolor 1: LAB{color1}")
    print(f"Kolor 2: LAB{color2}")
    print("-" * 40)
    
    # CIE76 (nasza implementacja)
    cie76_calc = CIE76Calculator()
    cie76_result = cie76_calc.calculate(color1, color2)
    print(f"CIE76 (dedykowana): {cie76_result:.3f}")
    
    # Inne metody
    methods = [
        (DeltaEMethod.CIE76, "CIE76 (ogólna)"),
        (DeltaEMethod.CIE94, "CIE94"),
        (DeltaEMethod.CIEDE2000, "CIEDE2000"),
        (DeltaEMethod.CMC, "CMC")
    ]
    
    for method, name in methods:
        calc = DeltaECalculator(method)
        result = calc.calculate(color1, color2)
        print(f"{name:15}: {result:.3f}")
    
    # Analiza różnic
    print(f"\n=== Analiza różnic ===")
    cie94_calc = DeltaECalculator(DeltaEMethod.CIE94)
    ciede2000_calc = DeltaECalculator(DeltaEMethod.CIEDE2000)
    
    cie94_result = cie94_calc.calculate(color1, color2)
    ciede2000_result = ciede2000_calc.calculate(color1, color2)
    
    print(f"CIE76 vs CIE94: {abs(cie76_result - cie94_result):.3f} różnicy")
    print(f"CIE76 vs CIEDE2000: {abs(cie76_result - ciede2000_result):.3f} różnicy")

# compare_cie76_with_others()
```

### Analiza Wydajności

```python
import time

def benchmark_cie76_performance():
    """Benchmark wydajności różnych implementacji CIE76"""
    # Generowanie danych testowych
    np.random.seed(42)
    n_colors = 10000
    colors1 = np.random.rand(n_colors, 3) * [100, 255, 255] - [0, 128, 128]
    colors2 = np.random.rand(n_colors, 3) * [100, 255, 255] - [0, 128, 128]
    
    print(f"=== Benchmark CIE76 ({n_colors} par kolorów) ===")
    
    # 1. Podstawowa implementacja (pętla)
    basic_calc = CIE76Calculator()
    start_time = time.time()
    
    basic_results = []
    for i in range(n_colors):
        result = basic_calc.calculate(tuple(colors1[i]), tuple(colors2[i]))
        basic_results.append(result)
    
    basic_time = time.time() - start_time
    print(f"Podstawowa (pętla):     {basic_time:.3f}s")
    
    # 2. Zoptymalizowana NumPy
    optimized_calc = OptimizedCIE76Calculator()
    start_time = time.time()
    
    optimized_results = optimized_calc.calculate_batch(colors1, colors2)
    
    optimized_time = time.time() - start_time
    print(f"Zoptymalizowana NumPy:  {optimized_time:.3f}s")
    
    # 3. Numba (jeśli dostępna)
    if NUMBA_AVAILABLE:
        numba_calc = NumbaCIE76Calculator()
        
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

# benchmark_cie76_performance()
```

---

## Ograniczenia i Problemy CIE76

### Znane Problemy

```python
def demonstrate_cie76_limitations():
    """Demonstruje ograniczenia CIE76"""
    calculator = CIE76Calculator()
    
    print("=== Ograniczenia CIE76 ===")
    
    # Problem 1: Niejednorodność percepcji
    print("\n1. Niejednorodność percepcji w przestrzeni LAB:")
    
    # Kolory o tej samej Delta E CIE76, ale różnej percepcji
    blue_pair1 = ((30, 0, -50), (30, 0, -45))  # Ciemne niebieskie
    blue_pair2 = ((70, 0, -50), (70, 0, -45))  # Jasne niebieskie
    
    delta_e1 = calculator.calculate(blue_pair1[0], blue_pair1[1])
    delta_e2 = calculator.calculate(blue_pair2[0], blue_pair2[1])
    
    print(f"Ciemne niebieskie:  ΔE = {delta_e1:.3f}")
    print(f"Jasne niebieskie:   ΔE = {delta_e2:.3f}")
    print(f"Różnica percepcji może być inna mimo podobnej Delta E!")
    
    # Problem 2: Brak uwzględnienia chromatyczności
    print("\n2. Brak uwzględnienia chromatyczności:")
    
    gray_pair = ((50, 0, 0), (55, 0, 0))      # Szare
    red_pair = ((50, 50, 50), (55, 50, 50))   # Czerwone
    
    delta_e_gray = calculator.calculate(gray_pair[0], gray_pair[1])
    delta_e_red = calculator.calculate(red_pair[0], red_pair[1])
    
    print(f"Szare kolory:       ΔE = {delta_e_gray:.3f}")
    print(f"Czerwone kolory:    ΔE = {delta_e_red:.3f}")
    print(f"CIE76 traktuje je tak samo, ale oko może różnie!")
    
    # Problem 3: Wrażliwość na jasność
    print("\n3. Nadmierna wrażliwość na zmiany jasności:")
    
    analyzer = CIE76ComponentAnalyzer()
    
    # Test wrażliwości
    base_color = (50, 20, -10)
    sensitivity = analyzer.compare_component_sensitivity(base_color, step_size=1.0)
    
    print(f"Wrażliwość na zmiany o 1 jednostkę:")
    for component, value in sensitivity['sensitivity'].items():
        print(f"{component}: ΔE = {value:.3f}")
    
    print(f"Najbardziej wrażliwa składowa: {sensitivity['most_sensitive']}")

# demonstrate_cie76_limitations()
```

### Kiedy Używać CIE76

```python
def cie76_usage_guidelines():
    """Wytyczne dotyczące użycia CIE76"""
    print("=== Kiedy używać CIE76 ===")
    
    guidelines = {
        "✅ ZALECANE": [
            "Szybkie porównania kolorów",
            "Wstępna selekcja kolorów",
            "Aplikacje wymagające wysokiej wydajności",
            "Proste sortowanie według podobieństwa",
            "Gdy dokładność percepcyjna nie jest krytyczna",
            "Batch processing dużych ilości danych"
        ],
        "❌ NIEZALECANE": [
            "Precyzyjne dopasowywanie kolorów",
            "Kontrola jakości w druku",
            "Aplikacje medyczne/naukowe",
            "Gdy wymagana jest zgodność z percepcją",
            "Porównywanie bardzo podobnych kolorów",
            "Profesjonalne zarządzanie kolorami"
        ],
        "⚠️ OSTROŻNIE": [
            "Kolory o niskiej chromatyczności",
            "Bardzo ciemne lub bardzo jasne kolory",
            "Kolory w obszarach niejednorodności LAB",
            "Gdy różnice są blisko progów percepcji"
        ]
    }
    
    for category, items in guidelines.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n=== Alternatywy ===")
    alternatives = {
        "CIE94": "Lepsza dla tekstyliów i grafiki",
        "CIEDE2000": "Najbardziej dokładna percepcyjnie",
        "CMC": "Dobra dla przemysłu tekstylnego"
    }
    
    for method, description in alternatives.items():
        print(f"  • {method}: {description}")

# cie76_usage_guidelines()
```

---

## Podsumowanie Części 3

W tej części szczegółowo omówiliśmy:

1. **Matematyczne podstawy CIE76** - prosta odległość euklidesowa
2. **Implementację podstawową** - klasa CIE76Calculator
3. **Optymalizacje wydajności** - NumPy i Numba
4. **Analizę składowych** - dekompozycja Delta E
5. **Ograniczenia i problemy** - kiedy nie używać CIE76

### Kluczowe Cechy CIE76

✅ **Prostota**: Najłatwiejsza do zrozumienia i implementacji  
✅ **Szybkość**: Najszybsza w obliczeniach  
✅ **Stabilność**: Deterministyczna i przewidywalna  
❌ **Dokładność**: Nie uwzględnia percepcji ludzkiego oka  
❌ **Jednorodność**: Niejednorodna w przestrzeni LAB  

### Zastosowania

- **Szybkie porównania** kolorów
- **Wstępna selekcja** w dużych zbiorach
- **Batch processing** obrazów
- **Sortowanie** według podobieństwa

### Co dalej?

**Część 4** będzie zawierać:
- Szczegółową implementację CIE94
- Funkcje wagowe dla różnych aplikacji
- Porównanie z CIE76
- Optymalizacje specyficzne dla CIE94

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Część 3 - CIE76 szczegółowa implementacja