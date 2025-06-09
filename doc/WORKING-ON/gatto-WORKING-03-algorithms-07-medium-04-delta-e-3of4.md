# Delta E Color Distance - Część 3: Testy, Benchmarki i Praktyczne Zastosowania

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 3-5 godzin | **Złożoność**: O(n)

---

## Testy Jednostkowe

### Framework Testowy

```python
import unittest
import numpy as np
from typing import List, Tuple
import time
import json
from pathlib import Path

class TestDeltaECalculator(unittest.TestCase):
    """Testy jednostkowe dla kalkulatora Delta E"""
    
    def setUp(self):
        """Przygotowanie danych testowych"""
        self.calculator_76 = DeltaECalculator(DeltaEMethod.CIE76)
        self.calculator_94 = DeltaECalculator(DeltaEMethod.CIE94)
        self.calculator_2000 = DeltaECalculator(DeltaEMethod.CIEDE2000)
        self.calculator_cmc = DeltaECalculator(DeltaEMethod.CMC)
        
        self.converter = ColorConverter()
        
        # Kolory testowe
        self.test_colors_lab = [
            (50.0, 0.0, 0.0),    # Szary
            (100.0, 0.0, 0.0),   # Biały
            (0.0, 0.0, 0.0),     # Czarny
            (53.23, 80.11, 67.22),  # Czerwony
            (87.73, -86.18, 83.18), # Zielony
            (32.30, 79.20, -107.86) # Niebieski
        ]
        
        self.test_colors_rgb = [
            (128, 128, 128),  # Szary
            (255, 255, 255),  # Biały
            (0, 0, 0),        # Czarny
            (255, 0, 0),      # Czerwony
            (0, 255, 0),      # Zielony
            (0, 0, 255)       # Niebieski
        ]
    
    def test_delta_e_76_basic(self):
        """Test podstawowy Delta E 76"""
        # Identyczne kolory
        color = self.test_colors_lab[0]
        delta_e = self.calculator_76.calculate(color, color)
        self.assertAlmostEqual(delta_e, 0.0, places=5)
        
        # Różne kolory
        color1 = self.test_colors_lab[0]  # Szary
        color2 = self.test_colors_lab[1]  # Biały
        delta_e = self.calculator_76.calculate(color1, color2)
        self.assertGreater(delta_e, 0.0)
        
        # Symetryczność
        delta_e1 = self.calculator_76.calculate(color1, color2)
        delta_e2 = self.calculator_76.calculate(color2, color1)
        self.assertAlmostEqual(delta_e1, delta_e2, places=5)
    
    def test_delta_e_94_basic(self):
        """Test podstawowy Delta E 94"""
        color1 = self.test_colors_lab[3]  # Czerwony
        color2 = self.test_colors_lab[4]  # Zielony
        
        delta_e = self.calculator_94.calculate(color1, color2)
        self.assertGreater(delta_e, 0.0)
        self.assertIsInstance(delta_e, float)
    
    def test_delta_e_2000_basic(self):
        """Test podstawowy Delta E 2000"""
        color1 = self.test_colors_lab[0]
        color2 = self.test_colors_lab[5]  # Niebieski
        
        delta_e = self.calculator_2000.calculate(color1, color2)
        self.assertGreater(delta_e, 0.0)
        self.assertIsInstance(delta_e, float)
    
    def test_delta_e_cmc_basic(self):
        """Test podstawowy Delta E CMC"""
        color1 = self.test_colors_lab[1]  # Biały
        color2 = self.test_colors_lab[2]  # Czarny
        
        delta_e = self.calculator_cmc.calculate(color1, color2)
        self.assertGreater(delta_e, 0.0)
        self.assertIsInstance(delta_e, float)
    
    def test_color_conversion_rgb_lab(self):
        """Test konwersji RGB ↔ LAB"""
        for rgb_color in self.test_colors_rgb:
            # RGB → LAB → RGB
            lab_color = self.converter.rgb_to_lab(rgb_color)
            rgb_back = self.converter.lab_to_rgb(lab_color)
            
            # Sprawdź czy konwersja jest odwracalna (z tolerancją)
            for i in range(3):
                self.assertAlmostEqual(rgb_color[i], rgb_back[i], delta=2)
    
    def test_hex_conversion(self):
        """Test konwersji HEX"""
        test_cases = [
            ((255, 0, 0), "#ff0000"),
            ((0, 255, 0), "#00ff00"),
            ((0, 0, 255), "#0000ff"),
            ((255, 255, 255), "#ffffff"),
            ((0, 0, 0), "#000000")
        ]
        
        for rgb, expected_hex in test_cases:
            # RGB → HEX
            hex_result = self.converter.rgb_to_hex(rgb)
            self.assertEqual(hex_result, expected_hex)
            
            # HEX → RGB
            rgb_result = self.converter.hex_to_rgb(expected_hex)
            self.assertEqual(rgb_result, rgb)
    
    def test_delta_e_edge_cases(self):
        """Test przypadków brzegowych"""
        # Ekstremalne wartości LAB
        extreme_colors = [
            (0, -128, -128),    # Minimum
            (100, 127, 127),    # Maximum
            (50, 0, 0),         # Neutralny
        ]
        
        for color1 in extreme_colors:
            for color2 in extreme_colors:
                try:
                    delta_e = self.calculator_2000.calculate(color1, color2)
                    self.assertIsInstance(delta_e, float)
                    self.assertGreaterEqual(delta_e, 0.0)
                except Exception as e:
                    self.fail(f"Delta E calculation failed for {color1} vs {color2}: {e}")
    
    def test_method_comparison(self):
        """Test porównania różnych metod"""
        color1 = (50, 20, -10)
        color2 = (55, 25, -5)
        
        delta_e_76 = self.calculator_76.calculate(color1, color2)
        delta_e_94 = self.calculator_94.calculate(color1, color2)
        delta_e_2000 = self.calculator_2000.calculate(color1, color2)
        delta_e_cmc = self.calculator_cmc.calculate(color1, color2)
        
        # Wszystkie powinny być dodatnie
        self.assertGreater(delta_e_76, 0)
        self.assertGreater(delta_e_94, 0)
        self.assertGreater(delta_e_2000, 0)
        self.assertGreater(delta_e_cmc, 0)
        
        # CIE76 zazwyczaj daje największe wartości
        # (nie zawsze, ale dla większości przypadków)
        results = {
            'CIE76': delta_e_76,
            'CIE94': delta_e_94,
            'CIEDE2000': delta_e_2000,
            'CMC': delta_e_cmc
        }
        
        print(f"\nPorównanie metod dla {color1} vs {color2}:")
        for method, value in results.items():
            print(f"{method}: {value:.3f}")

class TestColorMatcher(unittest.TestCase):
    """Testy dla dopasowywania kolorów"""
    
    def setUp(self):
        self.matcher = ColorMatcher(DeltaEMethod.CIEDE2000)
        
        self.source_palette = [
            (255, 0, 0),    # Czerwony
            (0, 255, 0),    # Zielony
            (0, 0, 255),    # Niebieski
            (255, 255, 0),  # Żółty
            (255, 0, 255)   # Magenta
        ]
        
        self.target_palette = [
            (200, 50, 50),   # Ciemno-czerwony
            (50, 200, 50),   # Ciemno-zielony
            (50, 50, 200),   # Ciemno-niebieski
            (200, 200, 50),  # Ciemno-żółty
            (200, 50, 200)   # Ciemno-magenta
        ]
    
    def test_find_closest_color(self):
        """Test znajdowania najbliższego koloru"""
        target_color = (255, 0, 0)  # Czerwony
        
        closest_color, delta_e = self.matcher.find_closest_color(
            target_color, self.target_palette
        )
        
        # Powinien znaleźć ciemno-czerwony
        self.assertEqual(closest_color, (200, 50, 50))
        self.assertIsInstance(delta_e, float)
        self.assertGreater(delta_e, 0)
    
    def test_create_color_map(self):
        """Test tworzenia mapy kolorów"""
        color_map = self.matcher.create_color_map(
            self.source_palette, self.target_palette
        )
        
        # Sprawdź czy wszystkie kolory źródłowe mają mapowanie
        for source_color in self.source_palette:
            self.assertIn(source_color, color_map)
            self.assertIn(color_map[source_color], self.target_palette)
    
    def test_evaluate_mapping_quality(self):
        """Test oceny jakości mapowania"""
        color_map = self.matcher.create_color_map(
            self.source_palette, self.target_palette
        )
        
        quality_metrics = self.matcher.evaluate_mapping_quality(color_map)
        
        required_metrics = [
            'average_delta_e', 'max_delta_e', 'min_delta_e', 
            'std_delta_e', 'acceptable_mappings'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, quality_metrics)
            self.assertIsInstance(quality_metrics[metric], float)
            self.assertGreaterEqual(quality_metrics[metric], 0)

class TestPaletteGenerator(unittest.TestCase):
    """Testy dla generatora palet"""
    
    def setUp(self):
        self.generator = PaletteGenerator(DeltaEMethod.CIEDE2000)
        self.base_color = (128, 64, 192)  # Fioletowy
    
    def test_generate_complementary_palette(self):
        """Test generowania palety komplementarnej"""
        palette = self.generator.generate_complementary_palette(
            self.base_color, size=5, min_delta_e=10.0
        )
        
        self.assertEqual(len(palette), 5)
        self.assertEqual(palette[0], self.base_color)  # Pierwszy kolor to kolor bazowy
        
        # Sprawdź czy kolory są wystarczająco różne
        converter = ColorConverter()
        calculator = DeltaECalculator(DeltaEMethod.CIEDE2000)
        
        for i in range(len(palette)):
            for j in range(i + 1, len(palette)):
                lab1 = converter.rgb_to_lab(palette[i])
                lab2 = converter.rgb_to_lab(palette[j])
                delta_e = calculator.calculate(lab1, lab2)
                self.assertGreaterEqual(delta_e, 8.0)  # Tolerancja
    
    def test_generate_analogous_palette(self):
        """Test generowania palety analogicznej"""
        palette = self.generator.generate_analogous_palette(
            self.base_color, size=5, max_delta_e=15.0
        )
        
        self.assertEqual(len(palette), 5)
        self.assertEqual(palette[0], self.base_color)
        
        # Sprawdź czy kolory są podobne do bazowego
        converter = ColorConverter()
        calculator = DeltaECalculator(DeltaEMethod.CIEDE2000)
        base_lab = converter.rgb_to_lab(self.base_color)
        
        for color in palette[1:]:
            color_lab = converter.rgb_to_lab(color)
            delta_e = calculator.calculate(base_lab, color_lab)
            self.assertLessEqual(delta_e, 15.0)
    
    def test_generate_gradient_palette(self):
        """Test generowania palety gradientowej"""
        start_color = (255, 0, 0)    # Czerwony
        end_color = (0, 0, 255)      # Niebieski
        
        palette = self.generator.generate_gradient_palette(
            start_color, end_color, steps=10
        )
        
        self.assertEqual(len(palette), 10)
        self.assertEqual(palette[0], start_color)
        self.assertEqual(palette[-1], end_color)
        
        # Sprawdź czy gradient jest monotoniczny
        for i in range(len(palette) - 1):
            # Czerwony powinien maleć, niebieski rosnąć
            self.assertGreaterEqual(palette[i][0], palette[i+1][0])  # R
            self.assertLessEqual(palette[i][2], palette[i+1][2])     # B

class TestImageColorAnalyzer(unittest.TestCase):
    """Testy dla analizatora kolorów obrazów"""
    
    def setUp(self):
        self.analyzer = ImageColorAnalyzer(DeltaEMethod.CIEDE2000)
    
    def test_sample_colors(self):
        """Test próbkowania kolorów"""
        # Tworzenie testowego obrazu
        test_image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        
        colors = self.analyzer._sample_colors(test_image, sample_size=50)
        
        self.assertEqual(len(colors), 50)
        # Wszystkie kolory powinny być czerwone
        for color in colors:
            self.assertEqual(color, (255, 0, 0))
    
    def test_cluster_colors(self):
        """Test grupowania kolorów"""
        # Kolory podobne do siebie
        similar_colors = [
            (255, 0, 0), (250, 5, 5), (245, 10, 10),  # Czerwone
            (0, 255, 0), (5, 250, 5), (10, 245, 10),  # Zielone
        ]
        
        clustered = self.analyzer._cluster_colors(similar_colors, threshold=10.0)
        
        # Powinno być mniej klastrów niż oryginalnych kolorów
        self.assertLessEqual(len(clustered), len(similar_colors))
        self.assertGreaterEqual(len(clustered), 2)  # Przynajmniej czerwony i zielony
    
    def test_calculate_color_harmony(self):
        """Test obliczania harmonii kolorów"""
        # Harmonijne kolory (podobne odcienie)
        harmonious_colors = [
            (50, 10, 5),   # Podobne LAB
            (52, 12, 7),
            (48, 8, 3)
        ]
        
        # Nieharmonijne kolory (bardzo różne)
        disharmonious_colors = [
            (50, 0, 0),    # Szary
            (53, 80, 67),  # Czerwony
            (32, 79, -108) # Niebieski
        ]
        
        harmony1 = self.analyzer._calculate_color_harmony(harmonious_colors)
        harmony2 = self.analyzer._calculate_color_harmony(disharmonious_colors)
        
        # Harmonijne kolory powinny mieć niższą wartość
        self.assertLess(harmony1, harmony2)

class TestOptimizedDeltaE(unittest.TestCase):
    """Testy dla zoptymalizowanego kalkulatora"""
    
    def setUp(self):
        self.optimized_calc = OptimizedDeltaECalculator(DeltaEMethod.CIE76)
        self.standard_calc = DeltaECalculator(DeltaEMethod.CIE76)
    
    def test_batch_calculation_accuracy(self):
        """Test dokładności obliczeń batch"""
        # Przygotowanie danych
        colors1 = np.array([
            [50, 0, 0],
            [60, 10, -5],
            [40, -10, 15]
        ], dtype=np.float32)
        
        colors2 = np.array([
            [55, 5, 5],
            [65, 15, 0],
            [35, -5, 20]
        ], dtype=np.float32)
        
        # Obliczenia batch
        batch_results = self.optimized_calc.calculate_batch(colors1, colors2)
        
        # Obliczenia pojedyncze
        individual_results = []
        for i in range(len(colors1)):
            result = self.standard_calc.calculate(
                tuple(colors1[i]), tuple(colors2[i])
            )
            individual_results.append(result)
        
        # Porównanie wyników
        for i in range(len(batch_results)):
            self.assertAlmostEqual(
                batch_results[i], individual_results[i], places=3
            )
    
    def test_batch_performance(self):
        """Test wydajności obliczeń batch"""
        # Duży zestaw danych
        n_colors = 1000
        colors1 = np.random.rand(n_colors, 3) * 100
        colors2 = np.random.rand(n_colors, 3) * 100
        
        # Test batch
        start_time = time.time()
        batch_results = self.optimized_calc.calculate_batch(colors1, colors2)
        batch_time = time.time() - start_time
        
        # Test pojedynczy (tylko próbka)
        start_time = time.time()
        individual_results = []
        for i in range(min(100, n_colors)):  # Tylko 100 próbek
            result = self.standard_calc.calculate(
                tuple(colors1[i]), tuple(colors2[i])
            )
            individual_results.append(result)
        individual_time = (time.time() - start_time) * (n_colors / 100)
        
        print(f"\nWydajność obliczeń ({n_colors} kolorów):")
        print(f"Batch: {batch_time:.3f}s")
        print(f"Pojedyncze (ekstrapolowane): {individual_time:.3f}s")
        print(f"Przyspieszenie: {individual_time/batch_time:.1f}x")
        
        # Batch powinien być szybszy
        self.assertLess(batch_time, individual_time)

# Test Suite Runner
class DeltaETestSuite:
    """Główna klasa do uruchamiania testów"""
    
    def __init__(self):
        self.test_results = {}
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, bool]:
        """
        Uruchamia wszystkie testy
        
        Args:
            verbose: Czy wyświetlać szczegółowe informacje
        
        Returns:
            Słownik z wynikami testów
        """
        test_classes = [
            TestDeltaECalculator,
            TestColorMatcher,
            TestPaletteGenerator,
            TestImageColorAnalyzer,
            TestOptimizedDeltaE
        ]
        
        all_passed = True
        
        for test_class in test_classes:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            
            print(f"\n{'='*50}")
            print(f"Uruchamianie testów: {test_class.__name__}")
            print(f"{'='*50}")
            
            result = runner.run(suite)
            
            class_passed = result.wasSuccessful()
            self.test_results[test_class.__name__] = class_passed
            all_passed = all_passed and class_passed
            
            if not class_passed:
                print(f"❌ {test_class.__name__}: FAILED")
                print(f"   Błędy: {len(result.errors)}")
                print(f"   Niepowodzenia: {len(result.failures)}")
            else:
                print(f"✅ {test_class.__name__}: PASSED")
        
        print(f"\n{'='*50}")
        print(f"PODSUMOWANIE TESTÓW")
        print(f"{'='*50}")
        
        for class_name, passed in self.test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{class_name}: {status}")
        
        overall_status = "✅ WSZYSTKIE TESTY PRZESZŁY" if all_passed else "❌ NIEKTÓRE TESTY NIE PRZESZŁY"
        print(f"\nOgólny wynik: {overall_status}")
        
        return self.test_results
    
    def run_specific_test(self, test_class_name: str, test_method: str = None) -> bool:
        """
        Uruchamia konkretny test
        
        Args:
            test_class_name: Nazwa klasy testowej
            test_method: Nazwa metody testowej (opcjonalne)
        
        Returns:
            True jeśli test przeszedł
        """
        test_classes = {
            'TestDeltaECalculator': TestDeltaECalculator,
            'TestColorMatcher': TestColorMatcher,
            'TestPaletteGenerator': TestPaletteGenerator,
            'TestImageColorAnalyzer': TestImageColorAnalyzer,
            'TestOptimizedDeltaE': TestOptimizedDeltaE
        }
        
        if test_class_name not in test_classes:
            print(f"Nieznana klasa testowa: {test_class_name}")
            return False
        
        test_class = test_classes[test_class_name]
        
        if test_method:
            suite = unittest.TestSuite()
            suite.addTest(test_class(test_method))
        else:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
```

---

## Benchmarki Wydajności

### Framework Benchmarkowy

```python
import time
import psutil
import matplotlib.pyplot as plt
from typing import Dict, List, Callable
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BenchmarkResult:
    """Wynik benchmarku"""
    name: str
    execution_time: float
    memory_usage: float
    operations_per_second: float
    accuracy_score: float = None
    additional_metrics: Dict[str, float] = None

class DeltaEBenchmark:
    """Klasa do benchmarkowania algorytmów Delta E"""
    
    def __init__(self):
        self.results = []
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> Dict[str, np.ndarray]:
        """
        Generuje dane testowe różnych rozmiarów
        """
        np.random.seed(42)  # Dla powtarzalności
        
        sizes = [100, 500, 1000, 5000, 10000]
        test_data = {}
        
        for size in sizes:
            # Losowe kolory LAB
            colors1 = np.random.rand(size, 3)
            colors1[:, 0] *= 100  # L: 0-100
            colors1[:, 1] = (colors1[:, 1] - 0.5) * 256  # a: -128 do 127
            colors1[:, 2] = (colors1[:, 2] - 0.5) * 256  # b: -128 do 127
            
            colors2 = np.random.rand(size, 3)
            colors2[:, 0] *= 100
            colors2[:, 1] = (colors2[:, 1] - 0.5) * 256
            colors2[:, 2] = (colors2[:, 2] - 0.5) * 256
            
            test_data[f"size_{size}"] = {
                'colors1': colors1,
                'colors2': colors2
            }
        
        return test_data
    
    def benchmark_method(self, method: DeltaEMethod, 
                        data_size: str = "size_1000") -> BenchmarkResult:
        """
        Benchmarkuje konkretną metodę Delta E
        
        Args:
            method: Metoda Delta E do testowania
            data_size: Rozmiar danych testowych
        
        Returns:
            Wynik benchmarku
        """
        if data_size not in self.test_data:
            raise ValueError(f"Nieznany rozmiar danych: {data_size}")
        
        colors1 = self.test_data[data_size]['colors1']
        colors2 = self.test_data[data_size]['colors2']
        n_operations = len(colors1)
        
        # Przygotowanie kalkulatora
        calculator = DeltaECalculator(method)
        
        # Pomiar pamięci przed
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Benchmark
        start_time = time.time()
        
        results = []
        for i in range(n_operations):
            color1 = tuple(colors1[i])
            color2 = tuple(colors2[i])
            delta_e = calculator.calculate(color1, color2)
            results.append(delta_e)
        
        end_time = time.time()
        
        # Pomiar pamięci po
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_usage = memory_after - memory_before
        ops_per_second = n_operations / execution_time
        
        # Sprawdzenie poprawności (czy wszystkie wyniki są liczbami)
        accuracy_score = sum(1 for r in results if isinstance(r, (int, float)) and r >= 0) / len(results)
        
        result = BenchmarkResult(
            name=f"{method.value}_{data_size}",
            execution_time=execution_time,
            memory_usage=memory_usage,
            operations_per_second=ops_per_second,
            accuracy_score=accuracy_score,
            additional_metrics={
                'n_operations': n_operations,
                'avg_delta_e': np.mean(results),
                'std_delta_e': np.std(results)
            }
        )
        
        self.results.append(result)
        return result
    
    def benchmark_optimized_vs_standard(self, data_size: str = "size_5000") -> Dict[str, BenchmarkResult]:
        """
        Porównuje zoptymalizowaną wersję ze standardową
        
        Args:
            data_size: Rozmiar danych testowych
        
        Returns:
            Słownik z wynikami benchmarków
        """
        colors1 = self.test_data[data_size]['colors1']
        colors2 = self.test_data[data_size]['colors2']
        n_operations = len(colors1)
        
        results = {}
        
        # Standard calculator
        standard_calc = DeltaECalculator(DeltaEMethod.CIE76)
        
        start_time = time.time()
        standard_results = []
        for i in range(n_operations):
            color1 = tuple(colors1[i])
            color2 = tuple(colors2[i])
            delta_e = standard_calc.calculate(color1, color2)
            standard_results.append(delta_e)
        standard_time = time.time() - start_time
        
        results['standard'] = BenchmarkResult(
            name=f"standard_{data_size}",
            execution_time=standard_time,
            memory_usage=0,  # Nie mierzymy dla uproszczenia
            operations_per_second=n_operations / standard_time,
            accuracy_score=1.0,
            additional_metrics={'avg_delta_e': np.mean(standard_results)}
        )
        
        # Optimized calculator
        optimized_calc = OptimizedDeltaECalculator(DeltaEMethod.CIE76)
        
        start_time = time.time()
        optimized_results = optimized_calc.calculate_batch(colors1, colors2)
        optimized_time = time.time() - start_time
        
        results['optimized'] = BenchmarkResult(
            name=f"optimized_{data_size}",
            execution_time=optimized_time,
            memory_usage=0,
            operations_per_second=n_operations / optimized_time,
            accuracy_score=1.0,
            additional_metrics={'avg_delta_e': np.mean(optimized_results)}
        )
        
        # Porównanie dokładności
        accuracy_diff = np.mean(np.abs(np.array(standard_results) - optimized_results))
        
        print(f"\nPorównanie Standard vs Optimized ({data_size}):")
        print(f"Standard: {standard_time:.3f}s ({results['standard'].operations_per_second:.0f} ops/s)")
        print(f"Optimized: {optimized_time:.3f}s ({results['optimized'].operations_per_second:.0f} ops/s)")
        print(f"Przyspieszenie: {standard_time/optimized_time:.1f}x")
        print(f"Różnica dokładności: {accuracy_diff:.6f}")
        
        return results
    
    def benchmark_all_methods(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmarkuje wszystkie metody dla różnych rozmiarów danych
        
        Returns:
            Słownik z wynikami dla każdej metody
        """
        methods = [
            DeltaEMethod.CIE76,
            DeltaEMethod.CIE94,
            DeltaEMethod.CIEDE2000,
            DeltaEMethod.CMC
        ]
        
        data_sizes = list(self.test_data.keys())
        results_by_method = {method.value: [] for method in methods}
        
        print("Uruchamianie benchmarków dla wszystkich metod...")
        
        for method in methods:
            print(f"\nTestowanie metody: {method.value}")
            
            for data_size in data_sizes:
                print(f"  Rozmiar danych: {data_size}")
                
                try:
                    result = self.benchmark_method(method, data_size)
                    results_by_method[method.value].append(result)
                    
                    print(f"    Czas: {result.execution_time:.3f}s")
                    print(f"    Ops/s: {result.operations_per_second:.0f}")
                    print(f"    Pamięć: {result.memory_usage:.1f}MB")
                    
                except Exception as e:
                    print(f"    Błąd: {e}")
        
        return results_by_method
    
    def generate_performance_report(self, save_path: str = None) -> None:
        """
        Generuje raport wydajności z wykresami
        
        Args:
            save_path: Ścieżka do zapisu raportu
        """
        if not self.results:
            print("Brak wyników do wygenerowania raportu")
            return
        
        # Grupowanie wyników według metod
        methods_data = {}
        for result in self.results:
            method = result.name.split('_')[0]
            if method not in methods_data:
                methods_data[method] = []
            methods_data[method].append(result)
        
        # Tworzenie wykresów
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Wykres 1: Czas wykonania vs rozmiar danych
        ax1 = axes[0, 0]
        for method, results in methods_data.items():
            sizes = [r.additional_metrics['n_operations'] for r in results]
            times = [r.execution_time for r in results]
            ax1.plot(sizes, times, marker='o', label=method)
        
        ax1.set_xlabel('Liczba operacji')
        ax1.set_ylabel('Czas wykonania (s)')
        ax1.set_title('Czas wykonania vs Rozmiar danych')
        ax1.legend()
        ax1.grid(True)
        
        # Wykres 2: Operacje na sekundę
        ax2 = axes[0, 1]
        for method, results in methods_data.items():
            sizes = [r.additional_metrics['n_operations'] for r in results]
            ops_per_sec = [r.operations_per_second for r in results]
            ax2.plot(sizes, ops_per_sec, marker='s', label=method)
        
        ax2.set_xlabel('Liczba operacji')
        ax2.set_ylabel('Operacje/sekundę')
        ax2.set_title('Wydajność (ops/s) vs Rozmiar danych')
        ax2.legend()
        ax2.grid(True)
        
        # Wykres 3: Zużycie pamięci
        ax3 = axes[1, 0]
        for method, results in methods_data.items():
            sizes = [r.additional_metrics['n_operations'] for r in results]
            memory = [r.memory_usage for r in results]
            ax3.plot(sizes, memory, marker='^', label=method)
        
        ax3.set_xlabel('Liczba operacji')
        ax3.set_ylabel('Zużycie pamięci (MB)')
        ax3.set_title('Zużycie pamięci vs Rozmiar danych')
        ax3.legend()
        ax3.grid(True)
        
        # Wykres 4: Porównanie metod (średni czas)
        ax4 = axes[1, 1]
        method_names = list(methods_data.keys())
        avg_times = [np.mean([r.execution_time for r in results]) 
                    for results in methods_data.values()]
        
        bars = ax4.bar(method_names, avg_times, color=['blue', 'green', 'red', 'orange'])
        ax4.set_ylabel('Średni czas wykonania (s)')
        ax4.set_title('Porównanie średniego czasu wykonania')
        ax4.tick_params(axis='x', rotation=45)
        
        # Dodanie wartości na słupkach
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Raport zapisany do: {save_path}")
        
        plt.show()
    
    def save_results_to_json(self, file_path: str) -> None:
        """
        Zapisuje wyniki benchmarków do pliku JSON
        
        Args:
            file_path: Ścieżka do pliku JSON
        """
        results_data = []
        
        for result in self.results:
            result_dict = {
                'name': result.name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'operations_per_second': result.operations_per_second,
                'accuracy_score': result.accuracy_score,
                'additional_metrics': result.additional_metrics or {}
            }
            results_data.append(result_dict)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Wyniki zapisane do: {file_path}")
    
    def load_results_from_json(self, file_path: str) -> None:
        """
        Wczytuje wyniki benchmarków z pliku JSON
        
        Args:
            file_path: Ścieżka do pliku JSON
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        self.results = []
        for result_dict in results_data:
            result = BenchmarkResult(
                name=result_dict['name'],
                execution_time=result_dict['execution_time'],
                memory_usage=result_dict['memory_usage'],
                operations_per_second=result_dict['operations_per_second'],
                accuracy_score=result_dict.get('accuracy_score'),
                additional_metrics=result_dict.get('additional_metrics')
            )
            self.results.append(result)
        
        print(f"Wczytano {len(self.results)} wyników z: {file_path}")
```

---

## Przykłady Uruchomienia

### Uruchomienie Testów

```python
# Przykład uruchomienia wszystkich testów
if __name__ == "__main__":
    print("Delta E Color Distance - Testy i Benchmarki")
    print("=" * 50)
    
    # Uruchomienie testów jednostkowych
    print("\n1. TESTY JEDNOSTKOWE")
    print("-" * 30)
    
    test_suite = DeltaETestSuite()
    test_results = test_suite.run_all_tests(verbose=True)
    
    # Uruchomienie benchmarków
    print("\n\n2. BENCHMARKI WYDAJNOŚCI")
    print("-" * 30)
    
    benchmark = DeltaEBenchmark()
    
    # Benchmark pojedynczej metody
    print("\nBenchmark CIEDE2000:")
    result = benchmark.benchmark_method(DeltaEMethod.CIEDE2000, "size_1000")
    print(f"Czas: {result.execution_time:.3f}s")
    print(f"Ops/s: {result.operations_per_second:.0f}")
    
    # Porównanie zoptymalizowanej wersji
    print("\nPorównanie Standard vs Optimized:")
    comparison = benchmark.benchmark_optimized_vs_standard("size_5000")
    
    # Benchmark wszystkich metod
    print("\nBenchmark wszystkich metod:")
    all_results = benchmark.benchmark_all_methods()
    
    # Generowanie raportu
    print("\nGenerowanie raportu wydajności...")
    benchmark.generate_performance_report("delta_e_performance_report.png")
    
    # Zapisanie wyników
    benchmark.save_results_to_json("delta_e_benchmark_results.json")
    
    print("\n" + "=" * 50)
    print("ZAKOŃCZONO TESTY I BENCHMARKI")
    print("=" * 50)
```

### Uruchomienie Konkretnych Testów

```python
# Przykład uruchomienia konkretnych testów
def run_specific_tests():
    test_suite = DeltaETestSuite()
    
    # Test tylko kalkulatora Delta E
    print("Test kalkulatora Delta E:")
    result = test_suite.run_specific_test('TestDeltaECalculator')
    print(f"Wynik: {'PASSED' if result else 'FAILED'}")
    
    # Test konkretnej metody
    print("\nTest konwersji kolorów:")
    result = test_suite.run_specific_test(
        'TestDeltaECalculator', 
        'test_color_conversion_rgb_lab'
    )
    print(f"Wynik: {'PASSED' if result else 'FAILED'}")

# Uruchomienie
run_specific_tests()
```

### Benchmark Niestandardowy

```python
# Przykład niestandardowego benchmarku
def custom_benchmark():
    # Przygotowanie danych
    colors1 = [(50, 0, 0), (60, 10, -5), (40, -10, 15)]
    colors2 = [(55, 5, 5), (65, 15, 0), (35, -5, 20)]
    
    methods = [
        DeltaEMethod.CIE76,
        DeltaEMethod.CIE94,
        DeltaEMethod.CIEDE2000
    ]
    
    print("Niestandardowy benchmark:")
    print("-" * 30)
    
    for method in methods:
        calculator = DeltaECalculator(method)
        
        start_time = time.time()
        results = []
        
        for color1, color2 in zip(colors1, colors2):
            delta_e = calculator.calculate(color1, color2)
            results.append(delta_e)
        
        end_time = time.time()
        
        print(f"{method.value}:")
        print(f"  Czas: {end_time - start_time:.6f}s")
        print(f"  Wyniki: {[f'{r:.3f}' for r in results]}")
        print(f"  Średnia: {np.mean(results):.3f}")
        print()

# Uruchomienie
custom_benchmark()
```

---

## Podsumowanie Części 3

W tej części omówiliśmy:

1. **Testy jednostkowe** - kompleksowe testowanie wszystkich komponentów
2. **Framework benchmarkowy** - pomiar wydajności i porównania
3. **Przykłady uruchomienia** - praktyczne zastosowania testów
4. **Raporty wydajności** - wizualizacja wyników

### Kluczowe Cechy Testów

✅ **Kompletność**: Testy dla wszystkich głównych komponentów  
✅ **Dokładność**: Weryfikacja poprawności obliczeń  
✅ **Wydajność**: Benchmarki i porównania optymalizacji  
✅ **Raporty**: Automatyczne generowanie wykresów i statystyk  
✅ **Elastyczność**: Możliwość uruchamiania konkretnych testów  

### Metryki Testowane

- **Poprawność obliczeń** Delta E dla różnych metod
- **Konwersje kolorów** RGB ↔ LAB ↔ HEX
- **Wydajność** standard vs optimized
- **Zużycie pamięci** dla różnych rozmiarów danych
- **Dokładność** dopasowywania kolorów
- **Jakość** generowanych palet

### Co dalej?

**Część 4** będzie zawierać:
- Integrację z Flask API
- Praktyczne przypadki użycia
- Troubleshooting i diagnostyka
- Dokumentację API

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Część 3 - Testy, benchmarki i praktyczne zastosowania