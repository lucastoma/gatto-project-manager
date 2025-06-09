# ACES Color Space Transfer - Część 5of6: Analiza Wydajności i Optymalizacje 🚀

> **Seria:** ACES Color Space Transfer  
> **Część:** 5 z 6 - Analiza Wydajności i Optymalizacje  
> **Wymagania:** [4of6 - Parametry i Konfiguracja](gatto-WORKING-03-algorithms-08-advanced-01-aces-4of6.md)  
> **Następna część:** [6of6 - Aplikacje Praktyczne](gatto-WORKING-03-algorithms-08-advanced-01-aces-6of6.md)

---

## 1. Analiza Złożoności Obliczeniowej

### 1.1 Teoretyczna Analiza Złożoności

```python
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import cProfile
import pstats
from memory_profiler import profile

@dataclass
class ComplexityAnalysis:
    """Analiza złożoności obliczeniowej algorytmów ACES."""
    
    operation_name: str
    time_complexity: str
    space_complexity: str
    actual_time: float
    memory_usage: float
    image_size: Tuple[int, int]
    
    def complexity_ratio(self, other: 'ComplexityAnalysis') -> float:
        """Stosunek złożoności między operacjami."""
        if other.actual_time > 0:
            return self.actual_time / other.actual_time
        return float('inf')

class ACESPerformanceAnalyzer:
    """Analizator wydajności algorytmów ACES."""
    
    def __init__(self):
        self.measurements = []
        self.baseline_measurements = {}
    
    @contextmanager
    def measure_operation(self, operation_name: str, image_shape: Tuple[int, ...]):
        """Context manager do pomiaru wydajności operacji."""
        
        # Pomiar przed operacją
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # Pomiar po operacji
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Zapisanie wyników
            measurement = ComplexityAnalysis(
                operation_name=operation_name,
                time_complexity=self._estimate_time_complexity(operation_name),
                space_complexity=self._estimate_space_complexity(operation_name),
                actual_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                image_size=image_shape[:2]
            )
            
            self.measurements.append(measurement)
    
    def _estimate_time_complexity(self, operation: str) -> str:
        """Oszacowanie teoretycznej złożoności czasowej."""
        complexity_map = {
            'convert_to_aces': 'O(n)',  # n = liczba pikseli
            'convert_from_aces': 'O(n)',
            'analyze_statistics': 'O(n)',
            'calculate_transformation': 'O(1)',  # Niezależne od rozmiaru obrazu
            'apply_transformation': 'O(n)',
            'tone_mapping': 'O(n)',
            'preserve_luminance': 'O(n)',
            'compress_gamut': 'O(n)',
            'histogram_matching': 'O(n log n)',  # Sortowanie histogramów
            'chromatic_adaptation': 'O(n)',
            'statistical_matching': 'O(n)',
            'perceptual_matching': 'O(n²)',  # Analiza lokalnych regionów
            'quality_evaluation': 'O(n)'
        }
        return complexity_map.get(operation, 'O(n)')
    
    def _estimate_space_complexity(self, operation: str) -> str:
        """Oszacowanie teoretycznej złożoności pamięciowej."""
        space_map = {
            'convert_to_aces': 'O(n)',
            'convert_from_aces': 'O(n)',
            'analyze_statistics': 'O(k)',  # k = liczba bins w histogramie
            'calculate_transformation': 'O(1)',
            'apply_transformation': 'O(n)',
            'tone_mapping': 'O(1)',  # In-place możliwe
            'preserve_luminance': 'O(n)',
            'compress_gamut': 'O(1)',  # In-place możliwe
            'histogram_matching': 'O(k)',  # k = liczba bins
            'chromatic_adaptation': 'O(1)',
            'statistical_matching': 'O(1)',
            'perceptual_matching': 'O(n)',  # Dodatkowe bufory
            'quality_evaluation': 'O(n)'
        }
        return space_map.get(operation, 'O(n)')
    
    def analyze_scalability(self, test_sizes: List[Tuple[int, int]]) -> Dict:
        """Analiza skalowalności dla różnych rozmiarów obrazów."""
        
        results = {
            'sizes': test_sizes,
            'operations': {},
            'scaling_factors': {},
            'memory_scaling': {}
        }
        
        # Grupowanie pomiarów według operacji
        operations = {}
        for measurement in self.measurements:
            op_name = measurement.operation_name
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(measurement)
        
        # Analiza każdej operacji
        for op_name, measurements in operations.items():
            # Sortowanie według rozmiaru obrazu
            measurements.sort(key=lambda x: x.image_size[0] * x.image_size[1])
            
            times = [m.actual_time for m in measurements]
            memories = [m.memory_usage for m in measurements]
            sizes = [m.image_size[0] * m.image_size[1] for m in measurements]
            
            results['operations'][op_name] = {
                'times': times,
                'memories': memories,
                'sizes': sizes
            }
            
            # Obliczenie współczynników skalowania
            if len(times) > 1:
                time_scaling = self._calculate_scaling_factor(sizes, times)
                memory_scaling = self._calculate_scaling_factor(sizes, memories)
                
                results['scaling_factors'][op_name] = time_scaling
                results['memory_scaling'][op_name] = memory_scaling
        
        return results
    
    def _calculate_scaling_factor(self, sizes: List[int], values: List[float]) -> float:
        """Obliczenie współczynnika skalowania."""
        if len(sizes) < 2 or len(values) < 2:
            return 1.0
        
        # Regresja liniowa w skali logarytmicznej
        log_sizes = np.log(sizes)
        log_values = np.log(np.maximum(values, 1e-10))  # Unikanie log(0)
        
        # y = ax + b => log(value) = a*log(size) + b
        # a to współczynnik skalowania
        coeffs = np.polyfit(log_sizes, log_values, 1)
        return coeffs[0]  # Współczynnik przy x
    
    def generate_performance_report(self) -> str:
        """Generowanie raportu wydajności."""
        
        if not self.measurements:
            return "Brak pomiarów do analizy."
        
        report = ["# Raport Wydajności ACES Color Transfer\n"]
        
        # Podsumowanie ogólne
        total_time = sum(m.actual_time for m in self.measurements)
        total_memory = max(m.memory_usage for m in self.measurements)
        
        report.append(f"## Podsumowanie Ogólne")
        report.append(f"- Całkowity czas: {total_time:.3f}s")
        report.append(f"- Maksymalne użycie pamięci: {total_memory:.1f}MB")
        report.append(f"- Liczba operacji: {len(self.measurements)}\n")
        
        # Analiza według operacji
        operations = {}
        for m in self.measurements:
            if m.operation_name not in operations:
                operations[m.operation_name] = []
            operations[m.operation_name].append(m)
        
        report.append("## Analiza Według Operacji\n")
        
        for op_name, measurements in operations.items():
            avg_time = np.mean([m.actual_time for m in measurements])
            avg_memory = np.mean([m.memory_usage for m in measurements])
            
            report.append(f"### {op_name}")
            report.append(f"- Średni czas: {avg_time:.4f}s")
            report.append(f"- Średnie użycie pamięci: {avg_memory:.1f}MB")
            report.append(f"- Złożoność czasowa: {measurements[0].time_complexity}")
            report.append(f"- Złożoność pamięciowa: {measurements[0].space_complexity}\n")
        
        # Rekomendacje
        report.append("## Rekomendacje Optymalizacji\n")
        report.extend(self._generate_optimization_recommendations(operations))
        
        return "\n".join(report)
    
    def _generate_optimization_recommendations(self, operations: Dict) -> List[str]:
        """Generowanie rekomendacji optymalizacji."""
        recommendations = []
        
        # Analiza najwolniejszych operacji
        avg_times = {}
        for op_name, measurements in operations.items():
            avg_times[op_name] = np.mean([m.actual_time for m in measurements])
        
        slowest_ops = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)[:3]
        
        recommendations.append("### Najwolniejsze Operacje:")
        for op_name, avg_time in slowest_ops:
            recommendations.append(f"- **{op_name}**: {avg_time:.4f}s")
            recommendations.extend(self._get_specific_recommendations(op_name))
        
        # Analiza użycia pamięci
        memory_usage = {}
        for op_name, measurements in operations.items():
            memory_usage[op_name] = max(m.memory_usage for m in measurements)
        
        memory_intensive = sorted(memory_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        
        recommendations.append("\n### Operacje Intensywne Pamięciowo:")
        for op_name, max_memory in memory_intensive:
            if max_memory > 100:  # MB
                recommendations.append(f"- **{op_name}**: {max_memory:.1f}MB")
                recommendations.append("  - Rozważ przetwarzanie w blokach")
                recommendations.append("  - Użyj memory mapping dla dużych obrazów")
        
        return recommendations
    
    def _get_specific_recommendations(self, operation: str) -> List[str]:
        """Specyficzne rekomendacje dla operacji."""
        recommendations_map = {
            'analyze_statistics': [
                "  - Zmniejsz liczbę bins w histogramach",
                "  - Użyj próbkowania dla dużych obrazów",
                "  - Cache wyników dla podobnych obrazów"
            ],
            'histogram_matching': [
                "  - Użyj interpolacji liniowej zamiast cubic",
                "  - Zmniejsz rozdzielczość histogramów",
                "  - Rozważ aproksymację dla real-time"
            ],
            'perceptual_matching': [
                "  - Zmniejsz rozmiar okna analizy",
                "  - Użyj wieloskalowej analizy",
                "  - Rozważ przejście na statistical_matching"
            ],
            'apply_transformation': [
                "  - Użyj vectoryzacji NumPy",
                "  - Rozważ implementację GPU",
                "  - Optymalizuj dostęp do pamięci"
            ]
        }
        
        return recommendations_map.get(operation, ["  - Brak specyficznych rekomendacji"])
```

---

## 2. Benchmarki Wydajności

### 2.1 System Benchmarkingu

```python
class ACESBenchmarkSuite:
    """Kompleksowy system benchmarkingu ACES."""
    
    def __init__(self):
        self.analyzer = ACESPerformanceAnalyzer()
        self.test_images = {}
        self.benchmark_results = {}
    
    def prepare_test_images(self) -> None:
        """Przygotowanie obrazów testowych o różnych rozmiarach."""
        
        test_sizes = [
            (256, 256),    # Mały
            (512, 512),    # Średni
            (1024, 1024),  # Duży
            (2048, 2048),  # Bardzo duży
            (4096, 4096),  # Ultra duży
            (1920, 1080),  # Full HD
            (3840, 2160),  # 4K
        ]
        
        for size in test_sizes:
            # Generowanie syntetycznych obrazów testowych
            self.test_images[size] = self._generate_test_image(size)
    
    def _generate_test_image(self, size: Tuple[int, int]) -> np.ndarray:
        """Generowanie obrazu testowego."""
        height, width = size
        
        # Gradient z szumem dla realistycznego testu
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Kanał czerwony - gradient poziomy
        red = X + 0.1 * np.random.random((height, width))
        
        # Kanał zielony - gradient pionowy
        green = Y + 0.1 * np.random.random((height, width))
        
        # Kanał niebieski - wzór radialny
        center_x, center_y = width // 2, height // 2
        blue = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) + 0.1 * np.random.random((height, width))
        
        # Kombinacja kanałów
        image = np.stack([red, green, blue], axis=-1)
        image = np.clip(image, 0, 1)
        
        # Konwersja do uint8
        return (image * 255).astype(np.uint8)
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Uruchomienie kompleksowego benchmarku."""
        
        if not self.test_images:
            self.prepare_test_images()
        
        results = {
            'method_comparison': {},
            'size_scaling': {},
            'parameter_sensitivity': {},
            'memory_analysis': {},
            'parallel_efficiency': {}
        }
        
        # Test różnych metod
        results['method_comparison'] = self._benchmark_methods()
        
        # Test skalowalności
        results['size_scaling'] = self._benchmark_size_scaling()
        
        # Test wrażliwości parametrów
        results['parameter_sensitivity'] = self._benchmark_parameter_sensitivity()
        
        # Analiza pamięci
        results['memory_analysis'] = self._benchmark_memory_usage()
        
        # Efektywność równoległości
        results['parallel_efficiency'] = self._benchmark_parallel_efficiency()
        
        self.benchmark_results = results
        return results
    
    def _benchmark_methods(self) -> Dict:
        """Benchmark różnych metod transformacji."""
        
        methods = [
            TransformMethod.CHROMATIC_ADAPTATION,
            TransformMethod.STATISTICAL_MATCHING,
            TransformMethod.HISTOGRAM_MATCHING,
            TransformMethod.PERCEPTUAL_MATCHING,
            TransformMethod.HYBRID
        ]
        
        test_size = (1024, 1024)
        source_image = self.test_images[test_size]
        target_image = self._generate_test_image(test_size)  # Inny obraz docelowy
        
        method_results = {}
        
        for method in methods:
            # Konfiguracja parametrów
            params = ACESParameters(method=method)
            aces_transfer = ACESColorTransfer(params)
            
            # Pomiar wydajności
            with self.analyzer.measure_operation(f"method_{method.value}", test_size):
                result = aces_transfer.transfer_colors(source_image, target_image)
            
            method_results[method.value] = {
                'success': 'error' not in result,
                'quality': result.get('quality', {}).get('overall_score', 0),
                'processing_time': result.get('metadata', {}).get('processing_time', 0)
            }
        
        return method_results
    
    def _benchmark_size_scaling(self) -> Dict:
        """Benchmark skalowalności względem rozmiaru obrazu."""
        
        scaling_results = {}
        method = TransformMethod.STATISTICAL_MATCHING  # Metoda bazowa
        
        for size in sorted(self.test_images.keys(), key=lambda x: x[0] * x[1]):
            source_image = self.test_images[size]
            target_image = self._generate_test_image(size)
            
            params = ACESParameters(method=method)
            aces_transfer = ACESColorTransfer(params)
            
            # Pomiar dla każdego rozmiaru
            with self.analyzer.measure_operation(f"size_{size[0]}x{size[1]}", size):
                result = aces_transfer.transfer_colors(source_image, target_image)
            
            pixel_count = size[0] * size[1]
            processing_time = result.get('metadata', {}).get('processing_time', 0)
            
            scaling_results[f"{size[0]}x{size[1]}"] = {
                'pixel_count': pixel_count,
                'processing_time': processing_time,
                'time_per_megapixel': processing_time / (pixel_count / 1e6) if pixel_count > 0 else 0
            }
        
        return scaling_results
    
    def _benchmark_parameter_sensitivity(self) -> Dict:
        """Benchmark wrażliwości na parametry."""
        
        test_size = (1024, 1024)
        source_image = self.test_images[test_size]
        target_image = self._generate_test_image(test_size)
        
        # Parametry do testowania
        parameter_tests = {
            'chunk_size': [256, 512, 1024, 2048, 4096],
            'tone_mapping_a': [1.5, 2.0, 2.51, 3.0, 3.5],
            'luminance_weight': [0.5, 0.7, 0.8, 0.9, 0.95],
            'gamut_compression': [0.3, 0.5, 0.7, 0.9, 1.0]
        }
        
        sensitivity_results = {}
        
        for param_name, values in parameter_tests.items():
            param_results = []
            
            for value in values:
                # Tworzenie parametrów z modyfikacją
                params = ACESParameters()
                
                if param_name == 'chunk_size':
                    params.performance.chunk_size = value
                elif param_name == 'tone_mapping_a':
                    params.tone_mapping.aces_a = value
                elif param_name == 'luminance_weight':
                    params.luminance.weight = value
                elif param_name == 'gamut_compression':
                    params.gamut.compression_strength = value
                
                aces_transfer = ACESColorTransfer(params)
                
                # Pomiar
                with self.analyzer.measure_operation(f"param_{param_name}_{value}", test_size):
                    result = aces_transfer.transfer_colors(source_image, target_image)
                
                param_results.append({
                    'value': value,
                    'processing_time': result.get('metadata', {}).get('processing_time', 0),
                    'quality': result.get('quality', {}).get('overall_score', 0)
                })
            
            sensitivity_results[param_name] = param_results
        
        return sensitivity_results
    
    def _benchmark_memory_usage(self) -> Dict:
        """Benchmark użycia pamięci."""
        
        memory_results = {}
        
        for size in self.test_images.keys():
            source_image = self.test_images[size]
            target_image = self._generate_test_image(size)
            
            # Pomiar pamięci przed
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            params = ACESParameters()
            aces_transfer = ACESColorTransfer(params)
            
            # Transfer z pomiarem pamięci
            result = aces_transfer.transfer_colors(source_image, target_image)
            
            # Pomiar pamięci po
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            pixel_count = size[0] * size[1]
            memory_per_megapixel = memory_used / (pixel_count / 1e6) if pixel_count > 0 else 0
            
            memory_results[f"{size[0]}x{size[1]}"] = {
                'memory_used_mb': memory_used,
                'memory_per_megapixel_mb': memory_per_megapixel,
                'theoretical_minimum_mb': pixel_count * 3 * 4 / 1024 / 1024,  # float32 RGB
                'efficiency_ratio': (pixel_count * 3 * 4 / 1024 / 1024) / max(memory_used, 1)
            }
        
        return memory_results
    
    def _benchmark_parallel_efficiency(self) -> Dict:
        """Benchmark efektywności przetwarzania równoległego."""
        
        test_size = (2048, 2048)  # Duży obraz dla widocznego efektu
        source_image = self.test_images[test_size]
        target_image = self._generate_test_image(test_size)
        
        thread_counts = [1, 2, 4, 8, 16]
        parallel_results = {}
        
        for num_threads in thread_counts:
            params = ACESParameters()
            params.performance.num_threads = num_threads
            params.performance.use_parallel = num_threads > 1
            
            aces_transfer = ACESColorTransfer(params)
            
            # Pomiar czasu dla różnej liczby wątków
            times = []
            for _ in range(3):  # 3 próby dla stabilności
                start_time = time.perf_counter()
                result = aces_transfer.transfer_colors(source_image, target_image)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            parallel_results[num_threads] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'speedup': parallel_results.get(1, {}).get('avg_time', avg_time) / avg_time,
                'efficiency': (parallel_results.get(1, {}).get('avg_time', avg_time) / avg_time) / num_threads
            }
        
        return parallel_results
    
    def generate_benchmark_report(self) -> str:
        """Generowanie raportu benchmarku."""
        
        if not self.benchmark_results:
            return "Brak wyników benchmarku. Uruchom run_comprehensive_benchmark() najpierw."
        
        report = ["# Raport Benchmarku ACES Color Transfer\n"]
        
        # Porównanie metod
        report.append("## Porównanie Metod Transformacji\n")
        method_results = self.benchmark_results['method_comparison']
        
        report.append("| Metoda | Czas [s] | Jakość | Status |")
        report.append("|--------|----------|--------|--------|")
        
        for method, results in method_results.items():
            status = "✅" if results['success'] else "❌"
            report.append(
                f"| {method} | {results['processing_time']:.3f} | "
                f"{results['quality']:.3f} | {status} |"
            )
        
        # Skalowalność
        report.append("\n## Analiza Skalowalności\n")
        scaling_results = self.benchmark_results['size_scaling']
        
        report.append("| Rozmiar | Piksele [MP] | Czas [s] | Czas/MP [s] |")
        report.append("|---------|--------------|----------|-------------|")
        
        for size, results in scaling_results.items():
            mp = results['pixel_count'] / 1e6
            report.append(
                f"| {size} | {mp:.1f} | {results['processing_time']:.3f} | "
                f"{results['time_per_megapixel']:.3f} |"
            )
        
        # Użycie pamięci
        report.append("\n## Analiza Pamięci\n")
        memory_results = self.benchmark_results['memory_analysis']
        
        report.append("| Rozmiar | Pamięć [MB] | Pamięć/MP [MB] | Efektywność |")
        report.append("|---------|-------------|----------------|-------------|")
        
        for size, results in memory_results.items():
            report.append(
                f"| {size} | {results['memory_used_mb']:.1f} | "
                f"{results['memory_per_megapixel_mb']:.1f} | "
                f"{results['efficiency_ratio']:.2f} |"
            )
        
        # Efektywność równoległości
        report.append("\n## Efektywność Równoległości\n")
        parallel_results = self.benchmark_results['parallel_efficiency']
        
        report.append("| Wątki | Czas [s] | Przyspieszenie | Efektywność |")
        report.append("|-------|----------|----------------|-------------|")
        
        for threads, results in parallel_results.items():
            report.append(
                f"| {threads} | {results['avg_time']:.3f} | "
                f"{results['speedup']:.2f}x | {results['efficiency']:.2f} |"
            )
        
        return "\n".join(report)
```

---

## 3. Optymalizacje Pamięciowe

### 3.1 Zarządzanie Pamięcią

```python
class ACESMemoryOptimizer:
    """Optymalizator pamięci dla ACES Color Transfer."""
    
    def __init__(self, max_memory_mb: float = 1000):
        self.max_memory_mb = max_memory_mb
        self.memory_pools = {}
        self.temp_arrays = []
    
    def optimize_for_large_images(
        self, 
        aces_transfer: 'ACESColorTransfer',
        source_image: np.ndarray,
        target_image: np.ndarray
    ) -> Dict:
        """Optymalizacja dla dużych obrazów."""
        
        image_size_mb = self._calculate_image_memory(source_image)
        
        if image_size_mb > self.max_memory_mb:
            # Przetwarzanie w blokach
            return self._process_in_chunks(
                aces_transfer, source_image, target_image
            )
        else:
            # Standardowe przetwarzanie z optymalizacjami
            return self._process_with_memory_optimization(
                aces_transfer, source_image, target_image
            )
    
    def _calculate_image_memory(self, image: np.ndarray) -> float:
        """Obliczenie zapotrzebowania na pamięć dla obrazu."""
        # Szacowanie: obraz źródłowy + ACES + wynik + bufory tymczasowe
        base_memory = image.nbytes / 1024 / 1024  # MB
        
        # Współczynnik dla dodatkowych buforów
        memory_factor = 4.0  # Konserwatywne oszacowanie
        
        return base_memory * memory_factor
    
    def _process_in_chunks(
        self,
        aces_transfer: 'ACESColorTransfer',
        source_image: np.ndarray,
        target_image: np.ndarray
    ) -> Dict:
        """Przetwarzanie obrazu w blokach."""
        
        height, width = source_image.shape[:2]
        
        # Obliczenie optymalnego rozmiaru bloku
        chunk_size = self._calculate_optimal_chunk_size(
            source_image, self.max_memory_mb
        )
        
        # Analiza statystyk dla całego obrazu (próbkowanie)
        source_sample = self._create_representative_sample(source_image)
        target_sample = self._create_representative_sample(target_image)
        
        # Obliczenie transformacji na podstawie próbek
        sample_aces_source = aces_transfer._convert_to_aces(source_sample)
        sample_aces_target = aces_transfer._convert_to_aces(target_sample)
        
        source_stats = aces_transfer._analyze_statistics(sample_aces_source)
        target_stats = aces_transfer._analyze_statistics(sample_aces_target)
        
        transform_data = aces_transfer._calculate_transformation(
            source_stats, target_stats
        )
        
        # Przetwarzanie w blokach
        result_image = np.zeros_like(source_image)
        
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                # Wyznaczenie granic bloku
                y_end = min(y + chunk_size, height)
                x_end = min(x + chunk_size, width)
                
                # Ekstrakcja bloku
                source_chunk = source_image[y:y_end, x:x_end]
                
                # Przetworzenie bloku
                chunk_result = self._process_single_chunk(
                    aces_transfer, source_chunk, transform_data
                )
                
                # Zapisanie wyniku
                result_image[y:y_end, x:x_end] = chunk_result
                
                # Zwolnienie pamięci
                del source_chunk, chunk_result
                self._cleanup_memory()
        
        return {
            'image': result_image,
            'metadata': {
                'processing_method': 'chunked',
                'chunk_size': chunk_size,
                'transform_data': transform_data
            }
        }
    
    def _calculate_optimal_chunk_size(
        self, 
        image: np.ndarray, 
        max_memory_mb: float
    ) -> int:
        """Obliczenie optymalnego rozmiaru bloku."""
        
        # Pamięć na piksel (w bajtach)
        bytes_per_pixel = image.dtype.itemsize * image.shape[2]
        
        # Dostępna pamięć na blok (z marginesem bezpieczeństwa)
        available_memory_bytes = max_memory_mb * 1024 * 1024 * 0.5
        
        # Maksymalna liczba pikseli w bloku
        max_pixels_per_chunk = available_memory_bytes // (bytes_per_pixel * 4)  # 4x bufor
        
        # Kwadratowy blok
        chunk_size = int(np.sqrt(max_pixels_per_chunk))
        
        # Ograniczenia praktyczne
        chunk_size = max(64, min(chunk_size, 2048))
        
        return chunk_size
    
    def _create_representative_sample(
        self, 
        image: np.ndarray, 
        sample_size: int = 10000
    ) -> np.ndarray:
        """Tworzenie reprezentatywnej próbki obrazu."""
        
        height, width = image.shape[:2]
        total_pixels = height * width
        
        if total_pixels <= sample_size:
            return image
        
        # Stratyfikowane próbkowanie
        step = int(np.sqrt(total_pixels / sample_size))
        
        # Indeksy próbkowania
        y_indices = np.arange(0, height, step)
        x_indices = np.arange(0, width, step)
        
        # Tworzenie siatki próbkowania
        sample_pixels = []
        for y in y_indices:
            for x in x_indices:
                if len(sample_pixels) < sample_size:
                    sample_pixels.append(image[y, x])
        
        # Konwersja do obrazu próbki
        sample_array = np.array(sample_pixels)
        sample_height = int(np.sqrt(len(sample_pixels)))
        sample_width = len(sample_pixels) // sample_height
        
        return sample_array[:sample_height * sample_width].reshape(
            sample_height, sample_width, image.shape[2]
        )
    
    def _process_single_chunk(
        self,
        aces_transfer: 'ACESColorTransfer',
        chunk: np.ndarray,
        transform_data: Dict
    ) -> np.ndarray:
        """Przetworzenie pojedynczego bloku."""
        
        # Konwersja do ACES
        chunk_aces = aces_transfer._convert_to_aces(chunk)
        
        # Aplikacja transformacji
        transformed_aces = aces_transfer._apply_transformation(
            chunk_aces, transform_data
        )
        
        # Post-processing
        if aces_transfer.params.use_tone_mapping:
            transformed_aces = aces_transfer._apply_tone_mapping(transformed_aces)
        
        if aces_transfer.params.gamut_compression:
            transformed_aces = aces_transfer._compress_gamut(transformed_aces)
        
        # Konwersja z powrotem
        result_chunk = aces_transfer._convert_from_aces(transformed_aces)
        
        return result_chunk
    
    def _cleanup_memory(self) -> None:
        """Czyszczenie pamięci."""
        import gc
        gc.collect()
    
    def _process_with_memory_optimization(
        self,
        aces_transfer: 'ACESColorTransfer',
        source_image: np.ndarray,
        target_image: np.ndarray
    ) -> Dict:
        """Przetwarzanie z optymalizacjami pamięciowymi."""
        
        # Użycie in-place operacji gdzie to możliwe
        original_params = aces_transfer.params
        
        # Optymalizacja parametrów dla pamięci
        optimized_params = self._optimize_parameters_for_memory(original_params)
        aces_transfer.params = optimized_params
        
        try:
            # Standardowe przetwarzanie
            result = aces_transfer.transfer_colors(source_image, target_image)
            
            # Dodanie informacji o optymalizacji
            result['metadata']['memory_optimized'] = True
            result['metadata']['optimization_method'] = 'in_place'
            
            return result
            
        finally:
            # Przywrócenie oryginalnych parametrów
            aces_transfer.params = original_params
    
    def _optimize_parameters_for_memory(self, params: ACESParameters) -> ACESParameters:
        """Optymalizacja parametrów dla oszczędności pamięci."""
        
        optimized = params
        
        # Zmniejszenie chunk_size dla lepszej lokalności pamięci
        optimized.performance.chunk_size = min(
            optimized.performance.chunk_size, 512
        )
        
        # Wyłączenie cache'owania jeśli włączone
        optimized.performance.enable_caching = False
        
        # Uproszczenie niektórych operacji
        if optimized.luminance.method == "histogram_match":
            optimized.luminance.histogram_bins = min(
                optimized.luminance.histogram_bins, 256
            )
        
        return optimized
```

---

## 4. Optymalizacje GPU

### 4.1 Implementacja CUDA/OpenCL

```python
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

class ACESGPUOptimizer:
    """Optymalizator GPU dla ACES Color Transfer."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cuda_available = CUDA_AVAILABLE
        
        if self.cuda_available:
            cp.cuda.Device(device_id).use()
            self._initialize_gpu_kernels()
    
    def _initialize_gpu_kernels(self) -> None:
        """Inicjalizacja kerneli GPU."""
        
        # Kernel dla konwersji sRGB -> ACES
        self.srgb_to_aces_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void srgb_to_aces_kernel(
            const float* input, 
            float* output, 
            const float* matrix,
            int width, 
            int height
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_pixels = width * height;
            
            if (idx < total_pixels) {
                int pixel_idx = idx * 3;
                
                // Usunięcie gamma (sRGB -> linear)
                float r = input[pixel_idx];
                float g = input[pixel_idx + 1];
                float b = input[pixel_idx + 2];
                
                // sRGB gamma removal
                r = (r <= 0.04045f) ? r / 12.92f : powf((r + 0.055f) / 1.055f, 2.4f);
                g = (g <= 0.04045f) ? g / 12.92f : powf((g + 0.055f) / 1.055f, 2.4f);
                b = (b <= 0.04045f) ? b / 12.92f : powf((b + 0.055f) / 1.055f, 2.4f);
                
                // Transformacja macierzowa sRGB -> ACES AP0
                output[pixel_idx] = matrix[0] * r + matrix[1] * g + matrix[2] * b;
                output[pixel_idx + 1] = matrix[3] * r + matrix[4] * g + matrix[5] * b;
                output[pixel_idx + 2] = matrix[6] * r + matrix[7] * g + matrix[8] * b;
                
                // Clamp do zakresu ACES
                output[pixel_idx] = fmaxf(0.0f, fminf(65504.0f, output[pixel_idx]));
                output[pixel_idx + 1] = fmaxf(0.0f, fminf(65504.0f, output[pixel_idx + 1]));
                output[pixel_idx + 2] = fmaxf(0.0f, fminf(65504.0f, output[pixel_idx + 2]));
            }
        }
        ''', 'srgb_to_aces_kernel')
        
        # Kernel dla tone mappingu ACES RRT
        self.tone_mapping_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void aces_rrt_kernel(
            const float* input,
            float* output,
            float a, float b, float c, float d, float e,
            int width, int height
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_pixels = width * height;
            
            if (idx < total_pixels) {
                int pixel_idx = idx * 3;
                
                for (int ch = 0; ch < 3; ch++) {
                    float x = input[pixel_idx + ch];
                    
                    // ACES RRT formula
                    float numerator = x * (a * x + b);
                    float denominator = x * (c * x + d) + e;
                    
                    float result = numerator / fmaxf(denominator, 1e-10f);
                    output[pixel_idx + ch] = fmaxf(0.0f, fminf(1.0f, result));
                }
            }
        }
        ''', 'aces_rrt_kernel')
        
        # Kernel dla transformacji statystycznej
        self.statistical_transform_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void statistical_transform_kernel(
            const float* input,
            float* output,
            const float* source_mean,
            const float* target_mean,
            const float* std_ratio,
            int width, int height
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_pixels = width * height;
            
            if (idx < total_pixels) {
                int pixel_idx = idx * 3;
                
                for (int ch = 0; ch < 3; ch++) {
                    float x = input[pixel_idx + ch];
                    
                    // T(x) = (x - μ_src) * σ_ratio + μ_tgt
                    float centered = x - source_mean[ch];
                    float scaled = centered * std_ratio[ch];
                    float result = scaled + target_mean[ch];
                    
                    output[pixel_idx + ch] = fmaxf(0.0f, fminf(65504.0f, result));
                }
            }
        }
        ''', 'statistical_transform_kernel')
    
    def gpu_transfer_colors(
        self,
        aces_transfer: 'ACESColorTransfer',
        source_image: np.ndarray,
        target_image: np.ndarray
    ) -> Dict:
        """Transfer kolorów z wykorzystaniem GPU."""
        
        if not self.cuda_available:
            raise RuntimeError("CUDA nie jest dostępne")
        
        # Transfer danych na GPU
        source_gpu = cp.asarray(source_image, dtype=cp.float32) / 255.0
        target_gpu = cp.asarray(target_image, dtype=cp.float32) / 255.0
        
        try:
            # Konwersja do ACES na GPU
            source_aces_gpu = self._gpu_convert_to_aces(source_gpu)
            target_aces_gpu = self._gpu_convert_to_aces(target_gpu)
            
            # Analiza statystyk na GPU
            source_stats = self._gpu_analyze_statistics(source_aces_gpu)
            target_stats = self._gpu_analyze_statistics(target_aces_gpu)
            
            # Obliczenie transformacji (CPU)
            transform_data = aces_transfer._calculate_transformation(
                source_stats, target_stats
            )
            
            # Aplikacja transformacji na GPU
            result_aces_gpu = self._gpu_apply_transformation(
                source_aces_gpu, transform_data
            )
            
            # Post-processing na GPU
            if aces_transfer.params.use_tone_mapping:
                result_aces_gpu = self._gpu_tone_mapping(
                    result_aces_gpu, aces_transfer.params.tone_mapping
                )
            
            # Konwersja z powrotem na GPU
            result_gpu = self._gpu_convert_from_aces(result_aces_gpu)
            
            # Transfer wyniku z powrotem na CPU
            result_image = cp.asnumpy(result_gpu * 255.0).astype(np.uint8)
            
            return {
                'image': result_image,
                'metadata': {
                    'processing_method': 'gpu',
                    'device_id': self.device_id
                }
            }
            
        finally:
            # Czyszczenie pamięci GPU
            cp.get_default_memory_pool().free_all_blocks()
    
    def _gpu_convert_to_aces(self, image_gpu: cp.ndarray) -> cp.ndarray:
        """Konwersja sRGB -> ACES na GPU."""
        
        height, width, channels = image_gpu.shape
        total_pixels = height * width
        
        # Macierz transformacji na GPU
        matrix_gpu = cp.array([
            0.4397010, 0.3829780, 0.1773350,
            0.0897923, 0.8134230, 0.0967616,
            0.0175439, 0.1115440, 0.8707040
        ], dtype=cp.float32)
        
        # Przygotowanie buforów
        input_flat = image_gpu.reshape(-1)
        output_flat = cp.zeros_like(input_flat)
        
        # Konfiguracja kernela
        block_size = 256
        grid_size = (total_pixels + block_size - 1) // block_size
        
        # Uruchomienie kernela
        self.srgb_to_aces_kernel(
            (grid_size,), (block_size,),
            (input_flat, output_flat, matrix_gpu, width, height)
        )
        
        return output_flat.reshape(height, width, channels)
    
    def _gpu_analyze_statistics(self, aces_image_gpu: cp.ndarray) -> Dict:
        """Analiza statystyk na GPU."""
        
        # Reshape do 2D
        pixels_gpu = aces_image_gpu.reshape(-1, 3)
        
        # Podstawowe statystyki
        stats = {
            'mean': cp.asnumpy(cp.mean(pixels_gpu, axis=0)),
            'std': cp.asnumpy(cp.std(pixels_gpu, axis=0)),
            'min': cp.asnumpy(cp.min(pixels_gpu, axis=0)),
            'max': cp.asnumpy(cp.max(pixels_gpu, axis=0))
        }
        
        # Percentyle
        percentiles = {}
        for p in [1, 5, 25, 50, 75, 95, 99]:
            percentiles[f'p{p}'] = cp.asnumpy(
                cp.percentile(pixels_gpu, p, axis=0)
            )
        stats['percentiles'] = percentiles
        
        # Luminancja ACES
        luminance_gpu = (
            0.2722287168 * pixels_gpu[:, 0] + 
            0.6740817658 * pixels_gpu[:, 1] + 
            0.0536895174 * pixels_gpu[:, 2]
        )
        
        stats['luminance'] = {
            'mean': cp.asnumpy(cp.mean(luminance_gpu)),
            'std': cp.asnumpy(cp.std(luminance_gpu)),
            'min': cp.asnumpy(cp.min(luminance_gpu)),
            'max': cp.asnumpy(cp.max(luminance_gpu))
        }
        
        return stats
    
    def _gpu_apply_transformation(
        self, 
        source_aces_gpu: cp.ndarray, 
        transform_data: Dict
    ) -> cp.ndarray:
        """Aplikacja transformacji na GPU."""
        
        height, width, channels = source_aces_gpu.shape
        total_pixels = height * width
        
        # Przygotowanie buforów
        input_flat = source_aces_gpu.reshape(-1)
        output_flat = cp.zeros_like(input_flat)
        
        if transform_data['method'] == 'statistical':
            # Parametry na GPU
            source_mean_gpu = cp.array(transform_data['source_mean'], dtype=cp.float32)
            target_mean_gpu = cp.array(transform_data['target_mean'], dtype=cp.float32)
            std_ratio_gpu = cp.array(transform_data['std_ratio'], dtype=cp.float32)
            
            # Konfiguracja kernela
            block_size = 256
            grid_size = (total_pixels + block_size - 1) // block_size
            
            # Uruchomienie kernela
            self.statistical_transform_kernel(
                (grid_size,), (block_size,),
                (input_flat, output_flat, source_mean_gpu, 
                 target_mean_gpu, std_ratio_gpu, width, height)
            )
            
        else:
            # Fallback na CPU dla innych metod
            source_aces_cpu = cp.asnumpy(source_aces_gpu)
            # ... implementacja CPU ...
            output_flat = cp.asarray(source_aces_cpu).reshape(-1)
        
        return output_flat.reshape(height, width, channels)
    
    def _gpu_tone_mapping(
        self, 
        aces_image_gpu: cp.ndarray, 
        tone_params: 'ToneMappingParameters'
    ) -> cp.ndarray:
        """Tone mapping na GPU."""
        
        height, width, channels = aces_image_gpu.shape
        total_pixels = height * width
        
        # Przygotowanie buforów
        input_flat = aces_image_gpu.reshape(-1)
        output_flat = cp.zeros_like(input_flat)
        
        # Konfiguracja kernela
        block_size = 256
        grid_size = (total_pixels + block_size - 1) // block_size
        
        # Uruchomienie kernela ACES RRT
        self.tone_mapping_kernel(
            (grid_size,), (block_size,),
            (input_flat, output_flat,
             tone_params.aces_a, tone_params.aces_b,
             tone_params.aces_c, tone_params.aces_d, tone_params.aces_e,
             width, height)
        )
        
        return output_flat.reshape(height, width, channels)
    
    def _gpu_convert_from_aces(self, aces_image_gpu: cp.ndarray) -> cp.ndarray:
        """Konwersja ACES -> sRGB na GPU."""
        
        # Macierz transformacji ACES -> sRGB
        matrix_gpu = cp.array([
            2.52169, -1.13413, -0.38756,
            -0.27648, 1.37272, -0.09624,
            -0.01538, -0.15298, 1.16835
        ], dtype=cp.float32)
        
        # Transformacja macierzowa
        pixels_gpu = aces_image_gpu.reshape(-1, 3)
        srgb_pixels_gpu = cp.dot(pixels_gpu, matrix_gpu.reshape(3, 3).T)
        
        # Clamp do [0, 1]
        srgb_pixels_gpu = cp.clip(srgb_pixels_gpu, 0.0, 1.0)
        
        # Aplikacja gamma sRGB
        gamma_corrected = cp.where(
            srgb_pixels_gpu <= 0.0031308,
            srgb_pixels_gpu * 12.92,
            1.055 * cp.power(srgb_pixels_gpu, 1.0/2.4) - 0.055
        )
        
        return gamma_corrected.reshape(aces_image_gpu.shape)
```

---

**Następna część:** [6of6 - Aplikacje Praktyczne](gatto-WORKING-03-algorithms-08-advanced-01-aces-6of6.md)  
**Poprzednia część:** [4of6 - Parametry i Konfiguracja](gatto-WORKING-03-algorithms-08-advanced-01-aces-4of6.md)  
**Powrót do:** [Spis treści](gatto-WORKING-03-algorithms-08-advanced-01-aces-0of6.md)

*Część 5of6 - Analiza Wydajności i Optymalizacje | Wersja 1.0 | 2024-01-20*