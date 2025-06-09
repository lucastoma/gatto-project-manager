# LAB Color Space Transfer - Część 3a: Testy i Benchmarki

**Część 3a z 3: Testy Jednostkowe i Benchmarki Wydajności**

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 2-3 godziny | **Złożoność**: O(n)

---

## Przegląd Części 3a

Ta część koncentruje się na testowaniu i benchmarkingu algorytmu LAB Color Transfer. Omówimy testy jednostkowe, testy wydajności oraz analizę jakości.

### Zawartość
- Testy jednostkowe i integracyjne
- Benchmarki wydajności
- Analiza jakości vs szybkości
- Testy regresji
- Profilowanie pamięci

---

## Testy Jednostkowe

### Test Suite dla LAB Transfer

```python
import unittest
import numpy as np
from PIL import Image
import tempfile
import os
import time

class TestLABColorTransfer(unittest.TestCase):
    def setUp(self):
        """Przygotowanie testów"""
        self.transfer = LABColorTransferAdvanced()
        
        # Utwórz testowe obrazy
        self.test_image_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.test_image_lab = self.transfer.rgb_to_lab_optimized(self.test_image_rgb)
        
        # Temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Czyszczenie po testach"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rgb_to_lab_conversion(self):
        """Test konwersji RGB → LAB"""
        # Test podstawowej konwersji
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        lab = self.transfer.rgb_to_lab_optimized(rgb)
        
        # Sprawdź wymiary
        self.assertEqual(lab.shape, rgb.shape)
        
        # Sprawdź zakresy LAB
        self.assertTrue(np.all(lab[:, :, 0] >= 0))  # L >= 0
        self.assertTrue(np.all(lab[:, :, 0] <= 100))  # L <= 100
        self.assertTrue(np.all(lab[:, :, 1] >= -128))  # a >= -128
        self.assertTrue(np.all(lab[:, :, 1] <= 127))  # a <= 127
        self.assertTrue(np.all(lab[:, :, 2] >= -128))  # b >= -128
        self.assertTrue(np.all(lab[:, :, 2] <= 127))  # b <= 127
    
    def test_lab_to_rgb_conversion(self):
        """Test konwersji LAB → RGB"""
        # Test round-trip conversion
        original_rgb = self.test_image_rgb
        lab = self.transfer.rgb_to_lab_optimized(original_rgb)
        recovered_rgb = self.transfer.lab_to_rgb_optimized(lab)
        
        # Sprawdź wymiary
        self.assertEqual(recovered_rgb.shape, original_rgb.shape)
        
        # Sprawdź zakresy RGB
        self.assertTrue(np.all(recovered_rgb >= 0))
        self.assertTrue(np.all(recovered_rgb <= 255))
        
        # Sprawdź podobieństwo (tolerancja na błędy konwersji)
        diff = np.abs(original_rgb.astype(float) - recovered_rgb.astype(float))
        mean_diff = np.mean(diff)
        self.assertLess(mean_diff, 5.0, "Round-trip conversion error too high")
    
    def test_basic_lab_transfer(self):
        """Test podstawowego transferu LAB"""
        source_lab = self.test_image_lab
        target_lab = np.random.rand(50, 50, 3) * 100  # Random target
        
        result_lab = self.transfer.basic_lab_transfer(source_lab, target_lab)
        
        # Sprawdź wymiary
        self.assertEqual(result_lab.shape, source_lab.shape)
        
        # Sprawdź czy transfer zmienił statystyki
        source_stats = self.transfer.calculate_lab_statistics(source_lab)
        result_stats = self.transfer.calculate_lab_statistics(result_lab)
        target_stats = self.transfer.calculate_lab_statistics(target_lab)
        
        # Statystyki wyniku powinny być bliższe targetowi niż source
        for channel in ['L', 'a', 'b']:
            source_diff = abs(source_stats[channel]['mean'] - target_stats[channel]['mean'])
            result_diff = abs(result_stats[channel]['mean'] - target_stats[channel]['mean'])
            
            if source_diff > 1:  # Tylko jeśli była różnica do skorygowania
                self.assertLess(result_diff, source_diff, 
                               f"Transfer failed for channel {channel}")
    
    def test_weighted_transfer(self):
        """Test transferu z wagami"""
        source_lab = self.test_image_lab
        target_lab = np.random.rand(50, 50, 3) * 100
        
        # Test z różnymi wagami
        weights = {'L': 0.5, 'a': 1.0, 'b': 0.8}
        result_lab = self.transfer.weighted_lab_transfer(source_lab, target_lab, weights)
        
        self.assertEqual(result_lab.shape, source_lab.shape)
        
        # Sprawdź czy wagi zostały zastosowane
        # (trudne do precyzyjnego testu, sprawdzamy tylko podstawowe właściwości)
        self.assertFalse(np.array_equal(result_lab, source_lab))
    
    def test_selective_transfer(self):
        """Test selektywnego transferu"""
        source_lab = self.test_image_lab.copy()
        target_lab = np.random.rand(50, 50, 3) * 100
        
        # Transfer tylko kanałów a i b
        result_lab = self.transfer.selective_lab_transfer(
            source_lab, target_lab, ['a', 'b']
        )
        
        # Kanał L powinien pozostać niezmieniony
        np.testing.assert_array_equal(
            result_lab[:, :, 0], source_lab[:, :, 0],
            "L channel should remain unchanged in selective transfer"
        )
        
        # Kanały a i b powinny się zmienić
        self.assertFalse(np.array_equal(result_lab[:, :, 1], source_lab[:, :, 1]))
        self.assertFalse(np.array_equal(result_lab[:, :, 2], source_lab[:, :, 2]))
    
    def test_statistics_calculation(self):
        """Test obliczania statystyk LAB"""
        # Utwórz obraz o znanych statystykach
        test_lab = np.zeros((10, 10, 3))
        test_lab[:, :, 0] = 50  # L = 50
        test_lab[:, :, 1] = 10  # a = 10
        test_lab[:, :, 2] = -5  # b = -5
        
        stats = self.transfer.calculate_lab_statistics(test_lab)
        
        # Sprawdź obliczone statystyki
        self.assertAlmostEqual(stats['L']['mean'], 50.0, places=1)
        self.assertAlmostEqual(stats['a']['mean'], 10.0, places=1)
        self.assertAlmostEqual(stats['b']['mean'], -5.0, places=1)
        
        # Sprawdź czy wszystkie statystyki są obecne
        for channel in ['L', 'a', 'b']:
            self.assertIn('mean', stats[channel])
            self.assertIn('std', stats[channel])
            self.assertIn('min', stats[channel])
            self.assertIn('max', stats[channel])
    
    def test_delta_e_calculation(self):
        """Test obliczania Delta E"""
        # Identyczne obrazy powinny mieć Delta E = 0
        lab1 = self.test_image_lab
        lab2 = lab1.copy()
        
        delta_e = self.transfer.calculate_delta_e_lab(lab1, lab2)
        
        self.assertEqual(delta_e.shape, lab1.shape[:2])
        np.testing.assert_array_almost_equal(delta_e, 0, decimal=5)
        
        # Test z różnymi obrazami
        lab2[:, :, 0] += 10  # Zmień L o 10
        delta_e = self.transfer.calculate_delta_e_lab(lab1, lab2)
        
        # Delta E powinno być około 10
        np.testing.assert_array_almost_equal(delta_e, 10, decimal=1)
    
    def test_config_validation(self):
        """Test walidacji konfiguracji"""
        config = LABTransferConfig()
        
        # Poprawna konfiguracja
        self.assertTrue(config.validate())
        
        # Niepoprawna metoda
        config.method = 'invalid_method'
        with self.assertRaises(AssertionError):
            config.validate()
        
        # Niepoprawne wagi
        config.method = 'basic'
        config.channel_weights['L'] = -1  # Niepoprawna waga
        with self.assertRaises(AssertionError):
            config.validate()
    
    def test_file_processing(self):
        """Test przetwarzania plików"""
        # Utwórz testowe pliki
        source_path = os.path.join(self.temp_dir, 'source.png')
        target_path = os.path.join(self.temp_dir, 'target.png')
        output_path = os.path.join(self.temp_dir, 'output.png')
        
        # Zapisz testowe obrazy
        Image.fromarray(self.test_image_rgb).save(source_path)
        target_rgb = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        Image.fromarray(target_rgb).save(target_path)
        
        # Przetestuj przetwarzanie
        success = self.transfer.process_with_config(source_path, target_path, output_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Sprawdź czy output jest poprawny
        result_image = Image.open(output_path)
        self.assertEqual(result_image.size, (100, 100))

class TestLABPerformance(unittest.TestCase):
    """Testy wydajności"""
    
    def setUp(self):
        self.transfer = LABColorTransferAdvanced()
        
        # Różne rozmiary obrazów do testów
        self.small_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.medium_image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        self.large_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
    
    def test_conversion_performance(self):
        """Test wydajności konwersji"""
        images = [
            ("small", self.small_image),
            ("medium", self.medium_image),
            ("large", self.large_image)
        ]
        
        for name, image in images:
            with self.subTest(size=name):
                # Test RGB → LAB
                start_time = time.time()
                lab = self.transfer.rgb_to_lab_optimized(image)
                rgb_to_lab_time = time.time() - start_time
                
                # Test LAB → RGB
                start_time = time.time()
                rgb = self.transfer.lab_to_rgb_optimized(lab)
                lab_to_rgb_time = time.time() - start_time
                
                print(f"\n{name} image ({image.shape}):")
                print(f"  RGB→LAB: {rgb_to_lab_time:.3f}s")
                print(f"  LAB→RGB: {lab_to_rgb_time:.3f}s")
                
                # Sprawdź czy czasy są rozsądne
                pixels = image.shape[0] * image.shape[1]
                self.assertLess(rgb_to_lab_time, pixels / 10000, 
                               f"RGB→LAB too slow for {name} image")
                self.assertLess(lab_to_rgb_time, pixels / 10000, 
                               f"LAB→RGB too slow for {name} image")
    
    def test_transfer_performance(self):
        """Test wydajności transferu"""
        source_lab = self.transfer.rgb_to_lab_optimized(self.medium_image)
        target_lab = self.transfer.rgb_to_lab_optimized(self.small_image)
        
        methods = [
            ('basic', lambda: self.transfer.basic_lab_transfer(source_lab, target_lab)),
            ('weighted', lambda: self.transfer.weighted_lab_transfer(source_lab, target_lab)),
            ('selective', lambda: self.transfer.selective_lab_transfer(source_lab, target_lab))
        ]
        
        for method_name, method_func in methods:
            with self.subTest(method=method_name):
                start_time = time.time()
                result = method_func()
                transfer_time = time.time() - start_time
                
                print(f"\n{method_name} transfer: {transfer_time:.3f}s")
                
                # Sprawdź wynik
                self.assertEqual(result.shape, source_lab.shape)
                
                # Sprawdź czas
                pixels = source_lab.shape[0] * source_lab.shape[1]
                self.assertLess(transfer_time, pixels / 50000, 
                               f"{method_name} transfer too slow")

# Testy regresji
class TestLABRegression(unittest.TestCase):
    """Testy regresji dla sprawdzenia czy zmiany nie psują istniejącej funkcjonalności"""
    
    def setUp(self):
        self.transfer = LABColorTransferAdvanced()
        
        # Referencyjne obrazy testowe
        self.reference_source = self.create_reference_image('source')
        self.reference_target = self.create_reference_image('target')
        
        # Oczekiwane wyniki (hash lub statystyki)
        self.expected_results = {
            'basic_transfer_mean_l': 45.2,
            'basic_transfer_mean_a': 2.1,
            'basic_transfer_mean_b': -1.8,
            'conversion_accuracy': 0.95
        }
    
    def create_reference_image(self, image_type):
        """Tworzy referencyjne obrazy testowe"""
        np.random.seed(42)  # Deterministyczne wyniki
        
        if image_type == 'source':
            # Obraz z dominującymi niebieskimi tonami
            image = np.zeros((50, 50, 3), dtype=np.uint8)
            image[:, :] = [100, 150, 200]  # Niebieski
            
            # Dodaj kontrolowany szum
            noise = np.random.normal(0, 5, image.shape)
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
            
        elif image_type == 'target':
            # Obraz z dominującymi czerwonymi tonami
            image = np.zeros((30, 30, 3), dtype=np.uint8)
            image[:, :] = [200, 100, 80]  # Czerwony
            
            # Dodaj kontrolowany szum
            noise = np.random.normal(0, 3, image.shape)
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def test_basic_transfer_regression(self):
        """Test regresji dla podstawowego transferu"""
        source_lab = self.transfer.rgb_to_lab_optimized(self.reference_source)
        target_lab = self.transfer.rgb_to_lab_optimized(self.reference_target)
        
        result_lab = self.transfer.basic_lab_transfer(source_lab, target_lab)
        
        # Sprawdź statystyki wyniku
        stats = self.transfer.calculate_lab_statistics(result_lab)
        
        # Porównaj z oczekiwanymi wynikami (z tolerancją)
        self.assertAlmostEqual(
            stats['L']['mean'], 
            self.expected_results['basic_transfer_mean_l'], 
            delta=2.0,
            msg="L channel mean regression detected"
        )
        
        self.assertAlmostEqual(
            stats['a']['mean'], 
            self.expected_results['basic_transfer_mean_a'], 
            delta=1.0,
            msg="a channel mean regression detected"
        )
        
        self.assertAlmostEqual(
            stats['b']['mean'], 
            self.expected_results['basic_transfer_mean_b'], 
            delta=1.0,
            msg="b channel mean regression detected"
        )
    
    def test_conversion_accuracy_regression(self):
        """Test regresji dla dokładności konwersji"""
        # Test round-trip accuracy
        original_rgb = self.reference_source
        lab = self.transfer.rgb_to_lab_optimized(original_rgb)
        recovered_rgb = self.transfer.lab_to_rgb_optimized(lab)
        
        # Oblicz dokładność konwersji
        diff = np.abs(original_rgb.astype(float) - recovered_rgb.astype(float))
        accuracy = 1.0 - (np.mean(diff) / 255.0)
        
        self.assertGreaterEqual(
            accuracy, 
            self.expected_results['conversion_accuracy'],
            "Conversion accuracy regression detected"
        )

if __name__ == '__main__':
    # Uruchom testy
    unittest.main(verbosity=2)
```

---

## Benchmarki Wydajności

### Benchmark Suite

```python
import time
import psutil
import matplotlib.pyplot as plt
from memory_profiler import profile

class LABTransferBenchmark:
    def __init__(self):
        self.transfer = LABColorTransferAdvanced()
        self.results = {}
    
    def benchmark_conversion_sizes(self):
        """Benchmark konwersji dla różnych rozmiarów"""
        sizes = [(100, 100), (250, 250), (500, 500), (750, 750), (1000, 1000)]
        
        rgb_to_lab_times = []
        lab_to_rgb_times = []
        memory_usage = []
        
        for width, height in sizes:
            print(f"\nBenchmarking {width}x{height}...")
            
            # Utwórz testowy obraz
            test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            # Benchmark RGB → LAB
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            lab = self.transfer.rgb_to_lab_optimized(test_image)
            
            rgb_to_lab_time = time.time() - start_time
            
            # Benchmark LAB → RGB
            start_time = time.time()
            
            rgb = self.transfer.lab_to_rgb_optimized(lab)
            
            lab_to_rgb_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Zapisz wyniki
            rgb_to_lab_times.append(rgb_to_lab_time)
            lab_to_rgb_times.append(lab_to_rgb_time)
            memory_usage.append(end_memory - start_memory)
            
            print(f"  RGB→LAB: {rgb_to_lab_time:.3f}s")
            print(f"  LAB→RGB: {lab_to_rgb_time:.3f}s")
            print(f"  Memory: {end_memory - start_memory:.1f}MB")
        
        # Zapisz wyniki
        self.results['conversion_benchmark'] = {
            'sizes': sizes,
            'rgb_to_lab_times': rgb_to_lab_times,
            'lab_to_rgb_times': lab_to_rgb_times,
            'memory_usage': memory_usage
        }
        
        return self.results['conversion_benchmark']
    
    def benchmark_transfer_methods(self):
        """Benchmark różnych metod transferu"""
        # Testowy obraz 500x500
        test_size = (500, 500)
        source_image = np.random.randint(0, 256, (*test_size, 3), dtype=np.uint8)
        target_image = np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)
        
        source_lab = self.transfer.rgb_to_lab_optimized(source_image)
        target_lab = self.transfer.rgb_to_lab_optimized(target_image)
        
        methods = {
            'basic': lambda: self.transfer.basic_lab_transfer(source_lab, target_lab),
            'weighted': lambda: self.transfer.weighted_lab_transfer(source_lab, target_lab),
            'selective': lambda: self.transfer.selective_lab_transfer(source_lab, target_lab, ['a', 'b']),
            'adaptive': lambda: self.transfer.adaptive_lab_transfer(source_lab, target_lab)
        }
        
        method_times = {}
        method_memory = {}
        
        for method_name, method_func in methods.items():
            print(f"\nBenchmarking {method_name} method...")
            
            # Wielokrotne uruchomienia dla dokładności
            times = []
            for i in range(5):
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                start_time = time.time()
                
                result = method_func()
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            method_times[method_name] = {
                'avg': avg_time,
                'std': std_time,
                'times': times
            }
            
            print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
        
        self.results['method_benchmark'] = method_times
        return method_times
    
    def benchmark_quality_vs_speed(self):
        """Benchmark jakości vs szybkości"""
        # Utwórz realistyczne obrazy testowe
        source_image = self.create_test_image('landscape')
        target_image = self.create_test_image('sunset')
        
        source_lab = self.transfer.rgb_to_lab_optimized(source_image)
        target_lab = self.transfer.rgb_to_lab_optimized(target_image)
        
        configs = {
            'fast': LABTransferConfig(),
            'balanced': LABTransferConfig(),
            'quality': LABTransferConfig()
        }
        
        # Konfiguruj dla różnych priorytetów
        configs['fast'].method = 'basic'
        configs['balanced'].method = 'weighted'
        configs['quality'].method = 'adaptive'
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\nTesting {config_name} configuration...")
            
            transfer = LABColorTransferAdvanced(config)
            
            start_time = time.time()
            
            if config.method == 'basic':
                result_lab = transfer.basic_lab_transfer(source_lab, target_lab)
            elif config.method == 'weighted':
                result_lab = transfer.weighted_lab_transfer(source_lab, target_lab)
            elif config.method == 'adaptive':
                result_lab = transfer.adaptive_lab_transfer(source_lab, target_lab)
            
            processing_time = time.time() - start_time
            
            # Oblicz jakość
            delta_e = transfer.calculate_delta_e_lab(result_lab, target_lab)
            quality_score = 100 - np.mean(delta_e)  # Wyższa wartość = lepsza jakość
            
            results[config_name] = {
                'time': processing_time,
                'quality': quality_score,
                'delta_e_mean': np.mean(delta_e),
                'delta_e_std': np.std(delta_e)
            }
            
            print(f"  Time: {processing_time:.3f}s")
            print(f"  Quality Score: {quality_score:.1f}")
            print(f"  Delta E: {np.mean(delta_e):.1f} ± {np.std(delta_e):.1f}")
        
        self.results['quality_vs_speed'] = results
        return results
    
    def create_test_image(self, image_type, size=(400, 400)):
        """Tworzy realistyczne obrazy testowe"""
        if image_type == 'landscape':
            # Symuluj krajobraz: niebo + ziemia
            image = np.zeros((*size, 3), dtype=np.uint8)
            
            # Niebo (górna połowa)
            sky_height = size[0] // 2
            image[:sky_height, :] = [135, 206, 235]  # Sky blue
            
            # Ziemia (dolna połowa)
            image[sky_height:, :] = [34, 139, 34]  # Forest green
            
            # Dodaj szum
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
            
        elif image_type == 'sunset':
            # Symuluj zachód słońca
            image = np.zeros((*size, 3), dtype=np.uint8)
            
            # Gradient od pomarańczowego do czerwonego
            for i in range(size[0]):
                ratio = i / size[0]
                color = [
                    int(255 * (1 - ratio * 0.3)),  # R
                    int(165 * (1 - ratio * 0.5)),  # G
                    int(0 * (1 - ratio))           # B
                ]
                image[i, :] = color
            
            # Dodaj szum
            noise = np.random.normal(0, 5, image.shape)
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        else:
            # Domyślny losowy obraz
            image = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        
        return image
    
    def plot_results(self):
        """Rysuje wykresy wyników benchmarków"""
        if 'conversion_benchmark' in self.results:
            self.plot_conversion_benchmark()
        
        if 'method_benchmark' in self.results:
            self.plot_method_benchmark()
        
        if 'quality_vs_speed' in self.results:
            self.plot_quality_vs_speed()
    
    def plot_conversion_benchmark(self):
        """Wykres wydajności konwersji"""
        data = self.results['conversion_benchmark']
        sizes = [w * h for w, h in data['sizes']]
        
        plt.figure(figsize=(12, 4))
        
        # Wykres czasów
        plt.subplot(1, 2, 1)
        plt.plot(sizes, data['rgb_to_lab_times'], 'b-o', label='RGB→LAB')
        plt.plot(sizes, data['lab_to_rgb_times'], 'r-o', label='LAB→RGB')
        plt.xlabel('Liczba pikseli')
        plt.ylabel('Czas [s]')
        plt.title('Wydajność konwersji kolorów')
        plt.legend()
        plt.grid(True)
        
        # Wykres pamięci
        plt.subplot(1, 2, 2)
        plt.plot(sizes, data['memory_usage'], 'g-o')
        plt.xlabel('Liczba pikseli')
        plt.ylabel('Zużycie pamięci [MB]')
        plt.title('Zużycie pamięci')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('lab_conversion_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_method_benchmark(self):
        """Wykres porównania metod"""
        data = self.results['method_benchmark']
        
        methods = list(data.keys())
        times = [data[method]['avg'] for method in methods]
        errors = [data[method]['std'] for method in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, times, yerr=errors, capsize=5, 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        
        plt.ylabel('Czas przetwarzania [s]')
        plt.title('Porównanie wydajności metod transferu LAB')
        plt.grid(True, alpha=0.3)
        
        # Dodaj wartości na słupkach
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('lab_method_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_quality_vs_speed(self):
        """Wykres jakości vs szybkości"""
        data = self.results['quality_vs_speed']
        
        configs = list(data.keys())
        times = [data[config]['time'] for config in configs]
        qualities = [data[config]['quality'] for config in configs]
        
        plt.figure(figsize=(8, 6))
        
        colors = ['red', 'orange', 'green']
        for i, config in enumerate(configs):
            plt.scatter(times[i], qualities[i], s=100, c=colors[i], 
                       label=config.capitalize(), alpha=0.7)
            plt.annotate(config, (times[i], qualities[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Czas przetwarzania [s]')
        plt.ylabel('Jakość (wyższa = lepsza)')
        plt.title('Jakość vs Szybkość - LAB Transfer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lab_quality_vs_speed.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    @profile
    def memory_profile_transfer(self):
        """Profilowanie pamięci dla transferu"""
        # Duży obraz testowy
        large_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        target_image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        
        print("Starting memory profiling...")
        
        # Konwersja do LAB
        source_lab = self.transfer.rgb_to_lab_optimized(large_image)
        target_lab = self.transfer.rgb_to_lab_optimized(target_image)
        
        # Transfer
        result_lab = self.transfer.basic_lab_transfer(source_lab, target_lab)
        
        # Konwersja z powrotem
        result_rgb = self.transfer.lab_to_rgb_optimized(result_lab)
        
        print("Memory profiling complete.")
        return result_rgb

# Uruchomienie benchmarków
if __name__ == '__main__':
    benchmark = LABTransferBenchmark()
    
    print("=== LAB Transfer Benchmark Suite ===")
    
    # Benchmark konwersji
    print("\n1. Benchmarking conversion performance...")
    conversion_results = benchmark.benchmark_conversion_sizes()
    
    # Benchmark metod
    print("\n2. Benchmarking transfer methods...")
    method_results = benchmark.benchmark_transfer_methods()
    
    # Benchmark jakości vs szybkości
    print("\n3. Benchmarking quality vs speed...")
    quality_results = benchmark.benchmark_quality_vs_speed()
    
    # Generuj wykresy
    print("\n4. Generating plots...")
    benchmark.plot_results()
    
    # Profilowanie pamięci
    print("\n5. Memory profiling...")
    benchmark.memory_profile_transfer()
    
    print("\n=== Benchmark Complete ===")
```

---

## Analiza Wyników Testów

### Metryki Wydajności

```python
class LABPerformanceAnalyzer:
    def __init__(self, benchmark_results):
        self.results = benchmark_results
    
    def analyze_scalability(self):
        """Analiza skalowalności algorytmu"""
        conversion_data = self.results['conversion_benchmark']
        
        sizes = [w * h for w, h in conversion_data['sizes']]
        rgb_to_lab_times = conversion_data['rgb_to_lab_times']
        
        # Oblicz throughput (piksele/sekunda)
        throughput = [size / time for size, time in zip(sizes, rgb_to_lab_times)]
        
        # Analiza złożoności
        # Sprawdź czy czas rośnie liniowo z liczbą pikseli
        correlation = np.corrcoef(sizes, rgb_to_lab_times)[0, 1]
        
        print(f"\n=== Analiza Skalowalności ===")
        print(f"Korelacja rozmiar-czas: {correlation:.3f}")
        print(f"Średni throughput: {np.mean(throughput):.0f} pikseli/s")
        
        if correlation > 0.95:
            print("✅ Algorytm ma liniową złożoność czasową")
        else:
            print("⚠️ Algorytm może mieć nieliniową złożoność")
        
        return {
            'correlation': correlation,
            'throughput': throughput,
            'scalability_rating': 'linear' if correlation > 0.95 else 'non-linear'
        }
    
    def analyze_method_efficiency(self):
        """Analiza efektywności różnych metod"""
        method_data = self.results['method_benchmark']
        
        print(f"\n=== Analiza Efektywności Metod ===")
        
        # Sortuj metody według czasu
        sorted_methods = sorted(method_data.items(), key=lambda x: x[1]['avg'])
        
        fastest_method = sorted_methods[0]
        slowest_method = sorted_methods[-1]
        
        print(f"Najszybsza metoda: {fastest_method[0]} ({fastest_method[1]['avg']:.3f}s)")
        print(f"Najwolniejsza metoda: {slowest_method[0]} ({slowest_method[1]['avg']:.3f}s)")
        
        # Oblicz względne różnice
        baseline_time = fastest_method[1]['avg']
        
        efficiency_ratios = {}
        for method, data in method_data.items():
            ratio = data['avg'] / baseline_time
            efficiency_ratios[method] = ratio
            print(f"{method}: {ratio:.2f}x wolniejszy od najszybszego")
        
        return efficiency_ratios
    
    def analyze_quality_tradeoffs(self):
        """Analiza kompromisów jakość-szybkość"""
        quality_data = self.results['quality_vs_speed']
        
        print(f"\n=== Analiza Kompromisów Jakość-Szybkość ===")
        
        # Oblicz efficiency score (jakość/czas)
        efficiency_scores = {}
        for config, data in quality_data.items():
            score = data['quality'] / data['time']
            efficiency_scores[config] = score
            
            print(f"{config}:")
            print(f"  Czas: {data['time']:.3f}s")
            print(f"  Jakość: {data['quality']:.1f}")
            print(f"  Efficiency Score: {score:.1f}")
        
        # Znajdź najlepszy kompromis
        best_compromise = max(efficiency_scores.items(), key=lambda x: x[1])
        print(f"\nNajlepszy kompromis: {best_compromise[0]}")
        
        return efficiency_scores
    
    def generate_performance_report(self):
        """Generuje raport wydajności"""
        print("\n" + "="*50)
        print("         RAPORT WYDAJNOŚCI LAB TRANSFER")
        print("="*50)
        
        # Analiza skalowalności
        scalability = self.analyze_scalability()
        
        # Analiza metod
        efficiency = self.analyze_method_efficiency()
        
        # Analiza jakości
        quality_tradeoffs = self.analyze_quality_tradeoffs()
        
        # Rekomendacje
        print(f"\n=== Rekomendacje ===")
        
        if scalability['scalability_rating'] == 'linear':
            print("✅ Algorytm dobrze skaluje się z rozmiarem obrazu")
        else:
            print("⚠️ Rozważ optymalizację dla dużych obrazów")
        
        # Znajdź najszybszą metodę
        fastest_method = min(efficiency.items(), key=lambda x: x[1])[0]
        print(f"✅ Dla szybkości: użyj metody '{fastest_method}'")
        
        # Znajdź najlepszy kompromis
        best_compromise = max(quality_tradeoffs.items(), key=lambda x: x[1])[0]
        print(f"✅ Dla balansu: użyj konfiguracji '{best_compromise}'")
        
        return {
            'scalability': scalability,
            'method_efficiency': efficiency,
            'quality_tradeoffs': quality_tradeoffs
        }

# Przykład użycia
if __name__ == '__main__':
    # Uruchom benchmarki
    benchmark = LABTransferBenchmark()
    
    # Zbierz wyniki
    benchmark.benchmark_conversion_sizes()
    benchmark.benchmark_transfer_methods()
    benchmark.benchmark_quality_vs_speed()
    
    # Analizuj wyniki
    analyzer = LABPerformanceAnalyzer(benchmark.results)
    report = analyzer.generate_performance_report()
```

---

## Testy Integracyjne

### Test Integracji z Głównym Systemem

```python
class TestLABIntegration(unittest.TestCase):
    """Testy integracji LAB Transfer z głównym systemem"""
    
    def setUp(self):
        # Symuluj główny system
        self.main_system = MainColorMatchingSystem()
        self.lab_transfer = LABColorTransferAdvanced()
        
        # Zarejestruj LAB transfer w systemie
        self.main_system.register_algorithm('lab_transfer', self.lab_transfer)
    
    def test_algorithm_registration(self):
        """Test rejestracji algorytmu w systemie"""
        algorithms = self.main_system.get_available_algorithms()
        self.assertIn('lab_transfer', algorithms)
    
    def test_end_to_end_processing(self):
        """Test przetwarzania end-to-end"""
        # Przygotuj pliki testowe
        source_path = 'test_source.jpg'
        target_path = 'test_target.jpg'
        output_path = 'test_output.jpg'
        
        # Utwórz testowe obrazy
        test_source = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        test_target = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
        
        Image.fromarray(test_source).save(source_path)
        Image.fromarray(test_target).save(target_path)
        
        try:
            # Przetwórz przez główny system
            result = self.main_system.process_images(
                source_path=source_path,
                target_path=target_path,
                output_path=output_path,
                algorithm='lab_transfer',
                config={'method': 'basic'}
            )
            
            self.assertTrue(result['success'])
            self.assertTrue(os.path.exists(output_path))
            
            # Sprawdź wynik
            output_image = Image.open(output_path)
            self.assertEqual(output_image.size, (200, 200))
            
        finally:
            # Cleanup
            for path in [source_path, target_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def test_error_handling(self):
        """Test obsługi błędów"""
        # Test z niepoprawnym plikiem
        with self.assertRaises(FileNotFoundError):
            self.main_system.process_images(
                source_path='nonexistent.jpg',
                target_path='also_nonexistent.jpg',
                output_path='output.jpg',
                algorithm='lab_transfer'
            )
        
        # Test z niepoprawną konfiguracją
        result = self.main_system.process_images(
            source_path='test_source.jpg',
            target_path='test_target.jpg',
            output_path='output.jpg',
            algorithm='lab_transfer',
            config={'method': 'invalid_method'}
        )
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
```

---

## Nawigacja

**◀️ Poprzednia część**: [Implementacja Zaawansowana](gatto-WORKING-03-algorithms-05-medium-02-lab-transfer-2of3.md)  
**▶️ Następna część**: [Integracja i Podsumowanie](gatto-WORKING-03-algorithms-05-medium-02-lab-transfer-3bof3.md)  
**🏠 Powrót do**: [Spis Treści Algorytmów](gatto-WORKING-03-algorithms-toc.md)

---

*Ostatnia aktualizacja: 2024-01-20*  
*Autor: GattoNero AI Assistant*  
*Wersja: 2.0*  
*Status: Część 3a - Testy i benchmarki* ✅