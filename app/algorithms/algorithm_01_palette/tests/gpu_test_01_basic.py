import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Czysty import, który powinien działać, gdy test jest uruchamiany przez 'unittest'
# Unittest sam dodaje korzeń projektu do ścieżki.
try:
    from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU, IS_GPU_ACCELERATED
    from app.algorithms.algorithm_01_palette.algorithm_gpu_taichi_init import TAICHI_AVAILABLE
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    # Ten blok jest na wszelki wypadek, gdyby ktoś próbował uruchomić plik bezpośrednio
    print(f"❌ [BŁĄD IMPORTU] Nie udało się zaimportować modułów. Upewnij się, że uruchamiasz testy za pomocą 'python -m unittest'.")
    print(f"Błąd: {e}")
    PaletteMappingAlgorithmGPU = None
    IS_GPU_ACCELERATED = False
    TAICHI_AVAILABLE = False
    IMPORTS_SUCCESSFUL = False

class TestGpuBasic(unittest.TestCase):
    """
    Podstawowy zestaw testów dla algorytmu GPU.
    Cel: Sprawdzenie, czy środowisko jest poprawnie skonfigurowane,
         czy Taichi (silnik GPU) się inicjalizuje i czy można wykonać
         prostą operację na danych testowych.
    """

    def setUp(self):
        """Metoda wywoływana przed każdym testem."""
        print("\n" + "="*70)
        print(f"🚀 Rozpoczynam test: {self._testMethodName}")
        print("="*70)
        if not IMPORTS_SUCCESSFUL:
             self.fail("Importy modułów aplikacji nie powiodły się. Test przerwany.")
        if not TAICHI_AVAILABLE:
            self.skipTest("Taichi nie jest dostępne. Pomijam testy GPU.")


    def test_01_gpu_initialization_and_simple_run(self):
        """
        Testuje inicjalizację algorytmu i podstawowe uruchomienie mapowania.
        Kroki:
        1. Sprawdza, czy Taichi zgłasza dostępność akceleracji GPU.
        2. Tworzy instancję algorytmu `PaletteMappingAlgorithmGPU`.
        3. Przygotowuje mały obraz testowy i paletę.
        4. Wywołuje wewnętrzną metodę `_map_pixels_to_palette`.
        5. Sprawdza, czy wynik ma poprawny kształt i typ.
        """
        print("📢 Krok 1: Sprawdzanie statusu akceleracji GPU...")
        if IS_GPU_ACCELERATED:
            print("✅ SUKCES: Taichi zgłasza gotowość do pracy na GPU!")
        else:
            print("⚠️ UWAGA: Taichi pracuje w trybie CPU. Test będzie kontynuowany, ale bez użycia GPU.")

        print("\n📢 Krok 2: Tworzenie instancji algorytmu `PaletteMappingAlgorithmGPU`...")
        try:
            algorithm = PaletteMappingAlgorithmGPU()
            self.assertIsNotNone(algorithm, "Instancja algorytmu nie powinna być None.")
            print("✅ SUKCES: Instancja algorytmu została pomyślnie utworzona.")
        except Exception as e:
            self.fail(f"Nie udało się utworzyć instancji algorytmu GPU. Błąd: {e}")

        print("\n📢 Krok 3: Przygotowanie danych testowych (obraz 4x4 piksele, paleta 2 kolory)...")
        # Prosty obraz 4x4 piksele w formacie RGB (uint8)
        test_image_array = np.array([
            [[255, 0, 0], [240, 10, 10], [10, 0, 0], [0, 10, 10]],
            [[0, 255, 0], [10, 240, 10], [0, 10, 0], [10, 0, 10]],
            [[0, 0, 255], [10, 10, 240], [0, 0, 10], [10, 10, 0]],
            [[255, 255, 0], [240, 240, 10], [10, 10, 10], [5, 5, 5]],
        ], dtype=np.uint8)
        
        # Prosta paleta: czarny i biały
        test_palette = [[0, 0, 0], [255, 255, 255]]
        
        # Domyślna konfiguracja
        test_config = algorithm.default_config.copy()
        test_config['force_cpu'] = False # Upewniamy się, że nie wymuszamy CPU

        print(f"  - Rozmiar obrazu: {test_image_array.shape}")
        print(f"  - Paleta: {test_palette}")
        print("✅ SUKCES: Dane testowe gotowe.")

        print("\n📢 Krok 4: Wywołanie metody `_map_pixels_to_palette`...")
        try:
            result_array = algorithm._map_pixels_to_palette(test_image_array, test_palette, test_config)
            print("✅ SUKCES: Metoda zakończyła pracę bez błędów.")
        except Exception as e:
            self.fail(f"Wywołanie `_map_pixels_to_palette` zakończyło się błędem: {e}")

        print("\n📢 Krok 5: Weryfikacja wyniku...")
        self.assertIsNotNone(result_array, "Wynik nie powinien być None.")
        self.assertIsInstance(result_array, np.ndarray, "Wynik powinien być tablicą numpy.")
        self.assertEqual(test_image_array.shape, result_array.shape, "Wynikowy obraz ma inne wymiary niż wejściowy.")
        self.assertEqual(result_array.dtype, np.uint8, "Typ danych wynikowej tablicy powinien być uint8.")
        
        unique_colors = np.unique(result_array.reshape(-1, 3), axis=0)
        for color in unique_colors:
            self.assertIn(list(color), test_palette, f"Znaleziono nieoczekiwany kolor {color} w wyniku.")

        print("✅ SUKCES: Wynik przeszedł wszystkie asercje! Obraz został przetworzony.")


    def tearDown(self):
        """Metoda wywoływana po każdym teście."""
        print("\n" + "-"*70)
        print(f"🏁 Zakończono test: {self._testMethodName}")
        print("-"*70)

if __name__ == '__main__':
    # Ta część jest bardziej informacyjna, główne uruchomienie powinno iść przez 'python -m unittest'
    print("="*80)
    print("||   URUCHAMIANIE PODSTAWOWYCH TESTÓW DLA ALGORYTMU GPU   ||")
    print("||   Aby uruchomić poprawnie, użyj komendy:             ||")
    print("||   python -m unittest ścieżka.do.pliku_testowego      ||")
    print("="*80)
    unittest.main(verbosity=2)
