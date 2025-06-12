import unittest
import numpy as np

import time # Potrzebne do pomiaru czasu
from app.core.development_logger import get_logger


try:
    from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU
    from app.algorithms.algorithm_01_palette.algorithm_gpu_cpu_fallback import map_pixels_to_palette_cpu
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Błąd importu: {e}")
    IMPORTS_SUCCESSFUL = False

@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Nie udało się zaimportować modułów aplikacji.")
class TestFinalAlgorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Inicjalizacja algorytmu tylko raz dla wszystkich testów w tej klasie."""
        print("\n" + "="*70)
        print("🚀 Inicjalizacja algorytmu OpenCL...")
        try:
            cls.algorithm = PaletteMappingAlgorithmGPU()
            cls.algorithm_available = True
        except Exception as e:
            print(f"⚠️ KRYTYCZNA UWAGA: Inicjalizacja algorytmu OpenCL nie powiodła się: {e}")
            cls.algorithm = None
            cls.algorithm_available = False
        print("="*70)

    def test_gpu_vs_cpu_consistency(self):
        """Porównuje wynik z GPU (OpenCL) z wynikiem z CPU, aby sprawdzić spójność."""
        if not self.algorithm_available:
            self.skipTest("Algorytm OpenCL nie jest dostępny, pomijam test spójności.")

        print("\n📢 START: Test spójności GPU vs CPU")
        
        # Stwórz duży obraz, aby na pewno uruchomić ścieżkę GPU
        image_size = 256
        test_image = np.random.randint(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
        test_palette = [[10, 20, 30], [100, 120, 140], [200, 220, 240]]
        test_config = {'hue_weight': 3.0}

        # 1. Obliczenia na GPU (OpenCL)
        print("⚙️  Uruchamiam obliczenia na GPU (OpenCL)...")
        start_gpu = time.time()
        result_gpu = self.algorithm._map_pixels_to_palette(test_image, test_palette, test_config)
        end_gpu = time.time()
        print(f"   -> Czas GPU: {(end_gpu - start_gpu) * 1000:.2f} ms")
        self.assertIsNotNone(result_gpu)
        
        # 2. Obliczenia na CPU (dla porównania)
        print("⚙️  Uruchamiam obliczenia na CPU dla weryfikacji...")
        
        logger = get_logger()
        start_cpu = time.time()
        result_cpu = map_pixels_to_palette_cpu(test_image, test_palette, test_config, logger)
        end_cpu = time.time()
        print(f"   -> Czas CPU: {(end_cpu - start_cpu) * 1000:.2f} ms")
        self.assertIsNotNone(result_cpu)

        # 3. Porównanie wyników
        print("📊 Porównywanie wyników GPU i CPU...")
        difference = np.sum(result_gpu.astype("int32") - result_cpu.astype("int32"))
        
        # Ze względu na różnice w precyzji obliczeń zmiennoprzecinkowych na CPU i GPU,
        # dopuszczamy niewielki próg błędu.
        self.assertAlmostEqual(difference, 0, delta=image_size*image_size, 
                               msg="Wyniki z GPU i CPU znacznie się różnią!")
        
        print("✅ SUKCES: Wyniki z GPU i CPU są spójne!")

if __name__ == '__main__':
    unittest.main(verbosity=2)

