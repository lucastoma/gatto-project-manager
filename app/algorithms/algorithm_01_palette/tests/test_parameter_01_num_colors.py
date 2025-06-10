import unittest
import numpy as np
from PIL import Image
import os
from .base_test_case import BaseAlgorithmTestCase
from ..algorithm import PaletteMappingAlgorithm
import logging

# Ustawienie logowania, aby widzieć komunikaty z testu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImprovedTestNumColors(BaseAlgorithmTestCase):
    """
    Ulepszony zestaw testów dla parametru `num_colors`.

    Kluczowa zmiana: Użycie złożonego obrazu 'master' (szum kolorów),
    aby dać algorytmowi K-means realne dane do ekstrakcji palety.
    Obraz 'target' to gradient, na którym efekty kwantyzacji są dobrze widoczne.
    """
    def setUp(self):
        """Metoda wywoływana przed każdym testem."""
        super().setUp()
        self.mapper = PaletteMappingAlgorithm()

        # 1. Stwórz ZŁOŻONY obraz wzorcowy (master) z bogatą paletą kolorów (szum)
        # To jest kluczowe, aby K-means miał z czego wybierać kolory.
        self.master_image_path = self.create_test_image(
            "master_complex.png", shape=(200, 200, 3)  # Domyślnie generuje szum
        )

        # 2. Stwórz obraz docelowy (target) w postaci gradientu
        # Na gradiencie najlepiej widać efekt kwantyzacji kolorów.
        self.target_image_path = self.create_gradient_image()

    def create_gradient_image(self):
        """Tworzy obraz z horyzontalnym gradientem RGB."""
        path = os.path.join(self.test_dir, "gradient_target.png")
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            arr[:, i, 0] = int(i * 2.55)
            arr[:, i, 1] = 128
            arr[:, i, 2] = 255 - int(i * 2.55)
        Image.fromarray(arr).save(path)
        return path

    def run_and_analyze(self, num_colors):
        """Uruchamia algorytm i zwraca metryki."""
        output_path = os.path.join(self.test_dir, f"result_{num_colors}.png")
        
        success = self.mapper.process_images(
            master_path=self.master_image_path,
            target_path=self.target_image_path,
            output_path=output_path,
            num_colors=num_colors
        )
        
        if not success:
            self.fail(f"Przetwarzanie dla num_colors={num_colors} nie powiodło się.")

        # Analiza wyniku
        original_img = Image.open(self.target_image_path)
        result_img = Image.open(output_path)
        original_arr = np.array(original_img)
        result_arr = np.array(result_img)
        
        metrics = {
            'unique_colors': len(np.unique(result_arr.reshape(-1, 3), axis=0)),
            'color_diff': np.mean(np.abs(original_arr.astype(float) - result_arr.astype(float)))
        }
        logging.info(f"Test dla num_colors={num_colors}: {metrics}")
        return metrics

    def test_num_colors_parameter_effect(self):
        """
        Testuje, czy zmiana `num_colors` prawidłowo wpływa na wynik.
        Oczekiwany efekt: Więcej kolorów -> niższy błąd (`color_diff`) i więcej unikalnych kolorów.
        """
        # --- ETAP 1: Uruchomienie testów dla różnych wartości ---

        # Case 1: Wartość typowa (baseline)
        logging.info("Uruchamiam test dla `num_colors=16` (wartość typowa)...")
        result_16 = self.run_and_analyze(16)

        # Case 2: Wartość skrajnie niska
        logging.info("Uruchamiam test dla `num_colors=4` (wartość niska)...")
        result_4 = self.run_and_analyze(4)

        # Case 3: Wartość wysoka
        logging.info("Uruchamiam test dla `num_colors=64` (wartość wysoka)...")
        result_64 = self.run_and_analyze(64)

        # --- ETAP 2: Weryfikacja logiki (Asercje) ---

        logging.info("Weryfikacja wyników...")

        # Sprawdzenie dla niskiej liczby kolorów (w porównaniu do baseline)
        self.assertLessEqual(result_4['unique_colors'], result_16['unique_colors'],
                             "Mniejsza paleta powinna dać mniej lub tyle samo unikalnych kolorów w wyniku.")
        self.assertGreater(result_4['color_diff'], result_16['color_diff'],
                           "Mniejsza paleta powinna skutkować większą średnią różnicą kolorów (większy błąd).")

        # Sprawdzenie dla wysokiej liczby kolorów (w porównaniu do baseline)
        self.assertGreaterEqual(result_64['unique_colors'], result_16['unique_colors'],
                                "Większa paleta powinna dać więcej lub tyle samo unikalnych kolorów w wyniku.")
        self.assertLess(result_64['color_diff'], result_16['color_diff'],
                        "Większa paleta powinna skutkować mniejszą średnią różnicą kolorów (mniejszy błąd).")
        
        logging.info("✅ Wszystkie asercje dla `num_colors` zakończone pomyślnie!")


if __name__ == '__main__':
    unittest.main(verbosity=2)