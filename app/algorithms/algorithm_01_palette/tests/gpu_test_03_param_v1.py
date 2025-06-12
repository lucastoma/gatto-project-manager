import unittest
import numpy as np
from PIL import Image
import os
import sys

# Dodaj ścieżkę do katalogu nadrzędnego, aby zaimportować moduły
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU

class TestPaletteMappingParameters(unittest.TestCase):
    def setUp(self):
        self.algorithm = PaletteMappingAlgorithmGPU()
        self.test_input_path = "test_input.jpg"
        self.test_master_path = "test_master.jpg"
        self.test_output_path = "test_output.jpg"
        # Reset config to default before each test
        self.algorithm.config = self.algorithm.default_config.copy()
        
        # Stwórz proste obrazy testowe
        input_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        input_img.save(self.test_input_path)
        master_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        master_img.save(self.test_master_path)

    def tearDown(self):
        # Usuń pliki testowe po zakończeniu
        if os.path.exists(self.test_input_path):
            os.remove(self.test_input_path)
        if os.path.exists(self.test_master_path):
            os.remove(self.test_master_path)
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)

    def run_test_with_params(self, **params):
        for key, value in params.items():
            self.algorithm.config[key] = value
        return self.algorithm.process_images(
            self.test_master_path, self.test_input_path, self.test_output_path
        )

    def load_image(self, path):
        return np.array(Image.open(path))

    def test_hue_weight_effect(self):
        """Testuje wpływ parametru hue_weight na mapowanie kolorów."""
        config_default = {"hue_weight": 3.0}
        config_low = {"hue_weight": 0.5}
        config_high = {"hue_weight": 6.0}
        result_default = self.run_test_with_params(**config_default)
        result_low = self.run_test_with_params(**config_low)
        result_high = self.run_test_with_params(**config_high)
        self.assertTrue(result_default, "Przetwarzanie z domyślnym hue_weight nie powiodło się")
        self.assertTrue(result_low, "Przetwarzanie z niskim hue_weight nie powiodło się")
        self.assertTrue(result_high, "Przetwarzanie z wysokim hue_weight nie powiodło się")
        self.assertFalse(np.array_equal(self.load_image(self.test_output_path), self.load_image(self.test_output_path)), "Wyniki dla różnych hue_weight są identyczne (default vs low)")
        self.assertFalse(np.array_equal(self.load_image(self.test_output_path), self.load_image(self.test_output_path)), "Wyniki dla różnych hue_weight są identyczne (low vs high)")

    def test_dithering_strength_effect(self):
        """Testuje wpływ parametru dithering_strength na dithering."""
        config_no_dither = {"dithering_method": "none", "dithering_strength": 0.0}
        config_low = {"dithering_method": "ordered", "dithering_strength": 2.0}
        config_high = {"dithering_method": "ordered", "dithering_strength": 8.0}
        result_no_dither = self.run_test_with_params(**config_no_dither)
        result_low = self.run_test_with_params(**config_low)
        result_high = self.run_test_with_params(**config_high)
        self.assertTrue(result_no_dither, "Przetwarzanie bez ditheringu nie powiodło się")
        self.assertTrue(result_low, "Przetwarzanie z niskim dithering_strength nie powiodło się")
        self.assertTrue(result_high, "Przetwarzanie z wysokim dithering_strength nie powiodło się")
        self.assertFalse(np.array_equal(self.load_image(self.test_output_path), self.load_image(self.test_output_path)), "Wyniki dla różnych dithering_strength są identyczne (none vs low)")
        self.assertFalse(np.array_equal(self.load_image(self.test_output_path), self.load_image(self.test_output_path)), "Wyniki dla różnych dithering_strength są identyczne (low vs high)")

    def test_edge_blur_effect(self):
        """Testuje wpływ parametrów wygładzania krawędzi."""
        config_no_blur = {"edge_blur_enabled": False, "edge_blur_radius": 0.0, "edge_blur_strength": 0.0}
        config_low = {"edge_blur_enabled": True, "edge_blur_radius": 1.0, "edge_blur_strength": 0.2}
        config_high = {"edge_blur_enabled": True, "edge_blur_radius": 2.0, "edge_blur_strength": 0.5}
        result_no_blur = self.run_test_with_params(**config_no_blur)
        result_low = self.run_test_with_params(**config_low)
        result_high = self.run_test_with_params(**config_high)
        self.assertTrue(result_no_blur, "Przetwarzanie bez wygładzania krawędzi nie powiodło się")
        self.assertTrue(result_low, "Przetwarzanie z niskim wygładzaniem krawędzi nie powiodło się")
        self.assertTrue(result_high, "Przetwarzanie z wysokim wygładzaniem krawędzi nie powiodło się")
        self.assertFalse(np.array_equal(self.load_image(self.test_output_path), self.load_image(self.test_output_path)), "Wyniki dla różnych parametrów wygładzania krawędzi są identyczne (none vs low)")
        self.assertFalse(np.array_equal(self.load_image(self.test_output_path), self.load_image(self.test_output_path)), "Wyniki dla różnych parametrów wygładzania krawędzi są identyczne (low vs high)")

    def test_preserve_extremes_effect(self):
        """Testuje wpływ zachowania skrajnych wartości (czerni i bieli)."""
        config_no_extremes = {"preserve_extremes": False, "extremes_threshold": 0}
        config_low = {"preserve_extremes": True, "extremes_threshold": 5}
        config_high = {"preserve_extremes": True, "extremes_threshold": 20}
        result_no_extremes = self.run_test_with_params(**config_no_extremes)
        result_low = self.run_test_with_params(**config_low)
        result_high = self.run_test_with_params(**config_high)
        self.assertTrue(result_no_extremes, "Przetwarzanie bez zachowania skrajnych wartości nie powiodło się")
        self.assertTrue(result_low, "Przetwarzanie z niskim progiem skrajnych wartości nie powiodło się")
        self.assertTrue(result_high, "Przetwarzanie z wysokim progiem skrajnych wartości nie powiodło się")
        self.assertFalse(np.array_equal(self.load_image(self.test_output_path), self.load_image(self.test_output_path)), "Wyniki dla różnych ustawień zachowania skrajnych wartości są identyczne (none vs low)")
        self.assertFalse(np.array_equal(self.load_image(self.test_output_path), self.load_image(self.test_output_path)), "Wyniki dla różnych ustawień zachowania skrajnych wartości są identyczne (low vs high)")

if __name__ == '__main__':
    unittest.main()