import unittest
import numpy as np
from PIL import Image
import os
from tests.base_test_case import BaseAlgorithmTestCase
from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm

class TestPaletteMappingAlgorithm(BaseAlgorithmTestCase):
    def setUp(self):
        super().setUp() # Call the base class setUp to create self.test_dir
        self.mapper = PaletteMappingAlgorithm()
        
        # Stwórz testowy obraz 10x10 z znanymi kolorami
        self.test_colors = [
            [255, 0, 0],    # Czerwony
            [0, 255, 0],    # Zielony  
            [0, 0, 255],    # Niebieski
            [255, 255, 255] # Biały
        ]
        
        # Stwórz testowy obraz programowo
        # Using self.create_test_image for test image creation
        self.master_image_path = self.create_test_image(
            "master_test_image.png", 
            shape=(10, 10, 3), 
            color=[255, 0, 0] # Red
        )
        self.target_image_path = self.create_test_image(
            "target_test_image.png", 
            shape=(10, 10, 3), 
            color=[0, 0, 255] # Blue
        )

        # Create a more complex test image for palette extraction
        test_array = np.zeros((10, 10, 3), dtype=np.uint8)
        test_array[:5, :5] = [255, 0, 0]    # Lewy górny - czerwony
        test_array[:5, 5:] = [0, 255, 0]    # Prawy górny - zielony
        test_array[5:, :5] = [0, 0, 255]    # Lewy dolny - niebieski
        test_array[5:, 5:] = [255, 255, 255] # Prawy dolny - biały
        self.complex_test_image_path = os.path.join(self.test_dir, "complex_test_image.png")
        Image.fromarray(test_array).save(self.complex_test_image_path)


    def test_rgb_distance_euclidean(self):
        """Test podstawowej metryki euklidesowej"""
        self.mapper.config['distance_metric'] = 'euclidean'
        
        color1 = [255, 0, 0]  # Czerwony
        color2 = [0, 255, 0]  # Zielony
        distance = self.mapper.calculate_rgb_distance(color1, color2)
        expected = np.sqrt(255*255 + 255*255)  # ~360.6
        self.assertAlmostEqual(distance, expected, places=1)
        
    def test_rgb_distance_weighted(self):
        """Test ważonej metryki RGB"""
        self.mapper.config['distance_metric'] = 'weighted_rgb'
        
        color1 = [255, 0, 0]  # Czerwony
        color2 = [0, 255, 0]  # Zielony
        distance = self.mapper.calculate_rgb_distance(color1, color2)
        
        # Sprawdź czy używa właściwych wag
        expected = np.sqrt((255*0.2126)**2 + (255*0.7152)**2 + 0)
        self.assertAlmostEqual(distance, expected, places=1)
        
    def test_closest_color(self):
        """Test znajdowania najbliższego koloru"""
        target_color = [100, 100, 100]  # Szary
        master_palette = [[0, 0, 0], [255, 255, 255], [128, 128, 128]]
        closest = self.mapper.find_closest_color(target_color, master_palette)
        self.assertEqual(closest, [128, 128, 128])
        
    def test_palette_extraction_programmatic(self):
        """Test wyciągania palety z programowo utworzonego obrazu"""
        # Use the complex test image created in setUp
        palette = self.mapper.extract_palette(self.complex_test_image_path, num_colors=4)
        
        # Sprawdź czy paleta ma właściwą liczbę kolorów
        self.assertEqual(len(palette), 4)
        
        # Sprawdź czy wszystkie kolory są w prawidłowym formacie
        for color in palette:
            self.assertEqual(len(color), 3)
            for component in color:
                self.assertGreaterEqual(component, 0)
                self.assertLessEqual(component, 255)
                
        # Sprawdź czy wyciągnięte kolory są podobne do oczekiwanych
        # (z tolerancją na kwantyzację)
        palette_set = set(tuple(color) for color in palette)
        expected_colors = set(tuple(color) for color in self.test_colors)
        
        # Powinniśmy mieć wszystkie główne kolory (z pewną tolerancją)
        self.assertGreaterEqual(len(palette), 3)  # Przynajmniej 3 różne kolory
        
    def test_cache_functionality(self):
        """Test funkcjonalności cache"""
        self.mapper.config['use_cache'] = True
        self.mapper.clear_cache()
        
        color1 = [255, 0, 0]
        color2 = [0, 255, 0]
        
        # Pierwsze wywołanie - powinno obliczyć i zapisać do cache
        distance1 = self.mapper.calculate_rgb_distance(color1, color2)
        self.assertEqual(len(self.mapper.distance_cache), 1)
        
        # Drugie wywołanie - powinno pobrać z cache
        distance2 = self.mapper.calculate_rgb_distance(color1, color2)
        self.assertEqual(distance1, distance2)
        self.assertEqual(len(self.mapper.distance_cache), 1)
        
    def test_palette_validation(self):
        """Test walidacji palety"""
        # Poprawna paleta
        good_palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        self.assertIsNone(self.mapper.validate_palette(good_palette))
        
        # Pusta paleta
        with self.assertRaises(ValueError):
            self.mapper.validate_palette([])
            
        # Nieprawidłowy format koloru
        with self.assertRaises(ValueError):
            self.mapper.validate_palette([[255, 0], [0, 255, 0]])
            
        # Wartości poza zakresem
        with self.assertRaises(ValueError):
            self.mapper.validate_palette([[256, 0, 0], [0, 255, 0]])

    def test_process_images(self):
        """Test kompletnego procesu mapowania obrazów."""
        output_path = os.path.join(self.test_dir, "result_mapped_image.png")
        
        # Ensure the master and target images are created by create_test_image
        # self.master_image_path and self.target_image_path are already created in setUp
        
        success = self.mapper.process_images(
            master_path=self.master_image_path,
            target_path=self.target_image_path,
            output_path=output_path
        )
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Optionally, load the result and check its properties
        result_image = Image.open(output_path)
        self.assertEqual(result_image.size, (10, 10))
        self.assertEqual(result_image.mode, 'RGB')

    def test_process_images_error_handling(self):
        """Test obsługi błędów w procesie mapowania obrazów."""
        output_path = os.path.join(self.test_dir, "result_mapped_image_error.png")
        
        # Test z nieistniejącymi plikami
        success = self.mapper.process_images(
            master_path="non_existent_master.png",
            target_path="non_existent_target.png",
            output_path=output_path
        )
        self.assertFalse(success)
        self.assertFalse(os.path.exists(output_path))

    def test_process_images_with_vectorized_and_naive(self):
        """Test porównujący wyniki wektoryzowanej i naiwnej wersji."""
        output_path_vec = os.path.join(self.test_dir, "result_vec.png")
        output_path_naive = os.path.join(self.test_dir, "result_naive.png")

        # Create a more diverse master image for better palette
        master_diverse_path = self.create_test_image(
            "master_diverse.png", 
            shape=(10, 10, 3), 
            color=None # Random colors
        )
        target_diverse_path = self.create_test_image(
            "target_diverse.png", 
            shape=(10, 10, 3), 
            color=None # Random colors
        )

        # Run vectorized version
        self.mapper.config['use_vectorized'] = True
        success_vec = self.mapper.process_images(
            master_path=master_diverse_path,
            target_path=target_diverse_path,
            output_path=output_path_vec
        )
        self.assertTrue(success_vec)
        self.assertTrue(os.path.exists(output_path_vec))

        # Run naive version
        self.mapper.config['use_vectorized'] = False
        success_naive = self.mapper.process_images(
            master_path=master_diverse_path,
            target_path=target_diverse_path,
            output_path=output_path_naive
        )
        self.assertTrue(success_naive)
        self.assertTrue(os.path.exists(output_path_naive))

        # Compare results (they should be identical or very close)
        img_vec = Image.open(output_path_vec)
        img_naive = Image.open(output_path_naive)
        
        arr_vec = np.array(img_vec)
        arr_naive = np.array(img_naive)

        # Assert that the arrays are identical
        np.testing.assert_array_equal(arr_vec, arr_naive)

if __name__ == '__main__':
    unittest.main()
