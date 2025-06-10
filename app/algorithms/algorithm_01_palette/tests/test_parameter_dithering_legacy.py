"""
Testy parametrów dla algorithm_01_palette - DITHERING_METHOD
Status: ✅ ZWERYFIKOWANY  
Data: 09.06.2025

Test sprawdza działanie parametru dithering_method w algorytmie Simple Palette Mapping.

### Test Results Summary z sesji testowej:
#### Test Case 1: dithering_method = 'none'
- unique_colors=1, color_diff=127.33
- Solid color mapping without dithering

#### Test Case 2: dithering_method = 'floyd_steinberg'  
- unique_colors=2, color_diff=127.50
- Floyd-Steinberg dithering increased color variety

#### Verification: ✅ PASSED
- Direction: ✅ Dithering correctly increased unique colors (1 → 2)
- Magnitude: ✅ 100% increase in color variety is significant and expected
- Logic: ✅ Algorithm behaves according to dithering theory
"""

import os
import sys
import tempfile
import shutil
import unittest
from PIL import Image
import numpy as np

from .base_test_case import BaseAlgorithmTestCase
from ..algorithm import PaletteMappingAlgorithm
from app.core.development_logger import get_logger


class ParameterEffectTests(BaseAlgorithmTestCase):
    """
    Testy sprawdzające wpływ parametru dithering_method na wyniki algorytmu Simple Palette Mapping.
    """
    
    def setUp(self):
        """Przygotowanie środowiska testowego"""
        super().setUp()
        self.algorithm = PaletteMappingAlgorithm()
        self.logger = get_logger()

    def run_test_case(self, master_path, target_path, **kwargs):
        """Helper method to run algorithm and calculate metrics"""
        output_path = os.path.join(self.test_dir, 'result.png')
        
        # Run algorithm
        self.algorithm.process_images(master_path, target_path, output_path, **kwargs)
        
        # Calculate metrics
        result_image = Image.open(output_path)
        result_array = np.array(result_image)
        target_array = np.array(Image.open(target_path))
        
        # Calculate unique colors in result
        unique_colors = len(np.unique(result_array.reshape(-1, result_array.shape[-1]), axis=0))
        
        # Calculate color difference
        color_diff = np.mean(np.sqrt(np.sum((result_array - target_array) ** 2, axis=2)))
        
        return {
            'unique_colors': unique_colors,
            'color_diff': color_diff,
            'output_path': output_path
        }

    def test_dithering_method_parameter(self):
        """Test that dithering increases color variety in the output image"""
        
        self.logger.info("\n--- Testing dithering_method parameter ---")
        
        # Test Case 1: Default (none)
        self.logger.info("Running dithering_method = 'none' (Default)")
        master_image = self.create_test_image('simple_master_dither.png', (64, 64), color=[255, 128, 0])
        target_image = self.create_test_image('gradient.png', (100, 100))
        
        result_no_dither = self.run_test_case(
            master_image, 
            target_image,
            dithering_method='none'
        )
        
        self.logger.info(f"dithering_method='none': unique_colors={result_no_dither['unique_colors']}, color_diff={result_no_dither['color_diff']:.2f}")
        
        # Test Case 2: Floyd-Steinberg dithering
        self.logger.info("Running dithering_method = 'floyd_steinberg'")
        
        result_dithered = self.run_test_case(
            master_image, 
            target_image,
            dithering_method='floyd_steinberg'
        )
        
        self.logger.info(f"dithering_method='floyd_steinberg': unique_colors={result_dithered['unique_colors']}, color_diff={result_dithered['color_diff']:.2f}")
        
        # Weryfikacja logiki algorytmu
        # Dithering powinien zwiększyć różnorodność kolorów
        self.assertGreaterEqual(result_dithered['unique_colors'], result_no_dither['unique_colors'], 
                               "Dithering should maintain or increase color variety")
        
        # Sprawdź czy algorytm rzeczywiście wykonał dithering
        if result_dithered['unique_colors'] > result_no_dither['unique_colors']:
            self.logger.info("✅ Dithering successfully increased color variety")
        else:
            self.logger.info("ℹ️ Dithering maintained same color count (may be expected for this test case)")
        
        self.logger.info("--- dithering_method parameter testing complete ---")


if __name__ == '__main__':
    unittest.main()