"""
Parameter Test: edge_blur_method
Algorithm: algorithm_01_palette
Parameter ID: 18

Behavioral Test: Verify that different blur methods produce measurably
different outputs, confirming that the method selection logic is working.
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from tests.base_test_case import BaseAlgorithmTestCase
from app.algorithms.algorithm_01_palette.algorithm import create_palette_mapping_algorithm
from app.core.development_logger import get_logger

class TestEdgeBlurMethod(BaseAlgorithmTestCase):

    def setUp(self):
        super().setUp()
        self.algorithm = create_palette_mapping_algorithm()
        self.logger = get_logger()

        # A checkerboard image is ideal for visually and numerically comparing blur patterns.
        checkerboard = np.zeros((64, 64, 3), dtype=np.uint8)
        checkerboard[0:32, 0:32] = [255, 0, 0]
        checkerboard[0:32, 32:64] = [0, 0, 255]
        self.target_image_path = self.create_test_image('target_checkerboard.png', arr_data=checkerboard)
        # Use the target checkerboard image as master palette source to provide multi-colored palette
        self.master_image_path = self.target_image_path

    def run_and_analyze(self, method):
        """Helper to run the algorithm with a specific blur method."""
        output_path = os.path.join(self.test_dir, f'result_method_{method}.png')
        self.algorithm.process_images(
            master_path=self.master_image_path,
            target_path=self.target_image_path,
            output_path=output_path,
            edge_blur_enabled=True,
            edge_blur_method=method,
            edge_blur_radius=2.0,
            edge_blur_strength=0.7,
            num_colors=4
        )
        result_image = Image.open(output_path)
        colors = result_image.getcolors(256*256)
        unique_colors = len(colors) if colors is not None else 0
        return {'image': result_image, 'unique_colors': unique_colors}

    def test_edge_blur_method_logic(self):
        """
        Behavioral Test for `edge_blur_method`.

        - Theory: The algorithm has a specific implementation for 'gaussian' and
          a fallback (uniform_filter) for any other method name. This test
          verifies that these two code paths produce different results.
        - Verification: Compare the output images from 'gaussian' and a fallback
          method (e.g., 'uniform'). The resulting images and their metrics
          (like unique color count) should differ.
        """
        self.logger.info("\n--- BEHAVIORAL TEST: edge_blur_method ---")
        self.logger.info("Verifying that different methods produce different results.")

        # --- Test Case 1: Gaussian (explicitly implemented) ---
        self.logger.info("Running with edge_blur_method = 'gaussian'")
        result_gaussian = self.run_and_analyze('gaussian')
        self.logger.info(f"Result (gaussian): {result_gaussian['unique_colors']} unique colors.")

        # --- Test Case 2: Fallback/Default (e.g., 'uniform_filter') ---
        # Any string other than 'gaussian' will trigger the fallback.
        self.logger.info("Running with edge_blur_method = 'uniform' (triggers fallback)")
        result_fallback = self.run_and_analyze('uniform')
        self.logger.info(f"Result (fallback): {result_fallback['unique_colors']} unique colors.")

        # --- Verification ---
        self.logger.info("Verifying algorithm logic (Reactivity)...")
        gaussian_array = np.array(result_gaussian['image'])
        fallback_array = np.array(result_fallback['image'])

        # Reactivity Check: The images should not be identical.
        are_arrays_equal = np.array_equal(gaussian_array, fallback_array)
        self.assertFalse(
            are_arrays_equal,
            "FAIL: 'gaussian' and fallback method produced identical images. The logic switch is not working."
        )

        # The number of unique colors might also differ, which is a good secondary check.
        self.assertNotEqual(
            result_gaussian['unique_colors'],
            result_fallback['unique_colors'],
            "FAIL: Different blur methods resulted in the same number of unique colors."
        )

        self.logger.info("✅ PASS: Different blur methods produce measurably different outputs as expected.")

if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)