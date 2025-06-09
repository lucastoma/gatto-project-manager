"""
Parameter Test: edge_blur_strength
Algorithm: algorithm_01_palette
Parameter ID: 16

Behavioral Test: Verify that increasing the blur strength results in more
intense color mixing, leading to a greater number of unique intermediate colors.
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from tests.base_test_case import BaseAlgorithmTestCase
from app.algorithms.algorithm_01_palette.algorithm import create_palette_mapping_algorithm
from app.core.development_logger import get_logger

class TestEdgeBlurStrength(BaseAlgorithmTestCase):

    def setUp(self):
        super().setUp()
        self.algorithm = create_palette_mapping_algorithm()
        self.logger = get_logger()

        # A checkerboard image provides sharp edges for blending.
        checkerboard = np.zeros((64, 64, 3), dtype=np.uint8)
        checkerboard[0:32, 0:32] = [255, 0, 0] # Red
        checkerboard[0:32, 32:64] = [0, 0, 255] # Blue
        self.target_image_path = self.create_test_image('target_checkerboard.png', arr_data=checkerboard)

        # CRITICAL FIX: Use the multi-color target image as the master
        # to ensure a valid, multi-color palette is generated.
        self.master_image_path = self.target_image_path

    def run_and_analyze(self, strength):
        """Helper to run the algorithm with a specific strength."""
        output_path = os.path.join(self.test_dir, f'result_strength_{strength}.png')
        self.algorithm.process_images(
            master_path=self.master_image_path,
            target_path=self.target_image_path,
            output_path=output_path,
            edge_blur_enabled=True,
            edge_blur_radius=2.0, # Keep radius constant and reasonably large
            edge_blur_strength=strength,
            num_colors=4
        )
        result_array = np.array(Image.open(output_path))
        unique_colors = len(np.unique(result_array.reshape(-1, 3), axis=0))
        return {'unique_colors': unique_colors}

    def test_edge_blur_strength_logic(self):
        """
        Behavioral Test for `edge_blur_strength`.

        - Theory: Higher strength values should increase the influence of the
          blurred color in the final mix, creating more diverse intermediate
          colors.
        - Verification: Compare `unique_colors` for weak, medium, and strong
          strength values. We expect: colors(strong) >= colors(medium) >= colors(weak).
        """
        self.logger.info("\n--- BEHAVIORAL TEST: edge_blur_strength ---")
        self.logger.info("Verifying that a higher strength increases color diversity.")

        # --- Test Cases: Weak, Default, Strong ---
        self.logger.info("Running with weak strength (0.1)...")
        result_weak = self.run_and_analyze(0.1)
        self.logger.info(f"Result (s=0.1): {result_weak['unique_colors']} unique colors.")

        self.logger.info("Running with default strength (0.5)...")
        result_default = self.run_and_analyze(0.5)
        self.logger.info(f"Result (s=0.5): {result_default['unique_colors']} unique colors.")

        self.logger.info("Running with strong strength (0.9)...")
        result_strong = self.run_and_analyze(0.9)
        self.logger.info(f"Result (s=0.9): {result_strong['unique_colors']} unique colors.")

        # --- Verification ---
        self.logger.info("Verifying algorithm logic (Reactivity and Direction)...")
        # Direction Check: The number of unique colors should generally increase with strength.
        self.assertGreaterEqual(
            result_default['unique_colors'],
            result_weak['unique_colors'],
            "FAIL: Default strength should produce >= colors than weak strength."
        )
        self.assertGreaterEqual(
            result_strong['unique_colors'],
            result_default['unique_colors'],
            "FAIL: Strong strength should produce >= colors than default strength."
        )
        # Reactivity Check: There should be a measurable difference between weak and strong.
        self.assertGreater(
            result_strong['unique_colors'],
            result_weak['unique_colors'],
            "FAIL: Increasing strength from weak to strong had no measurable effect."
        )
        self.logger.info("✅ PASS: Increasing blur strength correctly increased color variety as per theory.")

if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)