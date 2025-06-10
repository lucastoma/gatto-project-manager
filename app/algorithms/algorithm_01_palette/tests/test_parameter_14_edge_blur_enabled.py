"""
Parameter Test: edge_blur_enabled
Algorithm: algorithm_01_palette
Parameter ID: 14

Behavioral Test: Verify that enabling edge blending measurably changes the output
by creating more intermediate colors along sharp edges.
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from .base_test_case import BaseAlgorithmTestCase
from ..algorithm import PaletteMappingAlgorithm
from app.core.development_logger import get_logger

class TestEdgeBlurEnabled(BaseAlgorithmTestCase):

    def setUp(self):
        super().setUp()
        self.algorithm = PaletteMappingAlgorithm()
        self.logger = get_logger()

        # Create a test image with very sharp edges (checkerboard)
        # to maximize the effect of edge blending.
        checkerboard = np.zeros((64, 64, 3), dtype=np.uint8)
        checkerboard[0:32, 0:32] = [255, 0, 0]  # Red
        checkerboard[0:32, 32:64] = [0, 0, 255]  # Blue
        checkerboard[32:64, 0:32] = [0, 255, 0]  # Green
        checkerboard[32:64, 32:64] = [255, 255, 0] # Yellow
        self.target_image_path = self.create_test_image('target_checkerboard.png', arr_data=checkerboard)

        # Use the checkerboard itself as the master palette source.
        # This ensures the palette contains multiple colors (red, green, blue, yellow),
        # allowing the final mapped image to have edges that can be blended.
        self.master_image_path = self.target_image_path

    def run_and_analyze(self, **kwargs):
        """Helper to run the algorithm and return key metrics."""
        output_path = os.path.join(self.test_dir, f'result_{kwargs.get("edge_blur_enabled", "none")}.png')
        self.algorithm.process_images(
            master_path=self.master_image_path,
            target_path=self.target_image_path,
            output_path=output_path,
            **kwargs
        )
        result_array = np.array(Image.open(output_path))
        unique_colors = len(np.unique(result_array.reshape(-1, 3), axis=0))
        return {'unique_colors': unique_colors, 'output_path': output_path}

    def test_edge_blur_enabled_logic(self):
        """
        Behavioral Test for `edge_blur_enabled`.

        - Theory: Enabling edge blur should create new, intermediate colors
          along the boundaries of different color areas, thus increasing the
          total number of unique colors in the output image.
        - Verification: Compare the number of unique colors between the
          `enabled=False` and `enabled=True` states.
        """
        self.logger.info("\n--- BEHAVIORAL TEST: edge_blur_enabled ---")
        self.logger.info("Verifying that enabling the parameter increases color diversity.")

        # --- Test Case 1: Disabled (Baseline) ---
        self.logger.info("Running with edge_blur_enabled = False (Baseline)")
        result_disabled = self.run_and_analyze(edge_blur_enabled=False, num_colors=4)
        self.logger.info(f"Result (Disabled): {result_disabled['unique_colors']} unique colors.")

        # --- Test Case 2: Enabled ---
        self.logger.info("Running with edge_blur_enabled = True")
        result_enabled = self.run_and_analyze(
            edge_blur_enabled=True,
            edge_blur_radius=1.5,
            edge_blur_strength=0.5,
            num_colors=4
        )
        self.logger.info(f"Result (Enabled): {result_enabled['unique_colors']} unique colors.")

        # --- Verification ---
        self.logger.info("Verifying algorithm logic...")
        # Reactivity Check: The number of colors should be different.
        self.assertNotEqual(
            result_enabled['unique_colors'],
            result_disabled['unique_colors'],
            "FAIL: Parameter had no measurable effect (Reactivity)."
        )
        # Direction Check: Enabled should have MORE unique colors.
        self.assertGreater(
            result_enabled['unique_colors'],
            result_disabled['unique_colors'],
            "FAIL: Blending did not create more intermediate colors as expected (Direction)."
        )
        self.logger.info("✅ PASS: Enabling edge blur correctly increased color variety as per theory.")

if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)