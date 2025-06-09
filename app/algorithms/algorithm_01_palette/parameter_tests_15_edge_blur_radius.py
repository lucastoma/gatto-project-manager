"""
Parameter Test: edge_blur_radius
Algorithm: algorithm_01_palette
Parameter ID: 15

Behavioral Test: Verify that increasing the blur radius results in more
extensive blending, which should manifest as a greater number of
intermediate (unique) colors.
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from tests.base_test_case import BaseAlgorithmTestCase
from app.algorithms.algorithm_01_palette.algorithm import create_palette_mapping_algorithm
from app.core.development_logger import get_logger

class TestEdgeBlurRadius(BaseAlgorithmTestCase):

    def setUp(self):
        super().setUp()
        self.algorithm = create_palette_mapping_algorithm()
        self.logger = get_logger()

        # Use a striped image to have clear, parallel edges to blur.
        stripes = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(0, 64, 8):
            stripes[:, i:i+4] = [255, 0, 0]  # Red
            stripes[:, i+4:i+8] = [0, 0, 255]  # Blue
        self.target_image_path = self.create_test_image('target_stripes.png', arr_data=stripes)
        
        # Use the striped image itself as the master palette source to ensure a multi-color palette.
        self.master_image_path = self.target_image_path

    def run_and_analyze(self, radius):
        """Helper to run the algorithm with a specific radius."""
        output_path = os.path.join(self.test_dir, f'result_radius_{radius}.png')
        self.algorithm.process_images(
            master_path=self.master_image_path,
            target_path=self.target_image_path,
            output_path=output_path,
            edge_blur_enabled=True,
            edge_blur_radius=radius,
            edge_blur_strength=0.5, # Keep strength constant
            num_colors=4
        )
        result_array = np.array(Image.open(output_path))
        unique_colors = len(np.unique(result_array.reshape(-1, 3), axis=0))
        return {'unique_colors': unique_colors}

    def test_edge_blur_radius_logic(self):
        """
        Behavioral Test for `edge_blur_radius`.

        - Theory: A larger blur radius should affect a wider area around an
          edge, leading to more gradual transitions and thus a greater number
          of unique intermediate colors.
        - Verification: Compare `unique_colors` for small, medium, and large
          radius values. We expect: colors(large) >= colors(medium) >= colors(small).
        """
        self.logger.info("\n--- BEHAVIORAL TEST: edge_blur_radius ---")
        self.logger.info("Verifying that a larger radius increases color diversity.")

        # --- Test Cases: Small, Default, Large ---
        self.logger.info("Running with small radius (0.5)...")
        result_small = self.run_and_analyze(0.5)
        self.logger.info(f"Result (r=0.5): {result_small['unique_colors']} unique colors.")

        self.logger.info("Running with default radius (1.5)...")
        result_default = self.run_and_analyze(1.5)
        self.logger.info(f"Result (r=1.5): {result_default['unique_colors']} unique colors.")

        self.logger.info("Running with large radius (4.0)...")
        result_large = self.run_and_analyze(4.0)
        self.logger.info(f"Result (r=4.0): {result_large['unique_colors']} unique colors.")

        # --- Verification ---
        self.logger.info("Verifying algorithm logic (Reactivity and Direction)...")
        # Direction Check: The number of unique colors should generally increase with the radius.
        self.assertGreaterEqual(
            result_default['unique_colors'],
            result_small['unique_colors'],
            "FAIL: Default radius should produce >= colors than small radius."
        )
        self.assertGreaterEqual(
            result_large['unique_colors'],
            result_default['unique_colors'],
            "FAIL: Large radius should produce >= colors than default radius."
        )
        # Reactivity Check: At least one of the steps should show a change.
        self.assertTrue(
            result_large['unique_colors'] > result_small['unique_colors'],
            "FAIL: Increasing radius from small to large had no measurable effect."
        )
        self.logger.info("✅ PASS: Increasing blur radius correctly increased color variety as per theory.")

if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)