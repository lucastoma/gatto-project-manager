"""
Parameter Test: edge_detection_threshold
Algorithm: algorithm_01_palette
Parameter ID: 17

Behavioral Test: Verify that a lower threshold detects more edges, leading to
more widespread blending and thus a greater number of unique intermediate colors.
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from tests.base_test_case import BaseAlgorithmTestCase
from app.algorithms.algorithm_01_palette.algorithm import create_palette_mapping_algorithm
from app.core.development_logger import get_logger

class TestEdgeDetectionThreshold(BaseAlgorithmTestCase):

    def setUp(self):
        super().setUp()
        self.algorithm = create_palette_mapping_algorithm()
        self.logger = get_logger()

        # A gradient image provides edges of varying intensity, perfect for testing a threshold.
        self.target_image_path = self.create_test_image('target_gradient.png', shape=(100, 100, 3))
        # Use the target gradient image as master palette source to provide multi-colored palette
        self.master_image_path = self.target_image_path

    def run_and_analyze(self, threshold):
        """Helper to run the algorithm with a specific threshold."""
        output_path = os.path.join(self.test_dir, f'result_threshold_{threshold}.png')
        self.algorithm.process_images(
            master_path=self.master_image_path,
            target_path=self.target_image_path,
            output_path=output_path,
            edge_blur_enabled=True,
            edge_blur_radius=1.5,
            edge_blur_strength=0.5,
            edge_detection_threshold=threshold,
            num_colors=8
        )
        result_array = np.array(Image.open(output_path))
        unique_colors = len(np.unique(result_array.reshape(-1, 3), axis=0))
        return {'unique_colors': unique_colors}

    def test_edge_detection_threshold_logic(self):
        """
        Behavioral Test for `edge_detection_threshold`.

        - Theory: A lower threshold is more sensitive and should identify more pixels
          as being part of an edge. This should cause the blending effect to be
          applied more broadly, resulting in more unique colors.
        - Verification: Compare `unique_colors` for low, default, and high
          threshold values. We expect: colors(low) >= colors(default) >= colors(high).
        """
        self.logger.info("\n--- BEHAVIORAL TEST: edge_detection_threshold ---")
        self.logger.info("Verifying that a lower threshold increases color diversity.")

        # --- Test Cases: Low, Default, High ---
        self.logger.info("Running with low threshold (10)...")
        result_low = self.run_and_analyze(10)
        self.logger.info(f"Result (t=10): {result_low['unique_colors']} unique colors.")

        self.logger.info("Running with default threshold (25)...")
        result_default = self.run_and_analyze(25)
        self.logger.info(f"Result (t=25): {result_default['unique_colors']} unique colors.")

        self.logger.info("Running with high threshold (75)...")
        result_high = self.run_and_analyze(75)
        self.logger.info(f"Result (t=75): {result_high['unique_colors']} unique colors.")

        # --- Verification ---
        self.logger.info("Verifying algorithm logic (Reactivity and Direction)...")
        # Direction Check: The number of unique colors should generally decrease as the threshold increases.
        self.assertGreaterEqual(
            result_default['unique_colors'],
            result_high['unique_colors'],
            "FAIL: Default threshold should produce >= colors than high threshold."
        )
        self.assertGreaterEqual(
            result_low['unique_colors'],
            result_default['unique_colors'],
            "FAIL: Low threshold should produce >= colors than default threshold."
        )
        # Reactivity Check: There should be a measurable difference between low and high.
        self.assertTrue(
            result_low['unique_colors'] > result_high['unique_colors'],
            "FAIL: Changing threshold from low to high had no measurable effect."
        )
        self.logger.info("✅ PASS: Adjusting threshold correctly influenced color variety as per theory.")

if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)