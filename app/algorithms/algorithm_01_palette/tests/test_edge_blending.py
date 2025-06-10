"""Test module for edge blending functionality in the palette mapping algorithm."""

import unittest
import numpy as np
from PIL import Image
import os

from .base_test_case import BaseAlgorithmTestCase
from ..algorithm import PaletteMappingAlgorithm


class TestEdgeBlending(BaseAlgorithmTestCase):
    """Test cases for edge blending functionality."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.algorithm = PaletteMappingAlgorithm()
        
        # Create a test image with clear edges for blending tests
        edge_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Left half red, right half blue
        edge_image[:, :50] = [255, 0, 0]  # Red
        edge_image[:, 50:] = [0, 0, 255]  # Blue
        
        self.target_image_path = self.create_test_image('target_edges.png', arr_data=edge_image)
        self.master_image_path = self.target_image_path  # Use same image as master
    
    def test_edge_blending_enabled_vs_disabled(self):
        """Test that edge blending produces different results when enabled vs disabled."""
        # Test with edge blending disabled
        output_disabled = os.path.join(self.test_dir, 'result_no_blending.png')
        self.algorithm.process_images(
            master_path=self.master_image_path,
            target_path=self.target_image_path,
            output_path=output_disabled,
            edge_blur_enabled=False,
            num_colors=4
        )
        
        # Test with edge blending enabled
        output_enabled = os.path.join(self.test_dir, 'result_with_blending.png')
        self.algorithm.process_images(
            master_path=self.master_image_path,
            target_path=self.target_image_path,
            output_path=output_enabled,
            edge_blur_enabled=True,
            edge_blur_radius=2.0,
            edge_blur_strength=0.5,
            num_colors=4
        )
        
        # Load and compare results
        img_disabled = np.array(Image.open(output_disabled))
        img_enabled = np.array(Image.open(output_enabled))
        
        # Images should be different when edge blending is enabled
        self.assertFalse(
            np.array_equal(img_disabled, img_enabled),
            "Edge blending should produce different results when enabled"
        )
        
        # With edge blending, we should have more unique colors due to blending
        unique_disabled = len(np.unique(img_disabled.reshape(-1, 3), axis=0))
        unique_enabled = len(np.unique(img_enabled.reshape(-1, 3), axis=0))
        
        self.assertGreaterEqual(
            unique_enabled, unique_disabled,
            "Edge blending should create more or equal unique colors"
        )
    
    def test_edge_blending_parameters(self):
        """Test that edge blending parameters affect the output."""
        # Test with different blur radius values
        results = {}
        
        for radius in [0.5, 2.0, 4.0]:
            output_path = os.path.join(self.test_dir, f'result_radius_{radius}.png')
            self.algorithm.process_images(
                master_path=self.master_image_path,
                target_path=self.target_image_path,
                output_path=output_path,
                edge_blur_enabled=True,
                edge_blur_radius=radius,
                edge_blur_strength=0.5,
                num_colors=4
            )
            
            result_img = np.array(Image.open(output_path))
            unique_colors = len(np.unique(result_img.reshape(-1, 3), axis=0))
            results[radius] = unique_colors
        
        # Verify that different radius values produce different results
        self.assertTrue(
            len(set(results.values())) > 1,
            "Different blur radius values should produce different results"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)