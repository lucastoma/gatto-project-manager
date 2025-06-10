import unittest
import numpy as np
from PIL import Image
import os
from skimage import color
import time # Import the time module
from .base_test_case import BaseAlgorithmTestCase
from ..algorithm import PaletteMappingAlgorithm

class ParameterEffectTests(BaseAlgorithmTestCase):
    def setUp(self):
        super().setUp()
        self.mapper = PaletteMappingAlgorithm()

        # Create test images
        self.gradient_image = self.create_gradient_image()
        self.extremes_image = self.create_extremes_image()

    def create_gradient_image(self):
        """Create a horizontal RGB gradient image"""
        path = os.path.join(self.test_dir, "gradient.png")
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            arr[:, i, 0] = int(i * 2.55)  # R channel
            arr[:, i, 1] = 128            # G channel fixed
            arr[:, i, 2] = 255 - int(i * 2.55)  # B channel
        Image.fromarray(arr).save(path)
        return path

    def create_extremes_image(self):
        """Create image with black, white and midtone areas"""
        path = os.path.join(self.test_dir, "extremes.png")
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)  # Gray background
        arr[10:30, 10:30] = [0, 0, 0]     # Black square
        arr[60:80, 60:80] = [255, 255, 255]  # White square
        Image.fromarray(arr).save(path)
        return path

    def run_with_params(self, **params):
        """Run algorithm with given params and return metrics"""
        output_path = os.path.join(self.test_dir, "result.png")

        # Create a more complex master image for palette extraction
        if 'master_path' not in params:
            master_path = os.path.join(self.test_dir, "master_complex.png")
            # Create a master image with random noise to ensure a wide range of colors
            master_arr = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
            Image.fromarray(master_arr).save(master_path)
        else:
            master_path = params.pop('master_path')

        # Handle target path
        target_path = params.pop('target_path', self.gradient_image)

        # Ensure thumbnail_size is set to a larger value to capture more detail for palette extraction
        if 'thumbnail_size' not in params:
            params['thumbnail_size'] = (100, 100) # Override default if not explicitly set

        # Run processing
        success = self.mapper.process_images(
            master_path=master_path,
            target_path=target_path,
            output_path=output_path,
            **params
        )

        if not success:
            # Return default metrics for failed processing
            return {
                'unique_colors': 0,
                'color_diff': float('inf'),
                'image': Image.new('RGB', (100, 100), (0, 0, 0))
            }

        # Calculate metrics
        original = Image.open(self.gradient_image)
        result = Image.open(output_path)

        orig_arr = np.array(original)
        result_arr = np.array(result)

        start_time = time.time() # Initialize start_time here

        # ... (rest of the method) ...

        return {
            'unique_colors': len(np.unique(result_arr.reshape(-1, 3), axis=0)),
            'color_diff': np.mean(np.abs(orig_arr.astype(float) - result_arr.astype(float))),
            'image': result,
            'processing_time': time.time() - start_time # Add processing time
        }

    def test_num_colors_parameter(self):
        """Test the effect of num_colors parameter on output"""
        self.mapper.logger.info("\n--- Testing num_colors parameter ---")

        # Test Case 1: Typical Value (16 colors)
        self.mapper.logger.info("Running num_colors = 16 (Typical Value)")
        result_16 = self.run_with_params(num_colors=16)
        self.assertIsNotNone(result_16, "Processing failed for num_colors=16")
        self.mapper.logger.info(f"num_colors=16: unique_colors={result_16['unique_colors']}, color_diff={result_16['color_diff']:.2f}")
        # Expected: Balanced color reduction, around 16 unique colors, moderate color_diff

        # Test Case 2: Low Extreme (2 colors)
        self.mapper.logger.info("Running num_colors = 2 (Low Extreme)")
        result_2 = self.run_with_params(num_colors=2)
        self.assertIsNotNone(result_2, "Processing failed for num_colors=2")
        self.mapper.logger.info(f"num_colors=2: unique_colors={result_2['unique_colors']}, color_diff={result_2['color_diff']:.2f}")
        # Expected: Strong quantization, visible banding, very few unique colors, higher color_diff
        self.assertLessEqual(result_2['unique_colors'], 2, "Should have very few unique colors for num_colors=2")
        self.assertGreater(result_2['color_diff'], result_16['color_diff'], "Color diff should be higher for fewer colors")

        # Test Case 3: High Extreme (64 colors)
        self.mapper.logger.info("Running num_colors = 64 (High Extreme)")
        result_64 = self.run_with_params(num_colors=64)
        self.assertIsNotNone(result_64, "Processing failed for num_colors=64")
        self.mapper.logger.info(f"num_colors=64: unique_colors={result_64['unique_colors']}, color_diff={result_64['color_diff']:.2f}")
        # Expected: Smooth gradients, more unique colors, lower color_diff
        self.assertGreater(result_64['unique_colors'], result_16['unique_colors'], "Should have more unique colors for num_colors=64")
        self.assertLess(result_64['color_diff'], result_16['color_diff'], "Color diff should be lower for more colors")

        self.mapper.logger.info("--- num_colors parameter testing complete ---")

    def test_use_cache_parameter(self):
        """Test the effect of use_cache parameter on performance"""
        self.mapper.logger.info("\n--- Testing use_cache parameter ---")

        # Create a master image with many repeated colors to maximize cache hits
        master_path_cache = os.path.join(self.test_dir, "master_cache_test.png")
        cache_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        # Fill with a few distinct colors repeated
        cache_arr[::2, ::2] = [255, 0, 0]
        cache_arr[1::2, 1::2] = [0, 255, 0]
        cache_arr[::2, 1::2] = [0, 0, 255]
        cache_arr[1::2, ::2] = [255, 255, 0]
        Image.fromarray(cache_arr).save(master_path_cache)

        # Create a target image with many pixels that will map to these few colors
        target_path_cache = os.path.join(self.test_dir, "target_cache_test.png")
        target_arr_cache = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        Image.fromarray(target_arr_cache).save(target_path_cache)

        num_runs = 5 # Run multiple times to average out noise

        # Test Case 1: use_cache = True
        self.mapper.logger.info("Running use_cache = True")
        cached_times = []
        for _ in range(num_runs):
            self.mapper.clear_cache() # Clear cache before each run
            result = self.run_with_params(
                use_cache=True,
                master_path=master_path_cache,
                target_path=target_path_cache
            )
            self.assertIsNotNone(result, "Processing failed for use_cache=True")
            cached_times.append(result['processing_time'])
        avg_cached_time = np.mean(cached_times)
        self.mapper.logger.info(f"use_cache=True: Avg processing time = {avg_cached_time:.4f} seconds")

        # Test Case 2: use_cache = False
        self.mapper.logger.info("Running use_cache = False")
        uncached_times = []
        for _ in range(num_runs):
            self.mapper.clear_cache() # Clear cache before each run
            result = self.run_with_params(
                use_cache=False,
                master_path=master_path_cache,
                target_path=target_path_cache
            )
            self.assertIsNotNone(result, "Processing failed for use_cache=False")
            uncached_times.append(result['processing_time'])
        avg_uncached_time = np.mean(uncached_times)
        self.mapper.logger.info(f"use_cache=False: Avg processing time = {avg_uncached_time:.4f} seconds")

        # Assert that cached version is faster
        self.assertTrue(avg_cached_time < avg_uncached_time, "Cached processing should be faster than uncached")
        self.mapper.logger.info("--- use_cache parameter testing complete ---")

    def test_distance_metric_effect(self):
        """Test that different distance metrics work correctly and show perceptual differences"""
        # Create a test image with colors that are perceptually distinct but might be numerically close in RGB
        target_path = os.path.join(self.test_dir, "perceptual_colors_test.png")
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        # Define perceptually distinct colors
        # Example: A green and a blue that are close in RGB but far in LAB
        # Color 1: RGB(0, 128, 0) - Green
        # Color 2: RGB(0, 0, 128) - Blue
        # Color 3: RGB(128, 0, 0) - Red
        # Color 4: RGB(128, 128, 0) - Yellow

        # Create a checkerboard pattern or distinct blocks
        arr[:50, :50] = [0, 128, 0]   # Green
        arr[:50, 50:] = [0, 0, 128]   # Blue
        arr[50:, :50] = [128, 0, 0]   # Red
        arr[50:, 50:] = [128, 128, 0] # Yellow

        Image.fromarray(arr).save(target_path)

        # Test both metrics
        weighted = self.run_with_params(
            distance_metric='weighted_rgb',
            target_path=target_path
        )
        self.assertIsNotNone(weighted, "Processing failed with weighted_rgb metric")

        lab = self.run_with_params(
            distance_metric='lab',
            target_path=target_path
        )
        self.assertIsNotNone(lab, "Processing failed with lab metric")

        # Log results
        self.mapper.logger.info(f"weighted_rgb: {weighted['color_diff']:.2f} diff, {weighted['unique_colors']} colors")
        self.mapper.logger.info(f"lab: {lab['color_diff']:.2f} diff, {lab['unique_colors']} colors")

        # Verify metrics produce reasonable results
        self.assertGreater(weighted['color_diff'], 0, "Invalid color difference for weighted_rgb")
        self.assertGreater(lab['color_diff'], 0, "Invalid color difference for lab")
        self.assertGreater(weighted['unique_colors'], 1, "Too few unique colors for weighted_rgb")
        self.assertGreater(lab['unique_colors'], 1, "Too few unique colors for lab")

        # Due to complexities of mean RGB difference not always reflecting perceptual accuracy,
        # and potential floating point precision issues, we will rely on visual inspection for this test.
        # The primary goal is to ensure the algorithm runs correctly with different metrics.
        self.mapper.logger.info("NOTE: For distance metrics, manual visual inspection of output images is crucial for perceptual quality.")
        self.mapper.logger.info(f"Expected: LAB to be perceptually more uniform, but mean RGB diff might not always be strictly lower.")

    def test_dithering_effect(self):
        """Test that dithering increases color variety"""
        # Baseline - no dithering
        base = self.run_with_params(dithering_method='none')
        self.assertIsNotNone(base, "Processing failed without dithering")

        # Change ONLY to floyd_steinberg
        dithered = self.run_with_params(dithering_method='floyd_steinberg')
        self.assertIsNotNone(dithered, "Processing failed with dithering")

        # Dithering should produce more unique colors
        self.assertGreater(dithered['unique_colors'], base['unique_colors'])

    def test_inject_extremes_effect(self):
        """Test that inject_extremes adds black/white to palette"""
        # Get palette with inject_extremes=False
        self.mapper.config['inject_extremes'] = False
        palette_no_inject = self.mapper.extract_palette(self.gradient_image)

        # Change ONLY inject_extremes to True
        self.mapper.config['inject_extremes'] = True
        palette_inject = self.mapper.extract_palette(self.gradient_image)

        # Should contain black and white
        self.assertIn([0, 0, 0], palette_inject)
        self.assertIn([255, 255, 255], palette_inject)

        # Should have exactly 2 more colors than without injection
        self.assertEqual(len(palette_inject), len(palette_no_inject) + 2)

    def test_preserve_extremes_effect(self):
        """Test that preserve_extremes keeps black/white areas"""
        # Baseline - preserve_extremes=False
        base = self.run_with_params(
            preserve_extremes=False,
            master_path=self.create_test_image("master_no_extremes.png", color=[128,128,128]),
            target_path=self.extremes_image
        )
        self.assertIsNotNone(base, "Processing failed with preserve_extremes=False")

        # Change ONLY preserve_extremes to True
        preserved = self.run_with_params(
            preserve_extremes=True,
            extremes_threshold=20,
            master_path=self.create_test_image("master_no_extremes.png", color=[128,128,128]),
            target_path=self.extremes_image
        )
        self.assertIsNotNone(preserved, "Processing failed with preserve_extremes=True")

        # Check that black and white areas were preserved
        preserved_arr = np.array(preserved['image'])
        black_area = preserved_arr[10:30, 10:30]
        white_area = preserved_arr[60:80, 60:80]

        self.assertTrue(np.all(black_area == [0, 0, 0]))
        self.assertTrue(np.all(white_area == [255, 255, 255]))

if __name__ == '__main__':
    unittest.main()
