"""
Algorithm 01: Palette Mapping Tests
==================================

Comprehensive test suite for the palette mapping algorithm including:
- Unit tests for core functionality
- Integration tests with file I/O
- Performance benchmarks
- Edge case validation
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm, create_palette_mapping_algorithm
from app.algorithms.algorithm_01_palette.config import PaletteMappingConfig, get_config


class TestPaletteMappingAlgorithm(unittest.TestCase):
    """Test cases for the PaletteMappingAlgorithm class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.algorithm = create_palette_mapping_algorithm()
        self.test_dir = tempfile.mkdtemp()
        
        # Create test images
        self.master_image = self._create_test_image((100, 100, 3), 'master')
        self.target_image = self._create_test_image((80, 80, 3), 'target')
        
        self.master_path = os.path.join(self.test_dir, 'master.png')
        self.target_path = os.path.join(self.test_dir, 'target.png')
        
        cv2.imwrite(self.master_path, self.master_image)
        cv2.imwrite(self.target_path, self.target_image)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_image(self, shape, image_type='random'):
        """Create a test image with known properties."""
        if image_type == 'master':
            # Create image with distinct color regions
            image = np.zeros(shape, dtype=np.uint8)
            h, w = shape[:2]
            
            # Red region
            image[:h//2, :w//2] = [0, 0, 255]
            # Green region  
            image[:h//2, w//2:] = [0, 255, 0]
            # Blue region
            image[h//2:, :w//2] = [255, 0, 0]
            # Yellow region
            image[h//2:, w//2:] = [0, 255, 255]
            
        elif image_type == 'target':
            # Create more complex image
            image = np.random.randint(0, 256, shape, dtype=np.uint8)
            
        else:
            # Random image
            image = np.random.randint(0, 256, shape, dtype=np.uint8)
        
        return image
    
    def test_algorithm_initialization(self):
        """Test algorithm initialization."""
        self.assertEqual(self.algorithm.algorithm_id, "algorithm_01_palette")
        self.assertIsNotNone(self.algorithm.logger)
        self.assertIsNotNone(self.algorithm.profiler)
        self.assertIn('k_colors', self.algorithm.default_params)
    
    def test_extract_palette(self):
        """Test palette extraction from image."""
        k_colors = 4
        palette = self.algorithm.extract_palette(self.master_image, k_colors)
        
        self.assertEqual(len(palette), k_colors)
        self.assertEqual(palette.shape, (k_colors, 3))
        self.assertTrue(np.all(palette >= 0))
        self.assertTrue(np.all(palette <= 255))
    
    def test_map_colors(self):
        """Test color mapping functionality."""
        k_colors = 4
        master_palette = self.algorithm.extract_palette(self.master_image, k_colors)
        result = self.algorithm.map_colors(self.target_image, master_palette, k_colors)
        
        self.assertEqual(result.shape, self.target_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_process_success(self):
        """Test successful end-to-end processing."""
        result_path = self.algorithm.process(self.master_path, self.target_path, k_colors=8)
        
        self.assertTrue(os.path.exists(result_path))
        
        # Verify result image
        result_image = cv2.imread(result_path)
        self.assertIsNotNone(result_image)
        self.assertEqual(len(result_image.shape), 3)
    
    def test_process_invalid_k_colors(self):
        """Test processing with invalid k_colors parameter."""
        with self.assertRaises(ValueError):
            self.algorithm.process(self.master_path, self.target_path, k_colors=2)  # Too low
        
        with self.assertRaises(ValueError):
            self.algorithm.process(self.master_path, self.target_path, k_colors=50)  # Too high
    
    def test_process_missing_files(self):
        """Test processing with missing input files."""
        with self.assertRaises(FileNotFoundError):
            self.algorithm.process("nonexistent.jpg", self.target_path)
        
        with self.assertRaises(FileNotFoundError):
            self.algorithm.process(self.master_path, "nonexistent.jpg")
    
    def test_process_corrupted_files(self):
        """Test processing with corrupted image files."""
        # Create corrupted file
        corrupted_path = os.path.join(self.test_dir, 'corrupted.jpg')
        with open(corrupted_path, 'wb') as f:
            f.write(b'not an image')
        
        with self.assertRaises(RuntimeError):
            self.algorithm.process(corrupted_path, self.target_path)
    
    def test_get_algorithm_info(self):
        """Test algorithm information retrieval."""
        info = self.algorithm.get_algorithm_info()
        
        self.assertIn('algorithm_id', info)
        self.assertIn('name', info)
        self.assertIn('version', info)
        self.assertIn('parameters', info)
        self.assertEqual(info['algorithm_id'], "algorithm_01_palette")


class TestPaletteMappingConfig(unittest.TestCase):
    """Test cases for the PaletteMappingConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = PaletteMappingConfig()
        config.validate()  # Should not raise
        
        self.assertEqual(config.k_colors, 8)
        self.assertEqual(config.random_state, 42)
        self.assertTrue(config.enable_monitoring)
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = PaletteMappingConfig(
            k_colors=12,
            n_init=5,
            max_iter=200
        )
        config.validate()  # Should not raise
    
    def test_config_validation_failures(self):
        """Test configuration validation failures."""
        # Invalid k_colors
        with self.assertRaises(ValueError):
            config = PaletteMappingConfig(k_colors=2)
            config.validate()
        
        # Invalid memory limit
        with self.assertRaises(ValueError):
            config = PaletteMappingConfig(memory_limit_mb=10)
            config.validate()
        
        # Invalid image sizes
        with self.assertRaises(ValueError):
            config = PaletteMappingConfig(min_image_size=100, max_image_size=50)
            config.validate()
    
    def test_config_presets(self):
        """Test predefined configuration presets."""
        for preset_name in ['fast', 'balanced', 'quality', 'artistic', 'photorealistic']:
            config = get_config(preset_name)
            config.validate()  # Should not raise
    
    def test_config_serialization(self):
        """Test configuration serialization to/from dict."""
        original = PaletteMappingConfig(k_colors=16, n_init=15)
        
        # Convert to dict and back
        config_dict = original.to_dict()
        restored = PaletteMappingConfig.from_dict(config_dict)
        
        self.assertEqual(original.k_colors, restored.k_colors)
        self.assertEqual(original.n_init, restored.n_init)
        self.assertEqual(original.random_state, restored.random_state)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up benchmark fixtures."""
        self.algorithm = create_palette_mapping_algorithm()
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up benchmark fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_small_image_performance(self):
        """Benchmark performance on small images."""
        import time
        
        # Create small test images
        master = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        target = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        master_path = os.path.join(self.test_dir, 'master_small.png')
        target_path = os.path.join(self.test_dir, 'target_small.png')
        
        cv2.imwrite(master_path, master)
        cv2.imwrite(target_path, target)
        
        start_time = time.time()
        result_path = self.algorithm.process(master_path, target_path, k_colors=8)
        duration = time.time() - start_time
        
        self.assertTrue(os.path.exists(result_path))
        self.assertLess(duration, 5.0)  # Should complete in under 5 seconds
    
    def test_memory_usage_validation(self):
        """Test memory usage stays within reasonable bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create medium-sized test images
        master = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        target = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        
        master_path = os.path.join(self.test_dir, 'master_medium.png')
        target_path = os.path.join(self.test_dir, 'target_medium.png')
        
        cv2.imwrite(master_path, master)
        cv2.imwrite(target_path, target)
        
        result_path = self.algorithm.process(master_path, target_path, k_colors=8)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        self.assertTrue(os.path.exists(result_path))
        self.assertLess(memory_increase, 200)  # Should not use more than 200MB extra


class TestLegacyCompatibility(unittest.TestCase):
    """Test backward compatibility with legacy API."""
    
    def setUp(self):
        """Set up compatibility test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test images
        master = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        target = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        self.master_path = os.path.join(self.test_dir, 'master.png')
        self.target_path = os.path.join(self.test_dir, 'target.png')
        
        cv2.imwrite(self.master_path, master)
        cv2.imwrite(self.target_path, target)
    
    def tearDown(self):
        """Clean up compatibility test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_legacy_function_compatibility(self):
        """Test legacy simple_palette_mapping function."""
        from app.algorithms.algorithm_01_palette.algorithm import simple_palette_mapping
        
        result_path = simple_palette_mapping(self.master_path, self.target_path, k_colors=6)
        
        self.assertTrue(os.path.exists(result_path))
        result_image = cv2.imread(result_path)
        self.assertIsNotNone(result_image)


def run_algorithm_tests():
    """Run all algorithm tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPaletteMappingAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestPaletteMappingConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmarks))
    suite.addTests(loader.loadTestsFromTestCase(TestLegacyCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_algorithm_tests()
    sys.exit(0 if success else 1)
