"""
Comprehensive tests for LAB color transfer endpoints and methods.
"""
import numpy as np
import pytest
from app.algorithms.algorithm_05_lab_transfer.core import LABColorTransfer
from app.algorithms.algorithm_05_lab_transfer.metrics import calculate_delta_e, histogram_matching
import time
import os

class TestCoreMethods:
    """Expanded tests for core transfer methods."""
    @pytest.fixture
    def transfer(self):
        return LABColorTransfer()
    
    @pytest.fixture
    def test_images(self):
        """Generate various test image combinations."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        sample1_path = os.path.join(test_dir, 'test_images', 'sample1.npy')
        sample2_path = os.path.join(test_dir, 'test_images', 'sample2.npy')

        return {
            'random': (np.random.rand(100, 100, 3) * 100, np.random.rand(100, 100, 3) * 100),
            'extreme': (np.zeros((50, 50, 3)), np.ones((50, 50, 3)) * 100),
            'small': (np.random.rand(2, 2, 3) * 100, np.random.rand(2, 2, 3) * 100),
            'real_sample': (np.load(sample1_path), np.load(sample2_path))
        }
    
    def test_basic_transfer_variations(self, transfer, test_images):
        """Test basic transfer handles all image types."""
        for name, (source, target) in test_images.items():
            if name == 'real_sample':
                pytest.importorskip('numpy')  # Skip if test images not available
            result = transfer.basic_lab_transfer(source, target)
            assert result.shape == source.shape
            assert not np.allclose(result, source)  # Should change the image
    
    def test_weighted_transfer_validation(self, transfer):
        """Test weighted transfer handles invalid weights."""
        source = np.random.rand(10, 10, 3)
        target = np.random.rand(10, 10, 3)
        
        # Test partial weights
        with pytest.raises(ValueError):
            transfer.weighted_lab_transfer(source, target, {'L': 0.5})
            
        # Test invalid weight sums
        with pytest.raises(ValueError):
            transfer.weighted_lab_transfer(source, target, {'L': 2.0, 'a': -1.0, 'b': 0.0})
    
    def test_selective_transfer_edge_cases(self, transfer):
        """Test selective transfer with edge cases."""
        # Single pixel
        source = np.array([[[50, 0, 0]]])
        target = np.array([[[50, 10, 10]]])
        result = transfer.selective_lab_transfer(source, target)
        assert np.allclose(result[0,0,0], 50)  # L preserved
        assert not np.allclose(result[0,0,1:], 0)  # a/b changed
        
        # All same luminance
        source = np.full((10, 10, 3), 50)
        target = np.full((10, 10, 3), 70)
        result = transfer.selective_lab_transfer(source, target)
        assert np.allclose(result[:,:,0], 50)
    
    def test_adaptive_transfer_regions(self, transfer):
        """Test adaptive transfer properly segments regions."""
        # Create test image with clear luminance boundaries
        source = np.zeros((100, 100, 3))
        source[:30] = 30   # Dark
        source[30:70] = 60 # Medium
        source[70:] = 90   # Bright
        
        target = np.random.rand(100, 100, 3) * 100
        result = transfer.adaptive_lab_transfer(source, target)
        
        # Verify each region was processed differently
        dark = result[:30].mean(axis=(0,1))
        medium = result[30:70].mean(axis=(0,1))
        bright = result[70:].mean(axis=(0,1))
        
        assert not np.allclose(dark, medium, atol=5)
        assert not np.allclose(medium, bright, atol=5)

class TestUtilityFunctions:
    """Tests for utility functions like conversions and blending."""
    @pytest.fixture
    def transfer(self):
        return LABColorTransfer()
    
    def test_rgb_lab_conversion_accuracy(self, transfer):
        """Test RGB<->LAB conversions maintain color integrity."""
        # Known color values
        test_colors = [
            ([0, 0, 0], [0, 0, 0]),  # Black
            ([255, 255, 255], [100, 0, 0]),  # White
            ([255, 0, 0], [53.24, 80.09, 67.20]),  # Red
            ([0, 255, 0], [87.74, -86.18, 83.18]),  # Green
            ([0, 0, 255], [32.30, 79.19, -107.86])  # Blue
        ]
        
        for rgb, expected_lab in test_colors:
            rgb_array = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
            
            # Test RGB->LAB
            lab_result = transfer.rgb_to_lab_optimized(rgb_array)
            assert np.allclose(lab_result[0,0], expected_lab, atol=0.1)
            
            # Test LAB->RGB roundtrip
            rgb_roundtrip = transfer.lab_to_rgb_optimized(lab_result)
            assert np.allclose(rgb_roundtrip[0,0], rgb, atol=1)
    
    def test_tile_blending_edge_cases(self, transfer):
        """Test tile blending handles edge cases."""
        # Test small tile with large overlap
        small_tile = np.random.rand(5, 5, 3)
        blended = transfer.blend_tile_overlap(small_tile, overlap_size=3)
        assert blended.shape == small_tile.shape
        
        # Test zero overlap
        tile = np.random.rand(10, 10, 3)
        assert np.allclose(transfer.blend_tile_overlap(tile, overlap_size=0), tile)
        
        # Test full overlap (should still work)
        blended = transfer.blend_tile_overlap(tile, overlap_size=5)
        assert not np.allclose(blended, tile)
    
    def test_large_image_processing(self, transfer):
        """Test large image processing handles various sizes."""
        # Test exact tile size
        source = np.random.rand(512, 512, 3) * 100
        target = np.random.rand(512, 512, 3) * 100
        result = transfer.process_large_image(source, target, tile_size=512, overlap=32)
        assert result.shape == source.shape
        
        # Test non-multiple size
        source = np.random.rand(500, 600, 3) * 100
        target = np.random.rand(500, 600, 3) * 100
        result = transfer.process_large_image(source, target, tile_size=256, overlap=32)
        assert result.shape == source.shape

class TestMetrics:
    """Detailed validation of color difference metrics."""
    def test_ciede2000_known_values(self):
        """Test CIEDE2000 against known color difference values."""
        # Test cases from CIEDE2000 paper and standard implementations
        test_cases = [
            # Lab1, Lab2, expected delta
            ([50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425),  # Blue pair
            ([50, 3.1571, -77.2803], [50, 0, -82.7485], 2.8615),  
            ([50, 2.8361, -74.0200], [50, 0, -82.7485], 3.4412),
            ([50, -1.3802, -84.2814], [50, 0, -82.7485], 1.0000),  # Exact 1.0 diff
            ([50, -1.1848, -84.8006], [50, 0, -82.7485], 1.0000)
        ]
        
        for lab1, lab2, expected in test_cases:
            lab1_arr = np.array(lab1).reshape(1, 1, 3)
            lab2_arr = np.array(lab2).reshape(1, 1, 3)
            delta = calculate_delta_e(lab1_arr, lab2_arr)
            assert np.isclose(delta[0,0], expected, atol=0.0001)
    
    def test_histogram_matching_precision(self):
        """Test histogram matching produces expected distributions."""
        # Create test images with known histograms
        source = np.zeros((100, 100, 3))
        # Make source L-channel non-uniform, e.g., linear from 20 to 70
        source[:,:,0] = np.linspace(20, 70, 10000).reshape(100, 100)
        # source[:,:,1:] = 0 # a and b channels are already zero from np.zeros

        target = np.zeros((100, 100, 3))
        target[:,:,0] = np.linspace(0, 100, 10000).reshape(100, 100)  # Linear L for target
        # target[:,:,1:] = 0 # a and b channels are already zero from np.zeros
        
        matched = histogram_matching(source, target) # By default, matches L, a, b
        
        # Verify L channel matches target distribution
        hist_range = (0, 100)  # L-channel values are typically in [0, 100]
        source_hist = np.histogram(source[:,:,0].ravel(), bins=10, range=hist_range)[0]
        target_hist = np.histogram(target[:,:,0].ravel(), bins=10, range=hist_range)[0]
        matched_hist = np.histogram(matched[:,:,0].ravel(), bins=10, range=hist_range)[0]
        
        # Should match target histogram, not source
        assert not np.allclose(matched_hist, source_hist, atol=5)
        assert np.allclose(matched_hist, target_hist, atol=5)
    
    def test_metrics_performance(self):
        """Benchmark metrics performance on large images."""
        large1 = np.random.rand(1000, 1000, 3) * 100
        large2 = np.random.rand(1000, 1000, 3) * 100
        
        # Time CIEDE2000
        start = time.time()
        delta_map = calculate_delta_e(large1, large2)
        ciede_time = time.time() - start
        assert ciede_time < 2.0  # Should process 1MP image in <2s
        
        # Time histogram matching
        start = time.time()
        matched = histogram_matching(large1, large2)
        hist_time = time.time() - start
        assert hist_time < 1.0  # Should process 1MP image in <1s

class TestIntegration:
    """End-to-end processing tests."""
    @pytest.fixture
    def transfer(self):
        return LABColorTransfer()
    
    def test_end_to_end_processing(self, transfer):
        """Test complete workflow from RGB input to RGB output."""
        # Create test RGB images (0-255 range)
        source_rgb = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        target_rgb = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        
        # Process using all steps
        result_rgb = transfer.process_large_image(
            source_rgb, 
            target_rgb,
            method='adaptive',
            tile_size=64,
            overlap=16
        )
        
        # Verify valid output
        assert result_rgb.dtype == np.uint8
        assert np.all(result_rgb >= 0)
        assert np.all(result_rgb <= 255)
        assert not np.allclose(result_rgb, source_rgb)
    
    def test_batch_processing(self, transfer):
        """Test processing multiple source-target pairs."""
        sources = [(np.random.rand(50, 50, 3) * 255).astype(np.uint8) for _ in range(3)]
        targets = [(np.random.rand(50, 50, 3) * 255).astype(np.uint8) for _ in range(3)]
        
        results = []
        for src, tgt in zip(sources, targets):
            results.append(transfer.basic_lab_transfer(src, tgt))
        
        # Verify all processed correctly
        assert len(results) == 3
        for res in results:
            assert res.shape == (50, 50, 3)
    
    def test_error_handling(self, transfer):
        """Test proper error handling for invalid inputs."""
        # Test shape mismatch
        with pytest.raises(ValueError):
            transfer.basic_lab_transfer(
                np.random.rand(10, 10, 3),
                np.random.rand(20, 20, 3)
            )
            
        # Test invalid dtype
        with pytest.raises(ValueError):
            transfer.weighted_lab_transfer(
                np.random.rand(10, 10, 3).astype(np.float32),
                np.random.rand(10, 10, 3),
                {'L': 0.5, 'a': 0.5, 'b': 0.5}
            )
            
        # Test invalid method
        with pytest.raises(ValueError):
            transfer.process_large_image(
                np.random.rand(100, 100, 3),
                np.random.rand(100, 100, 3),
                method='invalid_method'
            )
