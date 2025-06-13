import numpy as np
import pytest
from app.algorithms.algorithm_05_lab_transfer.core import LABColorTransfer
from app.algorithms.algorithm_05_lab_transfer.metrics import calculate_delta_e

class TestLABTransfer:
    @pytest.fixture
    def lab_transfer(self):
        return LABColorTransfer()

    def test_basic_transfer(self, lab_transfer):
        """Test basic LAB transfer matches mean/std of target."""
        source = np.random.rand(100, 100, 3) * 100
        target = np.random.rand(100, 100, 3) * 100
        
        result = lab_transfer.basic_lab_transfer(source, target)
        
        # Verify mean/std matches target within tolerance
        for i in range(3):
            assert np.isclose(np.mean(result[:,:,i]), np.mean(target[:,:,i]), rtol=0.01)
            assert np.isclose(np.std(result[:,:,i]), np.std(target[:,:,i]), rtol=0.01)

    def test_selective_transfer(self, lab_transfer):
        """Test selective transfer preserves source L channel."""
        source = np.random.rand(100, 100, 3) * 100
        target = np.random.rand(100, 100, 3) * 100
        
        result = lab_transfer.selective_lab_transfer(source, target)
        
        # Verify L channel unchanged
        assert np.allclose(result[:,:,0], source[:,:,0])
        # Verify a/b channels changed
        assert not np.allclose(result[:,:,1:], source[:,:,1:])

    def test_weighted_transfer(self, lab_transfer):
        """Test weighted transfer with custom channel weights."""
        source = np.random.rand(100, 100, 3) * 100
        target = np.random.rand(100, 100, 3) * 100
        weights = {'L': 0.5, 'a': 0.8, 'b': 0.2}
        
        result = lab_transfer.weighted_lab_transfer(source, target, weights)
        
        # Verify L channel is partially transferred (weight=0.5)
        assert not np.allclose(result[:,:,0], source[:,:,0])
        assert not np.allclose(result[:,:,0], lab_transfer.basic_lab_transfer(source, target)[:,:,0])
        
        # Verify a channel is heavily transferred (weight=0.8)
        assert np.isclose(np.mean(result[:,:,1]), np.mean(target[:,:,1]), rtol=0.1)
        
        # Verify b channel is minimally transferred (weight=0.2)
        assert np.isclose(np.mean(result[:,:,2]), 
                         np.mean(source[:,:,2]) * 0.8 + np.mean(target[:,:,2]) * 0.2, 
                         rtol=0.1)

    def test_adaptive_transfer(self, lab_transfer):
        """Test adaptive transfer segments by luminance."""
        # Create test image with distinct luminance regions
        source = np.zeros((100, 100, 3))
        source[:33] = 30   # Dark region
        source[33:66] = 60 # Mid region
        source[66:] = 90   # Bright region
        
        target = np.random.rand(100, 100, 3) * 100
        
        result = lab_transfer.adaptive_lab_transfer(source, target)
        
        # Verify each region was processed differently
        dark_stats = [np.mean(result[:33,:,i]) for i in range(3)]
        mid_stats = [np.mean(result[33:66,:,i]) for i in range(3)]
        bright_stats = [np.mean(result[66:,:,i]) for i in range(3)]
        
        assert not np.allclose(dark_stats, mid_stats, rtol=0.1)
        assert not np.allclose(mid_stats, bright_stats, rtol=0.1)

    def test_tile_blending(self, lab_transfer):
        """Test tile blending smooths overlaps."""
        # Create test tile with sharp edges
        tile = np.zeros((100, 100, 3))
        tile[:50] = 1.0  # Top half
        
        blended = lab_transfer.blend_tile_overlap(tile, overlap_size=10)
        
        # Verify edges are smoothed
        assert not np.allclose(blended[45:55], tile[45:55])
        # Verify center is unchanged
        assert np.allclose(blended[10:-10, 10:-10], tile[10:-10, 10:-10])

    def test_ciede2000_metric(self):
        """Test CIEDE2000 calculation matches expected behavior."""
        # Identical colors should have delta=0
        lab1 = np.array([[[50, 0, 0]]])
        lab2 = np.array([[[50, 0, 0]]])
        assert calculate_delta_e(lab1, lab2)[0,0] == 0
        
        # Different colors should have delta>0
        lab3 = np.array([[[50, 10, 10]]])
        assert calculate_delta_e(lab1, lab3)[0,0] > 0
