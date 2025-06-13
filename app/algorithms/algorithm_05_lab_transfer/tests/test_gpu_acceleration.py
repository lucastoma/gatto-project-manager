"""
Tests for OpenCL GPU acceleration.
"""
import numpy as np
import pytest
import time

from app.algorithms.algorithm_05_lab_transfer.config import LABTransferConfig
from app.algorithms.algorithm_05_lab_transfer.core import LABColorTransfer
from app.algorithms.algorithm_05_lab_transfer.gpu_core import LABColorTransferGPU

try:
    import pyopencl
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Mark all tests in this module to be skipped if pyopencl is not installed
pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="pyopencl not found, skipping GPU tests")

@pytest.fixture
def sample_images():
    """Provide sample source and target images for testing."""
    source = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    target = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return source, target

class TestGPUAcceleration:

    def test_gpu_cpu_equivalence(self, sample_images):
        """Verify that GPU and CPU results are numerically close."""
        source_rgb, target_rgb = sample_images

        # Run with CPU
        config_cpu = LABTransferConfig(use_gpu=False)
        transfer_cpu = LABColorTransfer(config_cpu)
        source_lab_cpu = transfer_cpu.rgb_to_lab_optimized(source_rgb)
        target_lab_cpu = transfer_cpu.rgb_to_lab_optimized(target_rgb)
        result_cpu = transfer_cpu.basic_lab_transfer(source_lab_cpu, target_lab_cpu)

        # Run with GPU
        config_gpu = LABTransferConfig(use_gpu=True)
        transfer_gpu = LABColorTransfer(config_gpu)
        
        # Check if GPU was actually initialized
        if not transfer_gpu.gpu_transfer:
            pytest.skip("GPU context not available, cannot run equivalence test.")

        source_lab_gpu = transfer_gpu.rgb_to_lab_optimized(source_rgb)
        target_lab_gpu = transfer_gpu.rgb_to_lab_optimized(target_rgb)
        result_gpu = transfer_gpu.basic_lab_transfer(source_lab_gpu, target_lab_gpu)

        # Compare results
        assert np.allclose(result_cpu, result_gpu, atol=1e-4), \
            "GPU and CPU results should be nearly identical."

    def test_fallback_to_cpu(self, sample_images, monkeypatch):
        """Test that processing falls back to CPU if GPU init fails."""
        # Mock the LABColorTransferGPU.__init__ to simulate an initialization failure
        def mock_init_fails(self, *args, **kwargs):
            raise RuntimeError("Simulated GPU initialization failure")
        monkeypatch.setattr(LABColorTransferGPU, '__init__', mock_init_fails)

        source_rgb, target_rgb = sample_images
        config_gpu = LABTransferConfig(use_gpu=True)
        transfer = LABColorTransfer(config_gpu)

        # Ensure it fell back
        assert transfer.gpu_transfer is None, "Should have fallen back to CPU."

        # Ensure it still processes correctly on the CPU
        source_lab = transfer.rgb_to_lab_optimized(source_rgb)
        target_lab = transfer.rgb_to_lab_optimized(target_rgb)
        result = transfer.basic_lab_transfer(source_lab, target_lab)
        assert result is not None
        assert result.shape == source_lab.shape
