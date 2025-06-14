import pytest
import numpy as np
from app.algorithms.algorithm_05_lab_transfer.core import LABColorTransfer
from app.algorithms.algorithm_05_lab_transfer.config import LABTransferConfig

@pytest.fixture(scope="session")
def sample_images():
    """Provides a pair of sample RGB images for testing."""
    source_rgb = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    target_rgb = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    return source_rgb, target_rgb

@pytest.fixture(scope="module")
def cpu_transfer_instance():
    """Provides a LABColorTransfer instance forced to use CPU."""
    config = LABTransferConfig(use_gpu=False)
    return LABColorTransfer(config=config)

@pytest.fixture(scope="module")
def gpu_transfer_instance():
    """Provides a LABColorTransfer instance in strict GPU mode.

    If GPU initialization fails, the test using this fixture will be skipped.
    """
    try:
        config = LABTransferConfig(use_gpu=True)
        instance = LABColorTransfer(config=config, strict_gpu=True)
        if not instance.gpu_transfer:
             pytest.skip("GPU not available or initialization failed, skipping strict GPU test.")
        return instance
    except RuntimeError as e:
        pytest.skip(f"Skipping GPU test due to initialization failure: {e}")
