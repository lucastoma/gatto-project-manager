import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

# ----------------- GPU availability helpers -----------------
try:
    import pyopencl as cl
    def gpu_available() -> bool:
        try:
            return any(
                d.type == cl.device_type.GPU
                for p in cl.get_platforms() for d in p.get_devices()
            )
        except Exception:
            return False
except Exception:
    # pyopencl not installed or misconfigured -> no GPU
    def gpu_available() -> bool:
        return False

# ----------------- SESSION-level fixtures -----------------

@pytest.fixture(scope="session", autouse=False)
def gpu():
    """Skip entire test module if no GPU available."""
    if not gpu_available():
        pytest.skip("OpenCL GPU not available", allow_module_level=True)


# ----------------- Utility fixtures -----------------

@pytest.fixture
def synthetic_image(tmp_path):
    """Return a callable that creates and returns a synthetic RGB image path."""
    def _create(name: str = "synthetic.tif", size=(256, 256)) -> str:
        arr = (np.random.rand(size[1], size[0], 3) * 255).astype(np.uint8)
        path = Path(tmp_path) / name
        Image.fromarray(arr).save(path)
        return str(path)
    return _create

# ------------- Additional common fixtures (CPU tests) -------------

@pytest.fixture(scope="function")
def gradient_image(tmp_path):
    """Create horizontal RGB gradient, return path."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        arr[:, i, 0] = int(i * 2.55)
        arr[:, i, 1] = 128
        arr[:, i, 2] = 255 - int(i * 2.55)
    path = tmp_path / "gradient.png"
    Image.fromarray(arr).save(path)
    return str(path)

@pytest.fixture(scope="function")
def noise_image(tmp_path):
    """Random noise RGB image (200x200) with deterministic content."""
    rng = np.random.RandomState(42)
    arr = (rng.rand(200, 200, 3) * 255).astype(np.uint8)
    path = tmp_path / "noise.png"
    Image.fromarray(arr).save(path)
    return str(path)

@pytest.fixture
def checkerboard(tmp_path):
    """Create a 64×64 checkerboard image and return its path."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[0:32, 0:32] = [255, 0, 0]   # Red
    arr[0:32, 32:64] = [0, 0, 255]  # Blue
    arr[32:64, 0:32] = [0, 255, 0]  # Green
    arr[32:64, 32:64] = [255, 255, 0]  # Yellow
    path = tmp_path / "checkerboard.png"
    Image.fromarray(arr).save(path)
    return str(path)

from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm

@pytest.fixture
def algorithm_cpu():
    """Return CPU PaletteMappingAlgorithm instance."""
    return PaletteMappingAlgorithm()

# ----------------- Auto-skip for gpu marker -----------------

def pytest_runtest_setup(item):
    if 'gpu' in item.keywords and not gpu_available():
        pytest.skip("OpenCL GPU not available")
