import unittest
import numpy as np
from PIL import Image
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU

# Detect if any OpenCL GPU device is available; skip tests otherwise
try:
    import pyopencl as cl
    GPU_AVAILABLE = any(
        device.type == cl.device_type.GPU
        for platform in cl.get_platforms()
        for device in platform.get_devices()
    )
except Exception:
    GPU_AVAILABLE = False

@unittest.skipUnless(GPU_AVAILABLE, "OpenCL GPU not available")
class TestHueWeightGPU(unittest.TestCase):
    """Test GPU behaviour for hue_weight parameter using large TIFF file."""

    def setUp(self):
        self.algorithm = PaletteMappingAlgorithmGPU()
        self.master_path = os.path.join("uploads", "m1.tif")
        self.target_path = self.master_path  # We map palette of the same image for consistency
        self.output_files = []

    def tearDown(self):
        for path in self.output_files:
            if os.path.exists(path):
                os.remove(path)

    def _run(self, output_path: str, **config):
        self.output_files.append(output_path)
        return self.algorithm.process_images(
            self.master_path,
            self.target_path,
            output_path,
            **config
        )

    @staticmethod
    def _load_img(path):
        return np.array(Image.open(path))

    def test_hue_weight_difference(self):
        low_cfg = {"hue_weight": 1.0}
        high_cfg = {"hue_weight": 5.0}

        out_low = "gpu_hue_low.jpg"
        out_high = "gpu_hue_high.jpg"

        self.assertTrue(self._run(out_low, **low_cfg))
        self.assertTrue(self._run(out_high, **high_cfg))

        img_low = self._load_img(out_low)
        img_high = self._load_img(out_high)

        self.assertFalse(
            np.array_equal(img_low, img_high),
            "Images with different hue_weight values should differ when processed on GPU."
        )

if __name__ == "__main__":
    unittest.main()
