import unittest
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# Add project root to sys.path for package import
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))
from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU

try:
    import pyopencl as cl
    GPU_AVAILABLE = any(
        d.type == cl.device_type.GPU
        for p in cl.get_platforms() for d in p.get_devices()
    )
except Exception:
    GPU_AVAILABLE = False

@unittest.skipUnless(GPU_AVAILABLE, "OpenCL GPU not available")
class TestDitheringStrengthGPU(unittest.TestCase):
    """GPU test for dithering_strength parameter."""

    def setUp(self):
        self.alg = PaletteMappingAlgorithmGPU()
        self.master_path = os.path.join(str(PROJECT_ROOT), "uploads", "m1.tif")
        self.target_path = self.master_path
        if not os.path.exists(self.master_path):
            self.skipTest("Large test image uploads/m1.tif not found")
        self.tmp = []

    def tearDown(self):
        for p in self.tmp:
            if os.path.exists(p):
                os.remove(p)

    def _run(self, out_path: str, **cfg):
        self.tmp.append(out_path)
        return self.alg.process_images(self.master_path, self.target_path, out_path, **cfg)

    @staticmethod
    def _img(path):
        return np.array(Image.open(path))

    def test_dithering_strength_difference(self):
        cfg_none = {"dithering_method": "none", "dithering_strength": 0.0}
        cfg_high = {"dithering_method": "ordered", "dithering_strength": 8.0}

        out_none = "gpu_dither_none.jpg"
        out_high = "gpu_dither_high.jpg"

        self.assertTrue(self._run(out_none, **cfg_none))
        self.assertTrue(self._run(out_high, **cfg_high))

        img_none = self._img(out_none)
        img_high = self._img(out_high)

        self.assertFalse(np.array_equal(img_none, img_high),
                         "Images should differ with different dithering_strength on GPU")

if __name__ == "__main__":
    unittest.main()
