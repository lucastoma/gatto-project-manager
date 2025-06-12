import unittest
import numpy as np
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU

# GPU availability
try:
    import pyopencl as cl
    GPU_AVAILABLE = any(
        d.type == cl.device_type.GPU
        for p in cl.get_platforms() for d in p.get_devices()
    )
except Exception:
    GPU_AVAILABLE = False

@unittest.skipUnless(GPU_AVAILABLE, "OpenCL GPU not available")
class TestPreserveExtremesGPU(unittest.TestCase):
    """Test GPU behaviour for preserve_extremes parameter."""

    def setUp(self):
        self.alg = PaletteMappingAlgorithmGPU()
        self.master_path = os.path.join("uploads", "m1.tif")
        self.target_path = self.master_path
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

    def test_preserve_extremes_effect(self):
        cfg_off = {"preserve_extremes": False, "extremes_threshold": 0}
        cfg_on = {"preserve_extremes": True, "extremes_threshold": 15}

        out_off = "gpu_extremes_off.jpg"
        out_on = "gpu_extremes_on.jpg"

        self.assertTrue(self._run(out_off, **cfg_off))
        self.assertTrue(self._run(out_on, **cfg_on))

        img_off = self._img(out_off)
        img_on = self._img(out_on)

        self.assertFalse(
            np.array_equal(img_off, img_on),
            "Outputs should differ when preserve_extremes is toggled on GPU."
        )

if __name__ == "__main__":
    unittest.main()
