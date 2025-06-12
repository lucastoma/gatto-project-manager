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
class TestEdgeBlurGPU(unittest.TestCase):
    """Test GPU behaviour for edge blur parameters."""

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

    def test_edge_blur_effect(self):
        cfg_none = {"edge_blur_enabled": False, "edge_blur_radius": 0.0, "edge_blur_strength": 0.0}
        cfg_blur = {"edge_blur_enabled": True, "edge_blur_radius": 2.0, "edge_blur_strength": 0.5}

        out_none = "gpu_blur_none.jpg"
        out_blur = "gpu_blur_high.jpg"

        self.assertTrue(self._run(out_none, **cfg_none))
        self.assertTrue(self._run(out_blur, **cfg_blur))

        img_none = self._img(out_none)
        img_blur = self._img(out_blur)

        self.assertFalse(
            np.array_equal(img_none, img_blur),
            "Edge blur enabled vs disabled should result in different outputs on GPU."
        )

if __name__ == "__main__":
    unittest.main()
