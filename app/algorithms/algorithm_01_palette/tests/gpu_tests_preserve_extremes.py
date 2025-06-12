import unittest
import numpy as np
from PIL import Image
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Ensure root path for package import
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
class TestPreserveExtremesGPU(unittest.TestCase):
    """GPU test for preserve_extremes parameter."""

    def setUp(self):
        self.alg = PaletteMappingAlgorithmGPU()

        self.tmpdir = Path(tempfile.mkdtemp(prefix="gpu_extremes_"))

        uploads_master = PROJECT_ROOT / "uploads" / "m1.tif"
        if uploads_master.exists():
            self.master_path = str(uploads_master)
            self.target_path = self.master_path
        else:
            synth_path = self.tmpdir / "synthetic_master.tif"
            arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(synth_path)
            self.master_path = str(synth_path)
            self.target_path = self.master_path

        self.outputs = []

    def tearDown(self):
        for p in self.outputs:
            if os.path.exists(p):
                os.remove(p)
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, out_path: str, **cfg):
        self.outputs.append(out_path)
        return self.alg.process_images(self.master_path, self.target_path, out_path, **cfg)

    @staticmethod
    def _img(path):
        return np.array(Image.open(path))

    def test_preserve_extremes_effect(self):
        cfg_off = {"preserve_extremes": False, "extremes_threshold": 0}
        cfg_on = {"preserve_extremes": True, "extremes_threshold": 15}

        out_off = str(self.tmpdir / "gpu_extremes_off.jpg")
        out_on = str(self.tmpdir / "gpu_extremes_on.jpg")

        self.assertTrue(self._run(out_off, **cfg_off))
        self.assertTrue(self._run(out_on, **cfg_on))

        img_off = self._img(out_off)
        img_on = self._img(out_on)

        self.assertFalse(np.array_equal(img_off, img_on),
                         "Outputs should differ when preserve_extremes toggled on GPU")

if __name__ == "__main__":
    unittest.main()
