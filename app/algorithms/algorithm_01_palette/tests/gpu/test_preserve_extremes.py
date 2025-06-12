import numpy as np
from PIL import Image
import pytest

from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU


@pytest.mark.gpu
def test_preserve_extremes_effect(gpu, tmp_path, synthetic_image):
    alg = PaletteMappingAlgorithmGPU()
    master = synthetic_image("master_extremes.tif")

    off_cfg = dict(preserve_extremes=False, extremes_threshold=0)
    on_cfg = dict(preserve_extremes=True, extremes_threshold=15)

    out_off = tmp_path / "extremes_off.jpg"
    out_on = tmp_path / "extremes_on.jpg"

    assert alg.process_images(master, master, str(out_off), **off_cfg)
    assert alg.process_images(master, master, str(out_on), **on_cfg)

    img_off = np.array(Image.open(out_off))
    img_on = np.array(Image.open(out_on))
    assert not np.array_equal(img_off, img_on)
