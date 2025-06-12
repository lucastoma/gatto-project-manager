import numpy as np
from PIL import Image
import pytest

from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU


@pytest.mark.gpu
def test_hue_weight_effect(gpu, tmp_path, synthetic_image):
    alg = PaletteMappingAlgorithmGPU()
    master = synthetic_image("master_hue.tif")

    out_low = tmp_path / "hue_low.jpg"
    out_high = tmp_path / "hue_high.jpg"

    assert alg.process_images(master, master, str(out_low), hue_weight=1.0)
    assert alg.process_images(master, master, str(out_high), hue_weight=5.0)

    img_low = np.array(Image.open(out_low))
    img_high = np.array(Image.open(out_high))
    assert not np.array_equal(img_low, img_high)
