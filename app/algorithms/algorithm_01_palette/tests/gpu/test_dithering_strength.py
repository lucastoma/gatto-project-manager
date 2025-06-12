import numpy as np
from PIL import Image
import pytest

from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU


@pytest.mark.gpu
def test_dithering_strength_effect(gpu, tmp_path, synthetic_image):
    alg = PaletteMappingAlgorithmGPU()
    master = synthetic_image("master_dither.tif")

    out_none = tmp_path / "dither_none.jpg"
    out_high = tmp_path / "dither_high.jpg"

    # No dithering
    assert alg.process_images(master, master, str(out_none),
                              dithering_method="none", dithering_strength=0.0)

    # Ordered dithering with strong strength
    assert alg.process_images(master, master, str(out_high),
                              dithering_method="ordered", dithering_strength=8.0)

    img_none = np.array(Image.open(out_none))
    img_high = np.array(Image.open(out_high))
    assert not np.array_equal(img_none, img_high)
