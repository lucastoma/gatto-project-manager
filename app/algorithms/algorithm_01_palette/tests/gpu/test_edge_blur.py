import numpy as np
from PIL import Image
import pytest
from pathlib import Path

from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU


@pytest.mark.gpu
def test_edge_blur_effect(gpu, tmp_path, synthetic_image):
    """Ensure edge_blur config alters the GPU output."""
    alg = PaletteMappingAlgorithmGPU()
    master = synthetic_image("master_edge_blur.tif")

    out_none = tmp_path / "blur_none.jpg"
    out_blur = tmp_path / "blur_on.jpg"

    base_cfg = dict(edge_blur_radius=2.0, edge_blur_strength=0.5)

    assert alg.process_images(master, master, str(out_none), edge_blur_enabled=False, **base_cfg)
    assert alg.process_images(master, master, str(out_blur), edge_blur_enabled=True, **base_cfg)

    img_none = np.array(Image.open(out_none))
    img_blur = np.array(Image.open(out_blur))
    assert not np.array_equal(img_none, img_blur)
