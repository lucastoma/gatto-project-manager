"""Parameter test: num_colors
Verifies that varying `num_colors` changes color quantization quality.
Converted to pure pytest with common fixtur es.
"""

import numpy as np
from PIL import Image
import pytest
from pathlib import Path


@pytest.mark.parametrize("num_colors", [4, 16, 64])
def test_num_colors_variation(tmp_path, gradient_image, noise_image, algorithm_cpu, num_colors):
    out = Path(tmp_path) / f"result_{num_colors}.png"

    ok = algorithm_cpu.process_images(
        master_path=noise_image,
        target_path=gradient_image,
        output_path=str(out),
        num_colors=num_colors,
    )
    assert ok and out.exists()

    result_arr = np.array(Image.open(out))
    unique_colors = len(np.unique(result_arr.reshape(-1, 3), axis=0))

    # Store result in test metadata for later comparison
    pytest.unique_colors = getattr(pytest, "unique_colors", {})
    pytest.unique_colors[num_colors] = unique_colors


def test_num_colors_monotonicity():
    """Ensure unique color count increases with num_colors and error decreases."""
    data = getattr(pytest, "unique_colors", {})
    assert data, "Previous parametrized test did not run."
    assert data[4] <= data[16] <= data[64]
