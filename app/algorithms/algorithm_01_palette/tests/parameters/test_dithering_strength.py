"""Parameter test: dithering_strength
Checks that higher dithering_strength produces noisier (more color variance) output.
Simplified heuristic: compare mean absolute difference from non-dithered result.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest


@pytest.mark.parametrize("d_strength", [0.0, 0.5, 1.0])
def test_dithering_strength_variation(tmp_path, gradient_image, noise_image, algorithm_cpu, d_strength):
    out = Path(tmp_path) / f"out_{d_strength}.png"
    algorithm_cpu.process_images(
        master_path=noise_image,
        target_path=gradient_image,
        output_path=str(out),
        dithering_strength=d_strength,
        num_colors=8,
    )
    arr = np.array(Image.open(out))
    # Save to session data
    pytest._dither_outputs = getattr(pytest, "_dither_outputs", {})
    pytest._dither_outputs[d_strength] = arr


def test_dithering_strength_monotone():
    data = getattr(pytest, "_dither_outputs", {})
    assert data, "Param loop missing"
    # crude variance measure
    var = {k: np.var(v.astype(float)) for k, v in data.items()}
    assert var[0.0] <= var[0.5] <= var[1.0]
