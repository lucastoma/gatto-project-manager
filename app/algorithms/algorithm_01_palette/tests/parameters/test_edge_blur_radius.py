"""Parameter test: edge_blur_radius
Checks that increasing edge_blur_radius increases the number of unique colors (smoother blending).
"""
import numpy as np
from PIL import Image
from pathlib import Path
import pytest

@pytest.mark.parametrize("radius", [0.0, 1.0, 2.0])
def test_edge_blur_radius_effect(tmp_path, checkerboard, algorithm_cpu, radius):
    out = Path(tmp_path) / f"out_{radius}.png"
    algorithm_cpu.process_images(
        master_path=checkerboard,
        target_path=checkerboard,
        output_path=str(out),
        edge_blur_enabled=True,
        edge_blur_radius=radius,
        edge_blur_strength=0.5,
        num_colors=4,
    )
    arr = np.array(Image.open(out))
    pytest._radius_colors = getattr(pytest, "_radius_colors", {})
    pytest._radius_colors[radius] = len(np.unique(arr.reshape(-1, 3), axis=0))

def test_edge_blur_radius_monotonicity():
    data = getattr(pytest, "_radius_colors", {})
    assert data, "Param loop missing"
    assert data[0.0] <= data[1.0] <= data[2.0], "Unique color count should increase with radius"
