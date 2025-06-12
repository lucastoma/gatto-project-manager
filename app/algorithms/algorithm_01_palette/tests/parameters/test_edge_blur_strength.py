"""Parameter test: edge_blur_strength
Checks that increasing edge_blur_strength increases blending effect (more unique colors).
"""
import numpy as np
from PIL import Image
from pathlib import Path
import pytest

@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_edge_blur_strength_effect(tmp_path, checkerboard, algorithm_cpu, strength):
    out = Path(tmp_path) / f"out_{strength}.png"
    algorithm_cpu.process_images(
        master_path=checkerboard,
        target_path=checkerboard,
        output_path=str(out),
        edge_blur_enabled=True,
        edge_blur_radius=1.5,
        edge_blur_strength=strength,
        num_colors=4,
    )
    arr = np.array(Image.open(out))
    pytest._strength_colors = getattr(pytest, "_strength_colors", {})
    pytest._strength_colors[strength] = len(np.unique(arr.reshape(-1, 3), axis=0))

def test_edge_blur_strength_monotonicity():
    data = getattr(pytest, "_strength_colors", {})
    assert data, "Param loop missing"
    assert data[0.0] <= data[0.5] <= data[1.0], "Unique color count should increase with strength"
