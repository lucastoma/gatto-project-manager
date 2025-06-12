"""Parameter test: edge_detection_threshold
Checks that raising the threshold reduces the amount of blending (fewer unique colors).
"""
import numpy as np
from PIL import Image
from pathlib import Path
import pytest

@pytest.mark.parametrize("threshold", [0.05, 0.2, 0.5])
def test_edge_detection_threshold_effect(tmp_path, checkerboard, algorithm_cpu, threshold):
    out = Path(tmp_path) / f"out_{threshold}.png"
    algorithm_cpu.process_images(
        master_path=checkerboard,
        target_path=checkerboard,
        output_path=str(out),
        edge_blur_enabled=True,
        edge_blur_radius=1.5,
        edge_blur_strength=0.5,
        edge_detection_threshold=threshold,
        num_colors=4,
    )
    arr = np.array(Image.open(out))
    pytest._thresh_colors = getattr(pytest, "_thresh_colors", {})
    pytest._thresh_colors[threshold] = len(np.unique(arr.reshape(-1, 3), axis=0))

def test_edge_detection_threshold_inverse():
    data = getattr(pytest, "_thresh_colors", {})
    assert data, "Param loop missing"
    assert data[0.05] >= data[0.2] >= data[0.5], "Unique color count should decrease with threshold"
