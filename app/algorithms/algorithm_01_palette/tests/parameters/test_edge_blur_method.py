"""Parameter test: edge_blur_method
Checks that both supported methods run and produce output (no crash).
"""
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

@pytest.mark.parametrize("method", ["gaussian", "none"])
def test_edge_blur_method_runs(tmp_path, checkerboard, algorithm_cpu, method):
    out = Path(tmp_path) / f"out_{method}.png"
    ok = algorithm_cpu.process_images(
        master_path=checkerboard,
        target_path=checkerboard,
        output_path=str(out),
        edge_blur_enabled=True,
        edge_blur_radius=1.5,
        edge_blur_strength=0.5,
        edge_blur_method=method,
        num_colors=4,
    )
    assert ok and out.exists()
    arr = np.array(Image.open(out))
    assert arr.shape[2] == 3
