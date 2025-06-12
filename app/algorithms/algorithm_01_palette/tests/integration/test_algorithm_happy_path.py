"""Integration test: PaletteMappingAlgorithm happy path
Checks that the full algorithm runs end-to-end and produces a plausible output.
"""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm

def test_algorithm_happy_path(tmp_path):
    # Synthetic master: noise, target: gradient
    master = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    target = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        target[:, i, 0] = int(i * 2.55)
        target[:, i, 1] = 128
        target[:, i, 2] = 255 - int(i * 2.55)
    master_path = tmp_path / "master.png"
    target_path = tmp_path / "target.png"
    Image.fromarray(master).save(master_path)
    Image.fromarray(target).save(target_path)

    output_path = tmp_path / "result.png"
    alg = PaletteMappingAlgorithm()
    ok = alg.process_images(
        master_path=str(master_path),
        target_path=str(target_path),
        output_path=str(output_path),
        num_colors=8,
        edge_blur_enabled=True,
        edge_blur_radius=1.0,
        edge_blur_strength=0.5,
        dithering_strength=0.5,
    )
    assert ok and output_path.exists()
    result = np.array(Image.open(output_path))
    assert result.shape == (100, 100, 3)
    # At least some quantization should occur
    assert len(np.unique(result.reshape(-1, 3), axis=0)) < 100*100
