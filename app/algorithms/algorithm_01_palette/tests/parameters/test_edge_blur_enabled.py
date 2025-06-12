"""Behavioral parameter test: edge_blur_enabled
Checks that enabling edge blur increases color diversity on sharp-edged images.
Converted from legacy unittest to pure pytest.
"""

import numpy as np
from PIL import Image
import pytest
from pathlib import Path

from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm


@pytest.fixture(scope="function")
def checkerboard(tmp_path):
    """Return master/target checkerboard image path (64×64, 4 colors)."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[0:32, 0:32] = [255, 0, 0]   # Red
    arr[0:32, 32:64] = [0, 0, 255]  # Blue
    arr[32:64, 0:32] = [0, 255, 0]  # Green
    arr[32:64, 32:64] = [255, 255, 0]  # Yellow
    img_path = Path(tmp_path) / "checkerboard.png"
    Image.fromarray(arr).save(img_path)
    return str(img_path)


def _unique_colors(img_path):
    return len(np.unique(np.array(Image.open(img_path)).reshape(-1, 3), axis=0))


@pytest.mark.parametrize("edge_blur_enabled", [False, True])
def test_edge_blur_enabled_effect(tmp_path, checkerboard, edge_blur_enabled):
    alg = PaletteMappingAlgorithm()
    out = Path(tmp_path) / f"result_{edge_blur_enabled}.png"

    alg.process_images(
        master_path=checkerboard,
        target_path=checkerboard,
        output_path=str(out),
        edge_blur_enabled=edge_blur_enabled,
        edge_blur_radius=1.5,
        edge_blur_strength=0.5,
        num_colors=4,
    )

    assert out.exists()

    # Less strict assertion in parameterized loop – compute later
    # Return colors for comparison outside param loop


def test_edge_blur_enabled_increases_color_diversity(tmp_path, checkerboard):
    alg = PaletteMappingAlgorithm()
    out_disabled = Path(tmp_path) / "disabled.png"
    out_enabled = Path(tmp_path) / "enabled.png"

    alg.process_images(checkerboard, checkerboard, str(out_disabled),
                       edge_blur_enabled=False, num_colors=4)
    alg.process_images(checkerboard, checkerboard, str(out_enabled),
                       edge_blur_enabled=True, edge_blur_radius=1.5, edge_blur_strength=0.5, num_colors=4)

    colors_disabled = _unique_colors(out_disabled)
    colors_enabled = _unique_colors(out_enabled)

    assert colors_enabled != colors_disabled, "Parameter had no effect on color variety"
    assert colors_enabled > colors_disabled, "Enabling blur should increase unique colors"
