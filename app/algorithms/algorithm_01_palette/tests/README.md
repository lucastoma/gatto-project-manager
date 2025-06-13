# Palette Mapping Algorithm – Test Suite (v1.3)

**Last updated:** 2025-06-13  
**Test runner:** Pytest

---

## Folder structure

- `integration/` – end-to-end functional tests  
- `parameters/` – CPU parameter unit tests  
- `gpu/` – GPU-specific behavioural tests (tagged `@pytest.mark.gpu`)  
- `conftest.py` – shared fixtures & GPU skip helper  
- `logs/`, `reports/` – generated output artefacts (git-ignored)

---

## Quick start

```bash
# run every test (CPU + GPU if available)
python -m pytest

# skip GPU tests explicitly
python -m pytest -m "not gpu"

# run only GPU tests (requires OpenCL GPU)
python -m pytest -m gpu

# run a single test file
python -m pytest tests/parameters/test_num_colors.py
```

---

## Parameter coverage (CPU – `tests/parameters`)

| # | Parameter | Test file | Status |
|---|-----------|-----------|--------|
| 01 | `num_colors` | `test_num_colors.py` | ✅ |
| 02 | `distance_cache_enabled` | `test_distance_cache.py` | ✅ |
| 03 | `dithering_strength` | `test_dithering_strength.py` | ✅ |
| 04 | `edge_blur_enabled` | `test_edge_blur_enabled.py` | ✅ |
| 05 | `edge_blur_radius` | `test_edge_blur_radius.py` | ✅ |
| 06 | `edge_blur_strength` | `test_edge_blur_strength.py` | ✅ |
| 07 | `edge_detection_threshold` | `test_edge_detection_threshold.py` | ✅ |
| 08 | `edge_blur_method` | `test_edge_blur_method.py` | ✅ |

**Planned / missing CPU parameter tests**

- `distance_metric`
- `preprocess`
- `thumbnail_size`
- `use_vectorized`
- `inject_extremes`
- `preserve_extremes`
- `preview_mode`
- `extremes_threshold`
- any new parameters introduced in future releases

Contributions welcome – see "Adding new tests" below.

---

## GPU test coverage (`tests/gpu`)

| Feature | Test file |
|---------|-----------|
| `dithering_strength` effect | `test_dithering_strength.py` |
| `edge_blur_*` parameters | `test_edge_blur.py` |
| `hue_weight` parameter | `test_hue_weight.py` |
| `preserve_extremes` / `extremes_threshold` | `test_preserve_extremes.py` |

GPU tests are executed only when an OpenCL GPU is detected (see `conftest.py`).

---

## Integration tests

`integration/test_algorithm_happy_path.py` runs the algorithm end-to-end on synthetic images to ensure the default configuration produces a reasonable result without exceptions.

---

## Adding new tests

1. Choose the appropriate folder (`parameters/`, `gpu/`, or `integration/`).  
2. Name parameter tests `test_<parameter_name>.py`.  
3. Re-use fixtures from `conftest.py` (`gradient_image`, `noise_image`, etc.).  
4. Follow the three-tier methodology: typical, low extreme, high extreme.  
5. Assert at minimum: processing succeeds, output exists, and key metrics change in the expected direction.  
6. Update the coverage tables in this README.

Template:

```python
import pytest
from pathlib import Path
from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm

@pytest.mark.parametrize("<param>", [<typical>, <low>, <high>])
def test_<param>(tmp_path, gradient_image, noise_image, algorithm_cpu, <param>):
    out = Path(tmp_path) / "result.png"
    ok = algorithm_cpu.process_images(
        master_path=noise_image,
        target_path=gradient_image,
        output_path=str(out),
        <param>=<param>,
    )
    assert ok and out.exists()
```

---

## Known limitations

- Several parameters remain untested (see list above).  
- GPU results may vary slightly between vendors; assertions therefore focus on *difference* rather than absolute pixel values.

---

*Happy testing!* 🎨