import pytest
import numpy as np

# Map test mode names to actual method names in the core class
METHOD_MAP = {
    "basic": "basic_lab_transfer",
    "linear_blend": "weighted_lab_transfer",
    "selective": "selective_lab_transfer",
    "adaptive": "adaptive_lab_transfer",
    "hybrid": "basic_lab_transfer",  # Hybrid mode falls back to basic as per routes.py
}

# Define test cases for different modes and parameters
# (mode_name, params_dict)
mode_test_cases = [
    ("basic", {}),
    ("hybrid", {}),
    ("linear_blend", {'weights': {'L': 0.5, 'a': 0.7, 'b': 0.3}}),
    ("selective", {'selective_channels': ['a', 'b'], 'blend_factor': 0.5}),
    ("adaptive", {'num_segments': 4, 'delta_e_threshold': 10.0, 'min_segment_size_perc': 0.01}),
]

@pytest.mark.parametrize("mode, params", mode_test_cases)
def test_all_modes_on_cpu(cpu_transfer_instance, sample_images, mode, params):
    """Test all processing modes on a CPU instance."""
    source_rgb, target_rgb = sample_images
    source_lab = cpu_transfer_instance.rgb_to_lab_optimized(source_rgb)
    target_lab = cpu_transfer_instance.rgb_to_lab_optimized(target_rgb)

    # Get the correct method name from the map
    method_name = METHOD_MAP[mode]
    transfer_method = getattr(cpu_transfer_instance, method_name)

    # Call method with appropriate arguments
    # Selective and adaptive might require special handling for masks/params
    if mode == 'selective':
        mask = (np.random.rand(source_lab.shape[0], source_lab.shape[1]) > 0.5).astype(np.uint8) * 255
        result_lab = transfer_method(source_lab, target_lab, mask=mask, **params)
    elif params:
        result_lab = transfer_method(source_lab, target_lab, **params)
    else:
        result_lab = transfer_method(source_lab, target_lab)

    assert result_lab is not None
    assert result_lab.shape == source_lab.shape
    assert result_lab.dtype == source_lab.dtype

@pytest.mark.parametrize("mode, params", mode_test_cases)
def test_all_modes_on_gpu(gpu_transfer_instance, sample_images, mode, params):
    """Test all processing modes on a strict GPU instance."""
    source_rgb, target_rgb = sample_images
    source_lab = gpu_transfer_instance.rgb_to_lab_optimized(source_rgb)
    target_lab = gpu_transfer_instance.rgb_to_lab_optimized(target_rgb)

    # Get the correct method name from the map
    method_name = METHOD_MAP[mode]
    transfer_method = getattr(gpu_transfer_instance, method_name)

    # The gpu_transfer_instance will internally handle GPU delegation.
    # We just call the method as usual.
    if mode == 'selective':
        mask = (np.random.rand(source_lab.shape[0], source_lab.shape[1]) > 0.5).astype(np.uint8) * 255
        result_lab = transfer_method(source_lab, target_lab, mask=mask, **params)
    elif params:
        result_lab = transfer_method(source_lab, target_lab, **params)
    else:
        result_lab = transfer_method(source_lab, target_lab)

    assert result_lab is not None
    assert result_lab.shape == source_lab.shape
