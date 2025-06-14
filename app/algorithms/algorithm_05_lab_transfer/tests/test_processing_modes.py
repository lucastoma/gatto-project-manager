import pytest
import numpy as np

# Map test mode names to actual method names in the core class
METHOD_MAP = {
    "basic": "basic_lab_transfer",
    "linear_blend": "weighted_lab_transfer",
    "selective": "selective_lab_transfer",
    "adaptive": "adaptive_lab_transfer", # This will now use the GPU-side stats if on GPU instance
    "hybrid": "hybrid_transfer",
}

# Define test cases for CPU with specific parameters
cpu_mode_test_cases = [
    ("basic", {}),
    ("hybrid", {}), 
    ("linear_blend", {'weights': {'L': 0.5, 'a': 0.7, 'b': 0.3}}),
    ("selective", {'selective_channels': ['a', 'b'], 'blend_factor': 0.5}),
    ("adaptive", {'num_segments': 4, 'delta_e_threshold': 10.0, 'min_segment_size_perc': 0.01}),
]

# Define test cases for GPU.
# - 'adaptive' on GPU uses internal luminance segmentation; user-facing params are not taken by gpu_core.adaptive_lab_transfer.
#   The recent changes mean the internal statistics for this segmentation are now GPU-calculated.
gpu_mode_test_cases = [
    ("basic", {}),
    ("linear_blend", {'weights': {'L': 0.6, 'a': 0.2, 'b': 0.2}}),
    ("hybrid", {}),
    ("selective", {'selective_channels': ['L', 'b'], 'blend_factor': 0.8}),
    ("adaptive", {}), # GPU adaptive uses internal luminance segmentation.
]

@pytest.mark.parametrize("mode, params", cpu_mode_test_cases)
def test_all_modes_on_cpu(cpu_transfer_instance, sample_images, mode, params):
    """Test all processing modes on a CPU instance."""
    source_rgb, target_rgb = sample_images
    source_lab = cpu_transfer_instance.rgb_to_lab_optimized(source_rgb)
    target_lab = cpu_transfer_instance.rgb_to_lab_optimized(target_rgb)

    method_name = METHOD_MAP[mode]
    transfer_method = getattr(cpu_transfer_instance, method_name)

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
    assert result_lab.dtype == source_lab.dtype

@pytest.mark.parametrize("mode, params", gpu_mode_test_cases)
def test_all_modes_on_gpu(gpu_transfer_instance, sample_images, mode, params):
    """Test all processing modes on a strict GPU instance."""
    source_rgb, target_rgb = sample_images
    source_lab = gpu_transfer_instance.rgb_to_lab_optimized(source_rgb)
    target_lab = gpu_transfer_instance.rgb_to_lab_optimized(target_rgb)

    method_name = METHOD_MAP[mode]
    # The gpu_transfer_instance is already configured for GPU use by the fixture.
    # Methods on LABColorTransfer will delegate to LABColorTransferGPU internally.
    transfer_method = getattr(gpu_transfer_instance, method_name)

    if mode == 'selective':
        mask = (np.random.rand(source_lab.shape[0], source_lab.shape[1]) > 0.5).astype(np.uint8) * 255
        result_lab = transfer_method(source_lab, target_lab, mask=mask, **params)
    elif params:
        result_lab = transfer_method(source_lab, target_lab, **params)
    else:
        result_lab = transfer_method(source_lab, target_lab)

    assert result_lab is not None
    assert result_lab.shape == source_lab.shape
