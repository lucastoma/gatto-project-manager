/*
 * OpenCL Kernels for LAB Color Transfer
 */

__kernel void unified_lab_transfer(
    __global const float* source_lab,
    __global float* result_lab,
    const float s_mean_l, const float s_std_l, const float t_mean_l, const float t_std_l,
    const float s_mean_a, const float s_std_a, const float t_mean_a, const float t_std_a,
    const float s_mean_b, const float s_std_b, const float t_mean_b, const float t_std_b,
    const float weight_l, const float weight_a, const float weight_b, // Weights for weighted transfer
    const int selective_mode, // Flag for selective transfer (1 = preserve L)
    const int total_pixels)
{
    int gid = get_global_id(0);
    if (gid >= total_pixels) return;

    int index = gid * 3;

    // L channel
    float l_s = source_lab[index + 0];
    if (selective_mode == 1) {
        result_lab[index + 0] = l_s; // Preserve L channel
    } else {
        float std_ratio_l = (s_std_l > 1e-6f) ? (t_std_l / s_std_l) : 1.0f;
        float l_t = (l_s - s_mean_l) * std_ratio_l + t_mean_l;
        result_lab[index + 0] = l_s * (1.0f - weight_l) + l_t * weight_l;
    }

    // a channel
    float a_s = source_lab[index + 1];
    float std_ratio_a = (s_std_a > 1e-6f) ? (t_std_a / s_std_a) : 1.0f;
    float a_t = (a_s - s_mean_a) * std_ratio_a + t_mean_a;
    result_lab[index + 1] = a_s * (1.0f - weight_a) + a_t * weight_a;

    // b channel
    float b_s = source_lab[index + 2];
    float std_ratio_b = (s_std_b > 1e-6f) ? (t_std_b / s_std_b) : 1.0f;
    float b_t = (b_s - s_mean_b) * std_ratio_b + t_mean_b;
    result_lab[index + 2] = b_s * (1.0f - weight_b) + b_t * weight_b;
}

/*
 * Calculates a histogram for the L channel of a LAB image.
 * The histogram has a fixed size (e.g., 101 bins for L values 0-100).
 * The histogram buffer must be initialized to zeros before calling this kernel.
 */
__kernel void calculate_histogram(
    __global const float* source_lab,
    __global int* histogram,
    const int total_pixels)
{
    int gid = get_global_id(0);

    if (gid >= total_pixels) {
        return;
    }

    // We only care about the L channel, which is at index gid * 3
    float l_value = source_lab[gid * 3];

    // Map L value [0, 100] to an integer bin index [0, 100]
    int bin_index = (int)clamp(l_value, 0.0f, 100.0f);

    // Atomically increment the histogram bin
    // This ensures correctness when multiple threads write to the same bin
    atomic_inc(&histogram[bin_index]);
}

/*
 * Creates a segmentation mask based on luminance thresholds.
 * For each pixel, it outputs an integer (0, 1, or 2) corresponding
 * to the luminance segment (dark, mid, bright).
 */
__kernel void create_segmentation_mask(
    __global const float* lab_image,
    __global int* mask,
    const float threshold1, // e.g., 33rd percentile
    const float threshold2, // e.g., 66th percentile
    const int total_pixels)
{
    int gid = get_global_id(0);

    if (gid >= total_pixels) {
        return;
    }

    float l_value = lab_image[gid * 3];

    if (l_value <= threshold1) {
        mask[gid] = 0; // Dark segment
    } else if (l_value <= threshold2) {
        mask[gid] = 1; // Mid segment
    } else {
        mask[gid] = 2; // Bright segment
    }
}

/*
 * Applies color transfer based on pre-calculated segment statistics.
 * For each pixel, it identifies its segment, looks up the corresponding
 * source and target stats, and applies the color transfer formula.
 */
__kernel void apply_segmented_transfer(
    __global const float* source_lab,
    __global const int* source_mask, // Mask for the source image
    __global float* result_lab,      // Output buffer
    __global const float* s_stats,   // Source stats [seg0_L_mean, seg0_L_std, seg0_a_mean, ...]
    __global const float* t_stats,   // Target stats [seg0_L_mean, seg0_L_std, seg0_a_mean, ...]
    const int total_pixels)
{
    int gid = get_global_id(0);
    if (gid >= total_pixels) {
        return;
    }

    int segment_index = source_mask[gid];
    int pixel_index = gid * 3;

    // Each segment has 3 channels (L, a, b), and each channel has 2 stats (mean, std).
    // So, each segment's stats block has 3 * 2 = 6 floats.
    int stats_base_index = segment_index * 6;

    // Process each channel (L, a, b)
    for (int i = 0; i < 3; ++i) {
        int stats_offset = stats_base_index + i * 2;
        
        float s_mean = s_stats[stats_offset + 0];
        float s_std  = s_stats[stats_offset + 1];
        float t_mean = t_stats[stats_offset + 0];
        float t_std  = t_stats[stats_offset + 1];

        float pixel_val = source_lab[pixel_index + i];
        float std_ratio = (s_std > 1e-6f) ? (t_std / s_std) : 1.0f;

        result_lab[pixel_index + i] = (pixel_val - s_mean) * std_ratio + t_mean;
    }
}
