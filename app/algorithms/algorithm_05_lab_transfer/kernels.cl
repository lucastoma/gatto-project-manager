/*
 * OpenCL Kernels for LAB Color Transfer - Refactored
 */

// --- Kernel for Statistical Calculation (Parallel Reduction) ---

/*
 * Pass 1: Map and Partial Reduce
 * Each work-group calculates the sum and sum-of-squares for a portion of the image.
 * The partial results are stored in an intermediate buffer, which is then summed on the host.
 */
__kernel void stats_partial_reduce(
    __global const float* lab_image,
    __global float* partial_sums, // Output buffer for partial results [group0_sum_l, group0_sum_sq_l, ...]
    __local float* local_sums,   // Local memory for reduction within a work-group
    const int total_pixels,      // For segmented stats, this is num_pixels_in_segment
    const int data_offset_pixels // For segmented stats, offset in compacted_lab_data
)
{
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);
    int global_id = get_global_id(0);

    // Each work-item initializes its local memory slot for 6 values (sum and sum_sq for L, a, b)
    for (int i = 0; i < 6; ++i) {
        local_sums[local_id * 6 + i] = 0.0f;
    }

    // Each work-item processes multiple pixels in a strided loop
    for (int i = global_id; i < total_pixels; i += get_global_size(0)) {
        // When used for segmented stats, total_pixels is num_pixels_in_segment
        // and data_offset_pixels points to the start of the segment in compacted buffer.
        // When used for global stats, data_offset_pixels is 0.
        int base_idx_in_relevant_buffer = i + data_offset_pixels;
        int pixel_index = base_idx_in_relevant_buffer * 3;
        float l = lab_image[pixel_index + 0];
        float a = lab_image[pixel_index + 1];
        float b = lab_image[pixel_index + 2];
        
        local_sums[local_id * 6 + 0] += l;
        local_sums[local_id * 6 + 1] += l * l;
        local_sums[local_id * 6 + 2] += a;
        local_sums[local_id * 6 + 3] += a * a;
        local_sums[local_id * 6 + 4] += b;
        local_sums[local_id * 6 + 5] += b * b;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            for (int i = 0; i < 6; ++i) {
                local_sums[local_id * 6 + i] += local_sums[(local_id + offset) * 6 + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // The first work-item in the group writes the group's result to global memory
    if (local_id == 0) {
        for (int i = 0; i < 6; ++i) {
            partial_sums[group_id * 6 + i] = local_sums[i];
        }
    }
}


// --- Kernel for Basic Color Transfer ---

__kernel void basic_transfer(
    __global const float* source_lab,
    __global float* result_lab,
    __global const float* s_mean,
    __global const float* s_std,
    __global const float* t_mean,
    __global const float* t_std,
    const int total_pixels)
{
    int gid = get_global_id(0);
    if (gid >= total_pixels) return;

    int index = gid * 3;

    for (int i = 0; i < 3; ++i) {
        float s_pixel = source_lab[index + i];
        float std_ratio = (s_std[i] > 1e-6f) ? (t_std[i] / s_std[i]) : 1.0f;
        result_lab[index + i] = (s_pixel - s_mean[i]) * std_ratio + t_mean[i];
    }
}


// --- Kernels for Hybrid/Segmented Transfer ---

/*
 * Creates a 3-segment mask based on L-channel percentiles.
 */
__kernel void create_luminance_mask(
    __global const float* lab_image,
    __global int* mask,
    const float threshold1, // e.g., 33rd percentile
    const float threshold2, // e.g., 66th percentile
    const int total_pixels)
{
    int gid = get_global_id(0);
    if (gid >= total_pixels) return;

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
 */
__kernel void apply_segmented_transfer(
    __global const float* source_lab,
    __global const int* source_mask,
    __global float* result_lab,
    __global const float* s_stats,   // Source stats [seg0_L_mean, seg0_L_std, ...]
    __global const float* t_stats,   // Target stats [seg0_L_mean, seg0_L_std, ...]
    const int total_pixels)
{
    int gid = get_global_id(0);
    if (gid >= total_pixels) return;

    int segment_index = source_mask[gid];
    int pixel_index = gid * 3;
    int stats_base_index = segment_index * 6; // 6 stats per segment (mean/std for L,a,b)

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

// --- Kernel for Selective Color Transfer ---

/*
 * Applies selective color transfer based on a mask, blend factor, and selected channels.
 */
__kernel void selective_transfer(
    __global const float* source_lab,
    __global float* result_lab,
    __global const uchar* mask, // Mask (0-255), single channel
    __global const float* s_mean,
    __global const float* s_std,
    __global const float* t_mean,
    __global const float* t_std,
    const float blend_factor,
    const int process_l, // Flag to process L channel
    const int process_a, // Flag to process a channel
    const int process_b, // Flag to process b channel
    const int total_pixels)
{
    int gid = get_global_id(0);
    if (gid >= total_pixels) return;

    int index = gid * 3;
    
    // If mask value is low, just copy the source pixel to the result
    if (mask[gid] < 128) {
        result_lab[index + 0] = source_lab[index + 0];
        result_lab[index + 1] = source_lab[index + 1];
        result_lab[index + 2] = source_lab[index + 2];
        return;
    }

    // Process L channel
    float s_pixel_l = source_lab[index + 0];
    if (process_l == 1) {
        float std_ratio_l = (s_std[0] > 1e-6f) ? (t_std[0] / s_std[0]) : 1.0f;
        float transferred_pixel_l = (s_pixel_l - s_mean[0]) * std_ratio_l + t_mean[0];
        result_lab[index + 0] = (transferred_pixel_l * blend_factor) + (s_pixel_l * (1.0f - blend_factor));
    } else {
        result_lab[index + 0] = s_pixel_l;
    }

    // Process a channel
    float s_pixel_a = source_lab[index + 1];
    if (process_a == 1) {
        float std_ratio_a = (s_std[1] > 1e-6f) ? (t_std[1] / s_std[1]) : 1.0f;
        float transferred_pixel_a = (s_pixel_a - s_mean[1]) * std_ratio_a + t_mean[1];
        result_lab[index + 1] = (transferred_pixel_a * blend_factor) + (s_pixel_a * (1.0f - blend_factor));
    } else {
        result_lab[index + 1] = s_pixel_a;
    }

    // Process b channel
    float s_pixel_b = source_lab[index + 2];
    if (process_b == 1) {
        float std_ratio_b = (s_std[2] > 1e-6f) ? (t_std[2] / s_std[2]) : 1.0f;
        float transferred_pixel_b = (s_pixel_b - s_mean[2]) * std_ratio_b + t_mean[2];
        result_lab[index + 2] = (transferred_pixel_b * blend_factor) + (s_pixel_b * (1.0f - blend_factor));
    } else {
        result_lab[index + 2] = s_pixel_b;
    }
}

// --- Kernels for GPU-side Segment Statistics Calculation ---

/*
 * Counts pixels belonging to each segment.
 */
__kernel void count_pixels_per_segment(
    __global const int* segment_mask,      // input: mask identifying segment for each pixel
    __global int* segment_counts,          // output: array to store counts for each segment (atomic)
    const int total_image_pixels)
{
    int gid = get_global_id(0);
    if (gid >= total_image_pixels) return;

    int segment_id = segment_mask[gid];
    // Ensure segment_id is valid before atomic operation if necessary, though create_luminance_mask produces 0,1,2.
    atomic_inc(&segment_counts[segment_id]);
}

/*
 * Scatters pixel LAB data into a compacted buffer, ordered by segment.
 */
__kernel void scatter_pixels_by_segment(
    __global const float* source_lab,      // input: original lab image data (full image)
    __global const int* segment_mask,      // input: mask identifying segment for each pixel
    __global const int* segment_offsets,   // input: starting offset for each segment in compacted_data
    __global int* scatter_counters,        // temp: atomic counters for current write pos within each segment block (init to 0)
    __global float* compacted_lab_data,    // output: lab data reordered by segment
    const int total_image_pixels)
{
    int gid = get_global_id(0);
    if (gid >= total_image_pixels) return;

    int segment_id = segment_mask[gid];
    
    // Get the unique position for this pixel within its segment's block in the compacted_lab_data
    int local_offset_in_segment = atomic_inc(&scatter_counters[segment_id]);
    
    int output_pixel_idx = segment_offsets[segment_id] + local_offset_in_segment;
    
    // Copy L, a, b values from source_lab (indexed by gid) to compacted_lab_data (indexed by output_pixel_idx)
    compacted_lab_data[output_pixel_idx * 3 + 0] = source_lab[gid * 3 + 0];
    compacted_lab_data[output_pixel_idx * 3 + 1] = source_lab[gid * 3 + 1];
    compacted_lab_data[output_pixel_idx * 3 + 2] = source_lab[gid * 3 + 2];
}

// --- Kernel for Selective LAB Color Transfer ---
__kernel void selective_transfer_kernel(__global const float* source_lab,
                                      __global const float* target_lab,
                                      __global const uchar* mask,
                                      __global float* result_lab,
                                      const int process_L,
                                      const int process_a,
                                      const int process_b,
                                      const float blend_factor,
                                      const int width,
                                      const int height) {
    int id = get_global_id(0);
    // total_pixels can be derived from width * height, but since we have global_id directly related to pixels,
    // we can use it up to width*height. The check `id >= num_pixels` handles out-of-bounds.
    int num_pixels = width * height;

    if (id >= num_pixels) {
        return;
    }

    // Assuming source_lab, target_lab, result_lab are 1D arrays storing 3 floats (L,a,b) per pixel.
    // Mask is a 1D array storing 1 uchar per pixel.
    int base_idx = id * 3; 
    uchar mask_value = mask[id];

    float final_L, final_a, final_b;

    float source_L_val = source_lab[base_idx + 0];
    float source_a_val = source_lab[base_idx + 1];
    float source_b_val = source_lab[base_idx + 2];

    if (mask_value > 0) { // Apply transfer only in masked area (mask value non-zero)
        float target_L_val = target_lab[base_idx + 0];
        float target_a_val = target_lab[base_idx + 1];
        float target_b_val = target_lab[base_idx + 2];

        if (process_L) {
            final_L = source_L_val * (1.0f - blend_factor) + target_L_val * blend_factor;
        } else {
            final_L = source_L_val;
        }

        if (process_a) {
            final_a = source_a_val * (1.0f - blend_factor) + target_a_val * blend_factor;
        } else {
            final_a = source_a_val;
        }

        if (process_b) {
            final_b = source_b_val * (1.0f - blend_factor) + target_b_val * blend_factor;
        } else {
            final_b = source_b_val;
        }
    } else { // Outside mask, keep source values
        final_L = source_L_val;
        final_a = source_a_val;
        final_b = source_b_val;
    }

    result_lab[base_idx + 0] = final_L;
    result_lab[base_idx + 1] = final_a;
    result_lab[base_idx + 2] = final_b;
}

