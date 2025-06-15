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
        int pixel_index = base_idx_in_relevant_buffer * 3; // Each pixel has 3 float values (L, a, b)
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

    // Barrier to ensure all local sums are computed
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction within the work-group
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            for (int i = 0; i < 6; ++i) {
                local_sums[local_id * 6 + i] += local_sums[(local_id + offset) * 6 + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in each group writes the result to global memory
    if (local_id == 0) {
        for (int i = 0; i < 6; ++i) {
            partial_sums[group_id * 6 + i] = local_sums[i];
        }
    }
}


// --- Kernels for Adaptive Transfer (Luminance-based Segmentation) ---

/*
 * Kernel 1: Create Luminance Mask
 * Assigns each pixel to a segment based on its L value.
 * Output: segment_indices_map (for each pixel, its segment index)
 */
__kernel void create_luminance_mask(
    __global const float* lab_image,
    __global int* segment_indices_map,
    const int num_segments, // Added num_segments argument
    const int width,
    const int height
) {
    int id = get_global_id(0);
    int num_pixels = width * height;

    if (id >= num_pixels || num_segments <= 0) { // Added check for num_segments
        return;
    }

    float l_value = lab_image[id * 3]; // L channel

    // Dynamic segmentation based on num_segments, assuming L range [0, 100]
    const float l_min = 0.0f;
    const float l_max = 100.0f;
    
    if (num_segments == 1) {
        segment_indices_map[id] = 0;
    } else {
        float segment_width = (l_max - l_min) / (float)num_segments;
        if (segment_width < 0.00001f) { // Avoid division by zero or very small width
            segment_indices_map[id] = 0; // Default to first segment
        } else {
            int segment_index = (int)((l_value - l_min) / segment_width);
            segment_indices_map[id] = clamp(segment_index, 0, num_segments - 1);
        }
    }
}

/*
 * Kernel 2: Count Pixels per Segment
 * Counts how many pixels fall into each segment.
 * Output: segment_pixel_counts (array of size N, storing pixel count for each segment)
 * Host must initialize segment_pixel_counts to zeros.
 */
__kernel void count_pixels_per_segment(
    __global const int* segment_indices_map,
    __global int* segment_pixel_counts,
    const int num_segments, // Added num_segments argument
    const int total_pixels
) {
    int id = get_global_id(0);

    if (id >= total_pixels) {
        return;
    }

    int segment_idx = segment_indices_map[id];

    // Ensure segment_idx is within bounds [0, num_segments - 1] before atomic operation.
    if (segment_idx >= 0 && segment_idx < num_segments) {
        atomic_inc(&segment_pixel_counts[segment_idx]);
    }
}

/*
 * Kernel 3: Calculate Segment Offsets (Exclusive Scan)
 * Calculates the starting offset for each segment in the compacted data array.
 * Output: segment_offsets (array of size N, storing start offset for each segment)
 * This is typically done on the host after K2, or can be a separate kernel if needed.
 * For simplicity, this step is often handled on the host. If done in a kernel:
 */
__kernel void calculate_segment_offsets(
    __global const int* segment_pixel_counts, // Input: counts from K2
    __global int* segment_offsets,            // Output: offsets
    const int num_segments
) {
    // This kernel is best run as a single work-item or small work-group on host-like logic
    if (get_global_id(0) == 0) {
        segment_offsets[0] = 0;
        for (int i = 1; i < num_segments; ++i) {
            segment_offsets[i] = segment_offsets[i-1] + segment_pixel_counts[i-1];
        }
    }
}


/*
 * Kernel 4: Scatter Pixels by Segment (into a compacted buffer)
 * Rearranges pixel data so that all pixels belonging to the same segment are contiguous.
 * Output: compacted_lab_data (LAB data, pixels grouped by segment)
 *         temp_segment_counters (used to track current write position for each segment)
 * Host must initialize temp_segment_counters to zeros.
 */
__kernel void scatter_pixels_by_segment(
    __global const float* original_lab_image,
    __global const int* segment_indices_map,
    __global const int* segment_offsets,      // Input: offsets from K3
    __global float* compacted_lab_data,
    __global int* temp_segment_counters,      // Temp buffer, size N, init to 0 by host
    const int num_segments,                   // Added num_segments
    const int width,
    const int height
) {
    int id = get_global_id(0);
    int total_pixels = width * height;

    if (id >= total_pixels) {
        return;
    }

    int segment_idx = segment_indices_map[id];
    if (segment_idx < 0 || segment_idx >= num_segments) return; // Safety check

    // Get the current write position for this segment and increment it atomically
    int write_pos_in_segment = atomic_inc(&temp_segment_counters[segment_idx]);
    
    // Calculate the final write index in the compacted buffer
    int compacted_idx = segment_offsets[segment_idx] + write_pos_in_segment;

    // Copy L, a, b values
    compacted_lab_data[compacted_idx * 3 + 0] = original_lab_image[id * 3 + 0];
    compacted_lab_data[compacted_idx * 3 + 1] = original_lab_image[id * 3 + 1];
    compacted_lab_data[compacted_idx * 3 + 2] = original_lab_image[id * 3 + 2];
}


/*
 * Kernel 5: Apply Segmented Transfer
 * Applies basic statistical transfer independently for each segment.
 * Uses stats_partial_reduce for calculating stats per segment.
 * Output: result_lab_image (final processed image)
 * This kernel is more complex as it orchestrates stats calculation and application.
 * A simplified version might just apply pre-calculated scale/offset factors.
 * Here, we assume scale/offset factors (mean_s, std_s, mean_t, std_t for L,a,b for each segment)
 * are pre-calculated on the host after stats_partial_reduce is run for each segment
 * on both source and target (compacted) data.
 */
__kernel void apply_segmented_transfer(
    __global const float* source_lab_image, // Original source image
    __global float* result_lab_image,       // Output image
    __global const int* segment_indices_map,
    __global const float* segment_stats_source, // Array: [seg0_meanL_s, seg0_stdL_s, seg0_meanA_s, ..., segN_stdB_s] (6*N floats)
    __global const float* segment_stats_target, // Array: [seg0_meanL_t, seg0_stdL_t, ..., segN_stdB_t] (6*N floats)
    const int num_segments,
    const int width,
    const int height
) {
    int id = get_global_id(0);
    int total_pixels = width * height;

    if (id >= total_pixels) {
        return;
    }

    int segment_idx = segment_indices_map[id];
    if (segment_idx < 0 || segment_idx >= num_segments) return; // Safety

    int stats_base_idx = segment_idx * 6; // 6 stats per segment (mean_l, std_l, mean_a, std_a, mean_b, std_b)

    float src_l = source_lab_image[id * 3 + 0];
    float src_a = source_lab_image[id * 3 + 1];
    float src_b = source_lab_image[id * 3 + 2];

    float mean_l_s = segment_stats_source[stats_base_idx + 0];
    float std_l_s  = segment_stats_source[stats_base_idx + 1];
    float mean_a_s = segment_stats_source[stats_base_idx + 2];
    float std_a_s  = segment_stats_source[stats_base_idx + 3];
    float mean_b_s = segment_stats_source[stats_base_idx + 4];
    float std_b_s  = segment_stats_source[stats_base_idx + 5];

    float mean_l_t = segment_stats_target[stats_base_idx + 0];
    float std_l_t  = segment_stats_target[stats_base_idx + 1];
    float mean_a_t = segment_stats_target[stats_base_idx + 2];
    float std_a_t  = segment_stats_target[stats_base_idx + 3];
    float mean_b_t = segment_stats_target[stats_base_idx + 4];
    float std_b_t  = segment_stats_target[stats_base_idx + 5];

    // Apply transfer: (val - mean_s) * (std_t / std_s) + mean_t
    // Add epsilon to std_s to avoid division by zero
    float epsilon = 1e-6f;

    result_lab_image[id * 3 + 0] = (src_l - mean_l_s) * (std_l_t / (std_l_s + epsilon)) + mean_l_t;
    result_lab_image[id * 3 + 1] = (src_a - mean_a_s) * (std_a_t / (std_a_s + epsilon)) + mean_a_t;
    result_lab_image[id * 3 + 2] = (src_b - mean_b_s) * (std_b_t / (std_b_s + epsilon)) + mean_b_t;
}


// --- Kernel for Basic Color Transfer ---
__kernel void basic_transfer_kernel(
    __global const float* source_lab,
    __global float* result_lab,
    const float src_mean_l, const float src_std_l,
    const float src_mean_a, const float src_std_a,
    const float src_mean_b, const float src_std_b,
    const float tgt_mean_l, const float tgt_std_l,
    const float tgt_mean_a, const float tgt_std_a,
    const float tgt_mean_b, const float tgt_std_b,
    const int width, const int height) {
    
    int id = get_global_id(0);
    int num_pixels = width * height;

    if (id >= num_pixels) {
        return;
    }

    int base_idx = id * 3;
    float l = source_lab[base_idx + 0];
    float a = source_lab[base_idx + 1];
    float b = source_lab[base_idx + 2];

    float epsilon = 1e-6f; // To prevent division by zero

    // Transfer L channel
    l = (l - src_mean_l) * (tgt_std_l / (src_std_l + epsilon)) + tgt_mean_l;
    // Transfer a channel
    a = (a - src_mean_a) * (tgt_std_a / (src_std_a + epsilon)) + tgt_mean_a;
    // Transfer b channel
    b = (b - src_mean_b) * (tgt_std_b / (src_std_b + epsilon)) + tgt_mean_b;

    result_lab[base_idx + 0] = l;
    result_lab[base_idx + 1] = a;
    result_lab[base_idx + 2] = b;
}

// --- Kernel for Weighted Color Transfer ---
__kernel void weighted_transfer_kernel(
    __global const float* source_lab,
    __global float* result_lab,
    const float src_mean_l, const float src_std_l,
    const float src_mean_a, const float src_std_a,
    const float src_mean_b, const float src_std_b,
    const float tgt_mean_l, const float tgt_std_l,
    const float tgt_mean_a, const float tgt_std_a,
    const float tgt_mean_b, const float tgt_std_b,
    const float weight_l, const float weight_a, const float weight_b,
    const int width, const int height) {

    int id = get_global_id(0);
    int num_pixels = width * height;

    if (id >= num_pixels) {
        return;
    }

    int base_idx = id * 3;
    float l_s = source_lab[base_idx + 0];
    float a_s = source_lab[base_idx + 1];
    float b_s = source_lab[base_idx + 2];

    float epsilon = 1e-6f;

    // Transformed values
    float l_t = (l_s - src_mean_l) * (tgt_std_l / (src_std_l + epsilon)) + tgt_mean_l;
    float a_t = (a_s - src_mean_a) * (tgt_std_a / (src_std_a + epsilon)) + tgt_mean_a;
    float b_t = (b_s - src_mean_b) * (tgt_std_b / (src_std_b + epsilon)) + tgt_mean_b;

    // Weighted average
    result_lab[base_idx + 0] = (1.0f - weight_l) * l_s + weight_l * l_t;
    result_lab[base_idx + 1] = (1.0f - weight_a) * a_s + weight_a * a_t;
    result_lab[base_idx + 2] = (1.0f - weight_b) * b_s + weight_b * b_t;
}


// --- Kernel for Selective Color Transfer ---
__kernel void selective_transfer_kernel(__global const float* source_lab,
                                      __global const float* target_lab,
                                      __global const uchar* mask, // uchar mask
                                      __global float* result_lab,
                                      const int process_L, // Boolean flags (0 or 1)
                                      const int process_a,
                                      const int process_b,
                                      const float blend_factor,
                                      const int width,
                                      const int height) {
    int id = get_global_id(0);
    int num_pixels = width * height;

    if (id >= num_pixels) {
        return;
    }

    int base_idx = id * 3; 
    uchar mask_value = mask[id];

    float final_L, final_a, final_b;

    float source_L_val = source_lab[base_idx + 0];
    float source_a_val = source_lab[base_idx + 1];
    float source_b_val = source_lab[base_idx + 2];

    if (mask_value > 0) { 
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
    } else { 
        final_L = source_L_val;
        final_a = source_a_val;
        final_b = source_b_val;
    }

    result_lab[base_idx + 0] = final_L;
    result_lab[base_idx + 1] = final_a;
    result_lab[base_idx + 2] = final_b;
}

// Kernel for linear blending in the hybrid pipeline
// Blends current_source_lab with a statistically transformed version of original_target_lab
__kernel void linear_blend_pipeline_kernel(__global float3* current_source_lab,
                                           __global float3* original_target_lab, // This is the original target image buffer
                                           __global float3* result_lab,
                                           float src_mean_l, float src_std_l,       // Stats of current_source_lab
                                           float src_mean_a, float src_std_a,
                                           float src_mean_b, float src_std_b,
                                           float tgt_mean_l, float tgt_std_l,       // Stats of original_target_lab
                                           float tgt_mean_a, float tgt_std_a,
                                           float tgt_mean_b, float tgt_std_b,
                                           float weight_l, float weight_a, float weight_b,
                                           int width, int height)
{
    int gid = get_global_id(0);
    if (gid >= width * height) return;

    float3 current_src_pixel = current_source_lab[gid];
    float3 original_tgt_pixel = original_target_lab[gid];
    float3 transformed_tgt_pixel_for_blending; // Target pixel transformed to match current_src_pixel's stats context

    // Statistically transform original_tgt_pixel to match the statistics of current_src_pixel
    // This means: (original_tgt_pixel - its_mean) * (current_src_std / its_std) + current_src_mean
    if (tgt_std_l != 0.0f) transformed_tgt_pixel_for_blending.x = (original_tgt_pixel.x - tgt_mean_l) * (src_std_l / tgt_std_l) + src_mean_l;
    else transformed_tgt_pixel_for_blending.x = src_mean_l;

    if (tgt_std_a != 0.0f) transformed_tgt_pixel_for_blending.y = (original_tgt_pixel.y - tgt_mean_a) * (src_std_a / tgt_std_a) + src_mean_a;
    else transformed_tgt_pixel_for_blending.y = src_mean_a;

    if (tgt_std_b != 0.0f) transformed_tgt_pixel_for_blending.z = (original_tgt_pixel.z - tgt_mean_b) * (src_std_b / tgt_std_b) + src_mean_b;
    else transformed_tgt_pixel_for_blending.z = src_mean_b;

    // Blend the current_src_pixel with the transformed_tgt_pixel_for_blending
    result_lab[gid].x = current_src_pixel.x * (1.0f - weight_l) + transformed_tgt_pixel_for_blending.x * weight_l;
    result_lab[gid].y = current_src_pixel.y * (1.0f - weight_a) + transformed_tgt_pixel_for_blending.y * weight_a;
    result_lab[gid].z = current_src_pixel.z * (1.0f - weight_b) + transformed_tgt_pixel_for_blending.z * weight_b;
}
