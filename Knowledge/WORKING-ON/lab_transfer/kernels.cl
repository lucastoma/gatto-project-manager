/*
 * OpenCL Kernels for LAB Color Transfer
 */

__kernel void basic_lab_transfer(
    __global const float* source_lab,
    __global float* result_lab,
    const float s_mean_l, const float s_std_l,
    const float t_mean_l, const float t_std_l,
    const float s_mean_a, const float s_std_a,
    const float t_mean_a, const float t_std_a,
    const float s_mean_b, const float s_std_b,
    const float t_mean_b, const float t_std_b,
    const int total_pixels)
{
    int gid = get_global_id(0);

    if (gid >= total_pixels) {
        return;
    }

    int index = gid * 3;

    // L channel
    float l = source_lab[index + 0];
    float std_ratio_l = (s_std_l > 1e-6f) ? (t_std_l / s_std_l) : 1.0f;
    result_lab[index + 0] = (l - s_mean_l) * std_ratio_l + t_mean_l;

    // a channel
    float a = source_lab[index + 1];
    float std_ratio_a = (s_std_a > 1e-6f) ? (t_std_a / s_std_a) : 1.0f;
    result_lab[index + 1] = (a - s_mean_a) * std_ratio_a + t_mean_a;

    // b channel
    float b = source_lab[index + 2];
    float std_ratio_b = (s_std_b > 1e-6f) ? (t_std_b / s_std_b) : 1.0f;
    result_lab[index + 2] = (b - s_mean_b) * std_ratio_b + t_mean_b;
}
