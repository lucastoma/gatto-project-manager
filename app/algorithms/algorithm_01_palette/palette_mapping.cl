// Funkcja pomocnicza pozostaje bez zmian
float3 rgb_to_hsv(float r, float g, float b) {
    float cmax = fmax(r, fmax(g, b)); float cmin = fmin(r, fmin(g, b));
    float diff = cmax - cmin; float h = 0.0f, s = 0.0f;
    if (cmax == cmin) h = 0.0f;
    else if (cmax == r) h = fmod((60.0f * ((g - b) / diff) + 360.0f), 360.0f);
    else if (cmax == g) h = fmod((60.0f * ((b - r) / diff) + 120.0f), 360.0f);
    else if (cmax == b) h = fmod((60.0f * ((r - g) / diff) + 240.0f), 360.0f);
    if (cmax > 1e-6) s = (diff / cmax);
    return (float3)(h / 360.0f, s, cmax);
}

__kernel void map_palette(
    __global const float* pixels_flat_rgb,
    __global const float* palette_flat_hsv,  // Przyjmuje paletę w HSV!
    __global int* output_indices,
    const int num_palette_colors,
    const float hue_weight
) {
    int gid = get_global_id(0);
    
    // Konwersja piksela do HSV (tylko raz na wątek)
    float3 pixel_rgb = (float3)(pixels_flat_rgb[gid * 3], pixels_flat_rgb[gid * 3 + 1], pixels_flat_rgb[gid * 3 + 2]) / 255.0f;
    float3 pixel_hsv = rgb_to_hsv(pixel_rgb.x, pixel_rgb.y, pixel_rgb.z);

    float min_dist_sq = 1e30f;
    int best_idx = 0;

    // Porównanie z przekonwertowaną wcześniej paletą
    for (int i = 0; i < num_palette_colors; ++i) {
        float3 palette_hsv = (float3)(palette_flat_hsv[i * 3], palette_flat_hsv[i * 3 + 1], palette_flat_hsv[i * 3 + 2]);
        
        float delta_h = fabs(pixel_hsv.x - palette_hsv.x);
        delta_h = fmin(delta_h, 1.0f - delta_h);
        float delta_s = pixel_hsv.y - palette_hsv.y;
        float delta_v = pixel_hsv.z - palette_hsv.z;

        float dist_sq = (hue_weight * delta_h) * (hue_weight * delta_h) + delta_s * delta_s + delta_v * delta_v;
        
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_idx = i;
        }
    }
    output_indices[gid] = best_idx;
}
