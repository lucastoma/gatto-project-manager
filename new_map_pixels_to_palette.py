def _map_pixels_to_palette(
    self, image_array: np.ndarray, palette: List[List[int]], config: Dict[str, Any]
) -> np.ndarray:
    with self.profiler.profile_operation(
        "map_pixels_to_palette", algorithm_id=self.algorithm_id
    ):
        metric = config.get("distance_metric")
        palette_np = np.array(palette, dtype=np.float32)
        pixels_flat = image_array.reshape(-1, 3).astype(np.float32)

        if "hsv" in metric:
            pixels_hsv = color.rgb2hsv(pixels_flat / 255.0)
            palette_hsv = color.rgb2hsv(palette_np / 255.0)

            # Ustawienie domyślnych wag
            weights = np.full(
                (pixels_hsv.shape[0], 3), [config.get("hue_weight", 3.0), 1.0, 1.0]
            )

            distances_sq = self._calculate_hsv_distance_sq(
                pixels_hsv, palette_hsv, weights
            )
            
            # POPRAWIONA LOGIKA "Color Focus"
            if config.get("use_color_focus", False) and config.get("focus_ranges"):
                self.logger.info(
                    f"Using Color Focus with {len(config['focus_ranges'])} range(s)."
                )
                print(f"DEBUG: palette_hsv shape: {palette_hsv.shape}")
                print(f"DEBUG: palette_hsv colors: {palette_hsv}")
                
                # Dla każdego focus range
                for i, focus in enumerate(config["focus_ranges"]):
                    target_h = focus["target_hsv"][0] / 360.0
                    target_s = focus["target_hsv"][1] / 100.0
                    target_v = focus["target_hsv"][2] / 100.0

                    range_h = focus["range_h"] / 360.0
                    range_s = focus["range_s"] / 100.0
                    range_v = focus["range_v"] / 100.0

                    print(f"DEBUG: Focus {i+1} - target_hsv normalized: [{target_h:.3f}, {target_s:.3f}, {target_v:.3f}]")
                    print(f"DEBUG: Focus {i+1} - ranges normalized: H±{range_h/2:.3f}, S±{range_s/2:.3f}, V±{range_v/2:.3f}")

                    # Sprawdź które KOLORY Z PALETY pasują do focus range
                    palette_h_dist = np.abs(palette_hsv[:, 0] - target_h)
                    palette_hue_mask = np.minimum(palette_h_dist, 1.0 - palette_h_dist) <= (range_h / 2.0)
                    palette_sat_mask = np.abs(palette_hsv[:, 1] - target_s) <= (range_s / 2.0)
                    palette_val_mask = np.abs(palette_hsv[:, 2] - target_v) <= (range_v / 2.0)
                    
                    palette_final_mask = palette_hue_mask & palette_sat_mask & palette_val_mask
                    
                    print(f"DEBUG: Focus {i+1} - palette colors matching: {np.sum(palette_final_mask)}/{len(palette_hsv)}")
                    if np.sum(palette_final_mask) > 0:
                        matching_indices = np.where(palette_final_mask)[0]
                        print(f"DEBUG: Focus {i+1} - matching palette indices: {matching_indices}")
                        for idx in matching_indices:
                            print(f"DEBUG: Focus {i+1} - palette[{idx}] HSV: {palette_hsv[idx]} matches focus range")
                        
                        # APLIKUJ COLOR FOCUS: zmniejsz odległości do preferowanych kolorów palety
                        boost = focus.get("boost_factor", 1.0)
                        distances_sq[:, palette_final_mask] /= boost
                        print(f"DEBUG: Focus {i+1} - reduced distances to {np.sum(palette_final_mask)} palette colors by factor {boost}")
                    else:
                        print(f"DEBUG: Focus {i+1} - NO PALETTE COLORS MATCHED! Check ranges.")
            
            closest_indices = np.argmin(distances_sq, axis=1)

        elif metric == "lab" and SCIPY_AVAILABLE:
            # ... (logika dla LAB bez zmian)
            palette_lab = color.rgb2lab(palette_np / 255.0)
            kdtree = KDTree(palette_lab)
            pixels_lab = color.rgb2lab(pixels_flat / 255.0)
            _, closest_indices = kdtree.query(pixels_lab)
        else:
            # ... (logika dla RGB bez zmian)
            if metric == "lab":
                self.logger.warning(
                    "LAB metric used without Scipy. Falling back to slow calculation."
                )
            weights = (
                np.array([0.2126, 0.7152, 0.0722])
                if metric == "weighted_rgb"
                else np.array([1.0, 1.0, 1.0])
            )
            distances = np.sqrt(
                np.sum(
                    (
                        (
                            pixels_flat[:, np.newaxis, :]
                            - palette_np[np.newaxis, :, :]
                        )
                        * weights
                    )
                    ** 2,
                    axis=2,
                )
            )
            closest_indices = np.argmin(distances, axis=1)

        return palette_np[closest_indices].reshape(image_array.shape)
