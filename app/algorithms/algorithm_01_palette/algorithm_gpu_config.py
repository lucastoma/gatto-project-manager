# /app/algorithms/algorithm_01_palette/algorithm-gpu-config.py
# Moduł odpowiedzialny za zarządzanie konfiguracją algorytmu.

import json
from typing import Any, Dict

def get_default_config() -> Dict[str, Any]:
    """Zwraca słownik z domyślnymi wartościami konfiguracji algorytmu."""
    return {
        "num_colors": 8,
        "palette_method": "kmeans",
        "quality": 5,
        "distance_metric": "weighted_hsv",
        "hue_weight": 3.0,
        "saturation_weight": 1.0,
        "value_weight": 1.0,
        "use_color_focus": False,
        "focus_ranges": [],
        "dithering_method": "none",
        "dithering_strength": 8.0,
        "inject_extremes": True,
        "preserve_extremes": False,
        "extremes_threshold": 10,
        "edge_blur_enabled": False,
        "edge_blur_radius": 1.5,
        "edge_blur_strength": 0.3,
        "edge_detection_threshold": 25,
        "postprocess_median_filter": False,
        # Opcje specyficzne dla GPU
        "force_cpu": False,
        "gpu_batch_size": 2_000_000,
        "enable_kernel_fusion": True,
        "gpu_memory_cleanup": True,
        "use_64bit_indices": False,  # Dla bardzo dużych obrazów
        "_max_palette_size": 256,    # Dodane dla pełnej spójności konfiguracji
    }

def validate_run_config(config: Dict[str, Any], max_palette_size: int = 256):
    """
    Waliduje i normalizuje parametry konfiguracyjne w locie.
    Modyfikuje przekazany słownik `config`.
    """
    if "hue_weight" in config:
        config["hue_weight"] = max(0.1, min(10.0, float(config["hue_weight"])))
    if "gpu_batch_size" in config:
        config["gpu_batch_size"] = max(100_000, min(10_000_000, int(config["gpu_batch_size"])))
    if "num_colors" in config:
        config["num_colors"] = max(2, min(max_palette_size, int(config["num_colors"])))

def load_config(config_path: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wczytuje konfigurację z pliku JSON, waliduje ją i łączy z konfiguracją domyślną.
    """
    config = default_config.copy()
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        
        # Walidacja wczytanych wartości
        validate_run_config(user_config, default_config.get("_max_palette_size", 256))
        
        config.update(user_config)
    except Exception as e:
        # W przypadku błędu, logowanie powinno odbywać się w klasie, która ma logger.
        # Tutaj zwracamy tylko domyślną konfigurację.
        print(f"Warning: Error loading configuration from {config_path}: {e}. Using defaults.")
        return default_config
        
    return config
