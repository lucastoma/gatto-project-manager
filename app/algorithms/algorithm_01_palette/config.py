"""
Algorithm 01: Palette Mapping Configuration
===========================================
Konfiguracja dla algorytmu mapowania palety, w tym nowe opcje zaawansowane.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PaletteMappingConfig:
    """Konfiguracja dla Algorytmu Mapowania Palety."""
    
    # --- NOWE OPCJE ---
    # Domyślne wartości dla nowych, zaawansowanych parametrów.
    # API będzie je nadpisywać, jeśli zostaną podane w requeście.
    
    # Grupa 1: Kontrola nad Paletą
    k_colors: int = 16
    palette_source_area: str = "full_image"  # Opcje: 'full_image', 'selection', 'active_layer'
    exclude_colors: Optional[list] = None     # Lista kolorów RGB do wykluczenia, np. [[255,255,255]]

    # Grupa 2: Kontrola nad Mapowaniem
    distance_metric: str = "LAB"             # Opcje: 'RGB', 'LAB' (percepcyjna)
    use_dithering: bool = False              # Czy włączyć rozpraszanie (dithering)
    preserve_luminance: bool = True          # Czy zachować oryginalną jasność obrazu docelowego

    # Grupa 3: Kontrola nad Wydajnością
    preview_mode: bool = False
    preview_size: tuple = (500, 500)         # Maksymalny rozmiar dla podglądu

    # --- ISTNIEJĄCE PARAMETRY K-MEANS ---
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4

# Globalna funkcja do pobierania domyślnej konfiguracji
def get_default_config() -> PaletteMappingConfig:
    """Zwraca instancję z domyślną konfiguracją."""
    return PaletteMappingConfig()
