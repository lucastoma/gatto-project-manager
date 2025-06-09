#!/usr/bin/env python3
"""
Prosty test edge blending - sprawdzenie czy parametry działają
"""

import sys
import os
sys.path.append('.')

print("=== EDGE BLENDING TEST ===")

try:
    from app.algorithms.algorithm_01_palette.algorithm import create_palette_mapping_algorithm
    print("✅ Import algorytmu - OK")
    
    # Stwórz algorytm
    algorithm = create_palette_mapping_algorithm()
    print("✅ Tworzenie instancji - OK")
    
    # Sprawdź domyślną konfigurację
    config = algorithm.default_config()
    edge_params = {
        'edge_blur_enabled': config.get('edge_blur_enabled', 'MISSING'),
        'edge_blur_radius': config.get('edge_blur_radius', 'MISSING'),
        'edge_blur_strength': config.get('edge_blur_strength', 'MISSING'),
        'edge_detection_threshold': config.get('edge_detection_threshold', 'MISSING'),
        'edge_blur_method': config.get('edge_blur_method', 'MISSING')
    }
    
    print("🔍 Parametry edge blending w konfiguracji:")
    for param, value in edge_params.items():
        print(f"  {param}: {value}")
    
    # Sprawdź czy metody istnieją
    methods = ['apply_edge_blending', '_detect_palette_edges', '_apply_selective_blur']
    for method in methods:
        if hasattr(algorithm, method):
            print(f"✅ Metoda {method} - istnieje")
        else:
            print(f"❌ Metoda {method} - BRAK")
    
    print("\n=== TEST ZAKOŃCZONY ===")
    
except Exception as e:
    print(f"❌ BŁĄD: {e}")
    import traceback
    traceback.print_exc()
