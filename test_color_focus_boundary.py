#!/usr/bin/env python3
"""
Test Color Focus - obrazy na granicy kolorów palety
"""

import numpy as np
from PIL import Image
import tempfile
import os
import sys

# Dodaj projekt do ścieżki
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm


def test_color_focus_boundary():
    """Test gdzie Color Focus powinien zmienić mapowanie"""
    
    print("============================================================")
    print("TEST COLOR FOCUS - OBRAZY NA GRANICY KOLORÓW")
    print("============================================================")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        master_path = os.path.join(temp_dir, "master.png")
        target_path = os.path.join(temp_dir, "target.png") 
        result_without_path = os.path.join(temp_dir, "result_without.png")
        result_with_path = os.path.join(temp_dir, "result_with.png")
        
        # Master: dwa kolory - czerwony i niebieski
        master = np.zeros((100, 100, 3), dtype=np.uint8)
        master[0:50, :] = [255, 0, 0]    # Górna połowa: czerwony (H=0°)
        master[50:100, :] = [0, 0, 255]  # Dolna połowa: niebieski (H=240°)
        
        # Target: kolor pośredni między czerwonym a niebieskim
        # Fioletowy (H=270°) - blżej niebieskiego niż czerwonego
        target = np.full((100, 100, 3), [128, 0, 255], dtype=np.uint8)  # Fiolet
        
        # Zapisanie obrazów
        Image.fromarray(master).save(master_path)
        Image.fromarray(target).save(target_path)
        
        print("Master: czerwony (H=0°) + niebieski (H=240°)")
        print("Target: fioletowy (H=270°) - naturalnie bliżej niebieskiego")
        
        algorithm = PaletteMappingAlgorithm()
        
        # Test 1: BEZ Color Focus (powinien wybrać niebieski)
        print("\n--- TEST 1: BEZ COLOR FOCUS ---")
        success1 = algorithm.process_images(
            master_path=master_path,
            target_path=target_path, 
            output_path=result_without_path,
            num_colors=2,
            distance_metric="weighted_hsv"
        )
        
        # Analiza wyniku
        result1 = np.array(Image.open(result_without_path))
        dominant_color1 = result1[0, 0]  # Pierwszy piksel
        
        # Test 2: Z Color Focus na CZERWONY
        print("\n--- TEST 2: Z COLOR FOCUS NA CZERWONY ---")
        focus_config = [{
            "target_hsv": [270, 100, 100],  # Fioletowy (taki jak target)
            "range_h": 60,   # Szeroki zakres
            "range_s": 100,  # Pełny zakres saturacji
            "range_v": 100,  # Pełny zakres jasności
            "boost_factor": 100.0  # Bardzo wysoki boost dla czerwonego
        }]
        
        success2 = algorithm.process_images(
            master_path=master_path,
            target_path=target_path,
            output_path=result_with_path,
            num_colors=2,
            distance_metric="weighted_hsv", 
            use_color_focus=True,
            focus_ranges=focus_config
        )
        
        # Analiza wyniku
        result2 = np.array(Image.open(result_with_path))
        dominant_color2 = result2[0, 0]  # Pierwszy piksel
        
        # Porównanie
        print(f"\nBez Color Focus: RGB {dominant_color1}")
        print(f"Z Color Focus: RGB {dominant_color2}")
        print(f"Czerwony to: [255, 0, 0]")
        print(f"Niebieski to: [0, 0, 255]")
        
        # Sprawdź które kolory zostały wybrane
        is_red1 = np.allclose(dominant_color1, [255, 0, 0], atol=10)
        is_blue1 = np.allclose(dominant_color1, [0, 0, 255], atol=10)
        is_red2 = np.allclose(dominant_color2, [255, 0, 0], atol=10)
        is_blue2 = np.allclose(dominant_color2, [0, 0, 255], atol=10)
        
        print(f"\nBez focus -> {'Czerwony' if is_red1 else 'Niebieski' if is_blue1 else 'Inny'}")
        print(f"Z focus -> {'Czerwony' if is_red2 else 'Niebieski' if is_blue2 else 'Inny'}")
        
        # Test sukcesu
        color_focus_works = not np.array_equal(dominant_color1, dominant_color2)
        
        print("\n============================================================")
        if color_focus_works:
            print("✅ COLOR FOCUS DZIAŁA! Zmienił wybór koloru.")
            if is_blue1 and is_red2:
                print("✅ PERFECT! Bez focus->niebieski, z focus->czerwony (zgodnie z oczekiwaniem)")
            elif is_red1 and is_blue2:
                print("⚠️  ODWROTNY EFEKT: focus zmienił na niebieski")
        else:
            print("❌ COLOR FOCUS NIE DZIAŁA! Identyczne wyniki.")
            print("MOŻLIWE PRZYCZYNY:")
            print("- Boost factor może nie być stosowany poprawnie")
            print("- Logika Color Focus ma błąd")
            print("- Warunki zakresu nie są spełnione")
        print("============================================================")
        
        return color_focus_works


if __name__ == "__main__":
    test_color_focus_boundary()
