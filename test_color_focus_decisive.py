#!/usr/bin/env python3
"""
Test Color Focus - test decydujący gdzie Focus powinien zmienić wybór
"""

import numpy as np
from PIL import Image
import tempfile
import os
import sys

# Dodaj projekt do ścieżki
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm


def hsv_to_rgb(h, s, v):
    """Konwersja HSV do RGB (h w stopniach 0-360, s,v w % 0-100)"""
    h = h / 360.0
    s = s / 100.0  
    v = v / 100.0
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    i = i % 6
    
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    elif i == 5:
        r, g, b = v, p, q
    
    return [int(r * 255), int(g * 255), int(b * 255)]


def test_color_focus_decisive():
    """Test gdzie Color Focus powinien zmienić wybór"""
    
    print("============================================================")
    print("TEST COLOR FOCUS - DECYDUJĄCY PRZYPADEK")
    print("============================================================")
    
    # Master: czerwony + zielony (przeciwległe kolory)
    master = np.zeros((200, 200, 3), dtype=np.uint8)
    master[:, :100] = hsv_to_rgb(0, 100, 100)   # Czerwony H=0°
    master[:, 100:] = hsv_to_rgb(120, 100, 100) # Zielony H=120°
    
    # Target: pomarańczowy H=30° - BLIŻEJ czerwonego (0°) niż zielonego (120°) 
    # Odległości: |30-0|=30° vs |30-120|=90°
    target_color = hsv_to_rgb(30, 100, 100)  # Pomarańczowy
    target = np.full((100, 100, 3), target_color, dtype=np.uint8)
    
    print(f"Master: czerwony (H=0°) + zielony (H=120°)")
    print(f"Target: pomarańczowy (H=30°) - naturalnie bliżej czerwonego")
    print(f"Pomarańczowy RGB: {target_color}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        master_path = os.path.join(temp_dir, "master.png")
        target_path = os.path.join(temp_dir, "target.png") 
        result_without_path = os.path.join(temp_dir, "result_without.png")
        result_with_path = os.path.join(temp_dir, "result_with.png")
        
        Image.fromarray(master).save(master_path)
        Image.fromarray(target).save(target_path)
        
        algorithm = PaletteMappingAlgorithm()
        
        # Test 1: BEZ Color Focus (powinien wybrać czerwony)
        print("\n--- TEST 1: BEZ COLOR FOCUS ---")
        success1 = algorithm.process_images(
            master_path=master_path,
            target_path=target_path, 
            output_path=result_without_path,
            num_colors=2,
            distance_metric="weighted_hsv"
        )
        
        result1 = np.array(Image.open(result_without_path))
        dominant_color1 = result1[0, 0]
        
        # Test 2: Z Color Focus na ZIELONY (powinien zmienić wybór na zielony)
        print("\n--- TEST 2: Z COLOR FOCUS NA ZIELONY (Force) ---")
        focus_config = [{
            "target_hsv": [30, 100, 100],  # Pomarańczowy target
            "range_h": 40,  # Szeroki zakres: 10°-50° (pokrywa pomarańczowy)
            "range_s": 50,  # ±25%
            "range_v": 50,  # ±25%
            "boost_factor": 1000.0  # OGROMNY boost dla zielonego
        }]
        
        # ALE! Musimy sprawdzić które kolory z palety będą w focus range
        # Focus range: H=30° ±20° = 10°-50°
        # Czerwony: H=0° - NIE jest w zakresie (poza 10°-50°)
        # Zielony: H=120° - NIE jest w zakresie  
        # 
        # PROBLEM! Żaden kolor z palety nie jest w focus range!
        # Musimy zmienić strategię: focus na kolor z palety który CHCEMY wybrać
        
        print("Estrategia: Focus na zielony kolor z palety (H=120°)")
        focus_config = [{
            "target_hsv": [120, 100, 100],  # ZIELONY z palety (target color)
            "range_h": 60,   # Szeroki zakres: 90°-150° (pokrywa zielony H=120°)  
            "range_s": 50,   # ±25%
            "range_v": 50,   # ±25%
            "boost_factor": 1000.0  # OGROMNY boost dla zielonego
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
        
        result2 = np.array(Image.open(result_with_path))
        dominant_color2 = result2[0, 0]
        
        # Porównanie
        print(f"\nBez Color Focus: RGB {dominant_color1}")
        print(f"Z Color Focus: RGB {dominant_color2}")
        print(f"Czerwony to: [255, 0, 0]")
        print(f"Zielony to: [0, 255, 0]")
        
        # Sprawdź które kolory zostały wybrane
        is_red1 = np.allclose(dominant_color1, [255, 0, 0], atol=10)
        is_green1 = np.allclose(dominant_color1, [0, 255, 0], atol=10)
        is_red2 = np.allclose(dominant_color2, [255, 0, 0], atol=10)
        is_green2 = np.allclose(dominant_color2, [0, 255, 0], atol=10)
        
        print(f"\nBez focus -> {'Czerwony' if is_red1 else 'Zielony' if is_green1 else 'Inny'}")
        print(f"Z focus -> {'Czerwony' if is_red2 else 'Zielony' if is_green2 else 'Inny'}")
        
        # Test sukcesu
        color_focus_works = not np.array_equal(dominant_color1, dominant_color2)
        expected_result = is_red1 and is_green2  # Bez focus=czerwony, z focus=zielony
        
        print("\n============================================================")
        if color_focus_works and expected_result:
            print("✅ COLOR FOCUS DZIAŁA PERFEKCYJNIE!")
            print("✅ Bez focus: czerwony (naturalny), z focus: zielony (wymuszony)")
        elif color_focus_works:
            print("✅ COLOR FOCUS DZIAŁA! Zmienił wybór koloru.")
            print("⚠️  Ale nie zgodnie z oczekiwaniem")
        else:
            print("❌ COLOR FOCUS NIE DZIAŁA! Identyczne wyniki.")
        print("============================================================")
        
        return color_focus_works


if __name__ == "__main__":
    test_color_focus_decisive()
