#!/usr/bin/env python3
"""
Test zaawansowany Color Focus z wielokolorowym obrazem target
"""

import numpy as np
from PIL import Image
import tempfile
import os
import sys

# Dodaj projekt do ścieżki
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm


def create_multicolor_test_images():
    """Tworzy bardziej złożone obrazy testowe"""
    
    # Master: 4 czyste kolory w kwadratach
    master = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Kwadrat czerwony (góra-lewo)
    master[0:100, 0:100] = [255, 0, 0]  # Czerwony: H=0°
    
    # Kwadrat zielony (góra-prawo) 
    master[0:100, 100:200] = [0, 255, 0]  # Zielony: H=120°
    
    # Kwadrat niebieski (dół-lewo)
    master[100:200, 0:100] = [0, 0, 255]  # Niebieski: H=240°
    
    # Kwadrat żółty (dół-prawo)
    master[100:200, 100:200] = [255, 255, 0]  # Żółty: H=60°
    
    # Target: gradient od czerwonego do zielonego przez pomarańczowy
    target = np.zeros((200, 200, 3), dtype=np.uint8)
    
    for x in range(200):
        # Gradient od czerwonego (H=0°) do żółtego (H=60°) do zielonego (H=120°)
        if x < 100:
            # Czerwony -> Żółty (H: 0° -> 60°)
            ratio = x / 100.0
            h = ratio * 60  # 0° do 60°
            target[:, x] = hsv_to_rgb(h, 100, 100)
        else:
            # Żółty -> Zielony (H: 60° -> 120°)
            ratio = (x - 100) / 100.0
            h = 60 + ratio * 60  # 60° do 120°
            target[:, x] = hsv_to_rgb(h, 100, 100)
    
    return master, target


def hsv_to_rgb(h, s, v):
    """Konwersja HSV do RGB"""
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


def analyze_image_colors(image_path):
    """Analizuje kolory w obrazie i zwraca statystyki"""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Flatten i znajdź unikalne kolory
    pixels = img_array.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    
    # Policz wystąpienia każdego koloru
    color_counts = {}
    for color in unique_colors:
        mask = np.all(pixels == color, axis=1)
        count = np.sum(mask)
        color_counts[tuple(color)] = count
    
    # Sortuj po częstości
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'unique_colors': len(unique_colors),
        'most_common': sorted_colors[:5],  # Top 5 najczęstszych
        'all_colors': dict(color_counts)
    }


def test_color_focus_advanced():
    """Test zaawansowany z wielokolorowym obrazem"""
    
    print("============================================================")
    print("TEST ZAAWANSOWANY COLOR FOCUS - WIELOKOLOROWY OBRAZ")
    print("============================================================")
    
    # Tworzenie obrazów
    print("Tworzenie wielokolorowych obrazów testowych...")
    master_array, target_array = create_multicolor_test_images()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        master_path = os.path.join(temp_dir, "master_multi.png")
        target_path = os.path.join(temp_dir, "target_multi.png") 
        result_without_path = os.path.join(temp_dir, "result_without_focus.png")
        result_with_path = os.path.join(temp_dir, "result_with_focus.png")
        
        # Zapisanie obrazów
        Image.fromarray(master_array).save(master_path)
        Image.fromarray(target_array).save(target_path)
        
        print(f"Master: 4 kolory (czerwony, zielony, niebieski, żółty)")
        print(f"Target: gradient czerwony->żółty->zielony (H: 0°->120°)")
        
        # Algorytm
        algorithm = PaletteMappingAlgorithm()
        
        # Test 1: BEZ Color Focus
        print("\n--- TEST 1: BEZ COLOR FOCUS ---")
        success1 = algorithm.process_images(
            master_path=master_path,
            target_path=target_path, 
            output_path=result_without_path,
            num_colors=4,
            distance_metric="weighted_hsv",
            use_color_focus=False
        )
        
        stats1 = analyze_image_colors(result_without_path)
        print(f"Sukces: {success1}")
        print(f"Ilość unikalnych kolorów: {stats1['unique_colors']}")
        print(f"Top 3 kolory: {stats1['most_common'][:3]}")
        
        # Test 2: Z Color Focus na żółty (H=60°, target obszar czerwony->żółty)
        print("\n--- TEST 2: Z COLOR FOCUS NA ŻÓŁTY (H=60°) ---")
        focus_config = [{
            "target_hsv": [60, 100, 100],  # Żółty
            "range_h": 40,  # Szeroki zakres ±20°: 40°-80° (pokrywa czerwony->żółty)
            "range_s": 50,  # ±25%
            "range_v": 50,  # ±25% 
            "boost_factor": 20.0  # Bardzo wysoki boost
        }]
        
        success2 = algorithm.process_images(
            master_path=master_path,
            target_path=target_path,
            output_path=result_with_path,
            num_colors=4,
            distance_metric="weighted_hsv", 
            use_color_focus=True,
            focus_ranges=focus_config
        )
        
        stats2 = analyze_image_colors(result_with_path)
        print(f"Sukces: {success2}")
        print(f"Ilość unikalnych kolorów: {stats2['unique_colors']}")
        print(f"Top 3 kolory: {stats2['most_common'][:3]}")
        
        # Porównanie wyników
        print("\n--- PORÓWNANIE WYNIKÓW ---")
        print(f"Bez Color Focus - top kolory: {[color for color, count in stats1['most_common'][:3]]}")
        print(f"Z Color Focus - top kolory: {[color for color, count in stats2['most_common'][:3]]}")
        
        # Sprawdź czy żółty kolor pojawił się częściej z Color Focus
        yellow_rgb = (255, 255, 0)  # Żółty z palety master
        red_rgb = (255, 0, 0)      # Czerwony z palety master
        
        yellow_count1 = stats1['all_colors'].get(yellow_rgb, 0)
        yellow_count2 = stats2['all_colors'].get(yellow_rgb, 0)
        red_count1 = stats1['all_colors'].get(red_rgb, 0) 
        red_count2 = stats2['all_colors'].get(red_rgb, 0)
        
        print(f"\nLiczniki kolorów:")
        print(f"Żółty (255,255,0) - bez focus: {yellow_count1}, z focus: {yellow_count2}")
        print(f"Czerwony (255,0,0) - bez focus: {red_count1}, z focus: {red_count2}")
        
        # Sprawdź czy Color Focus wpłynął na wynik
        color_focus_works = (yellow_count2 > yellow_count1) or (stats1['most_common'] != stats2['most_common'])
        
        print(f"\nDistribution różna: {stats1['most_common'] != stats2['most_common']}")
        print(f"Więcej żółtego z focus: {yellow_count2 > yellow_count1}")
        
        print("\n============================================================")
        if color_focus_works:
            print("✅ COLOR FOCUS DZIAŁA! Zmienił dystrybucję kolorów.")
        else:
            print("❌ COLOR FOCUS NIE DZIAŁA! Brak różnic w wynikach.")
        print("============================================================")
        
        return color_focus_works


if __name__ == "__main__":
    test_color_focus_advanced()
