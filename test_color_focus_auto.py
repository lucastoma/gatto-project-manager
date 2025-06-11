#!/usr/bin/env python3
"""
Test automatyczny Color Focus z wygenerowanymi matematycznie obrazkami.
Ten test sprawdza czy funkcjonalność Color Focus rzeczywiście działa.
"""

import os
import sys
import numpy as np
from PIL import Image
import tempfile
import shutil

# Dodaj ścieżkę do modułów projektu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm

def rgb_to_hsv_single(r, g, b):
    """Konwersja pojedynczego koloru RGB na HSV"""
    r, g, b = r/255.0, g/255.0, b/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Hue
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation
    s = 0 if max_val == 0 else (diff / max_val) * 100
    
    # Value
    v = max_val * 100
    
    return [h, s, v]

def create_target_hsv_color(target_hsv):
    """Tworzy kolor RGB z docelowego HSV"""
    h, s, v = target_hsv[0], target_hsv[1]/100.0, target_hsv[2]/100.0
    
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
    return [int(r), int(g), int(b)]

def create_test_master_image(size=(200, 200)):
    """
    Tworzy obraz master z paletą kolorów różnych od target_hsv
    """
    master_colors = [
        [255, 0, 0],    # Czerwony
        [0, 255, 0],    # Zielony
        [0, 0, 255],    # Niebieski
        [255, 255, 0],  # Żółty
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [128, 128, 128], # Szary
        [255, 255, 255]  # Biały
    ]
    
    master_array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    colors_per_row = len(master_colors)
    strip_width = size[0] // colors_per_row
    
    for i, color in enumerate(master_colors):
        x_start = i * strip_width
        x_end = min((i + 1) * strip_width, size[0])
        master_array[:, x_start:x_end] = color
    
    return Image.fromarray(master_array)

def create_test_target_image(target_hsv, size=(200, 200)):
    """
    Tworzy obraz target z pojedynczym kolorem odpowiadającym target_hsv
    """
    target_rgb = create_target_hsv_color(target_hsv)
    print(f"Target HSV: {target_hsv} -> RGB: {target_rgb}")
    
    target_array = np.full((size[1], size[0], 3), target_rgb, dtype=np.uint8)
    return Image.fromarray(target_array)

def analyze_result_image(result_path, original_target_rgb, expected_changes):
    """
    Analizuje obraz wynikowy i sprawdza czy Color Focus miał wpływ
    """
    result_image = Image.open(result_path)
    result_array = np.array(result_image)
    
    # Znajdź dominujący kolor w wyniku
    unique_colors, counts = np.unique(result_array.reshape(-1, 3), axis=0, return_counts=True)
    dominant_color_idx = np.argmax(counts)
    dominant_color = unique_colors[dominant_color_idx]
    
    print(f"Original target RGB: {original_target_rgb}")
    print(f"Result dominant RGB: {dominant_color}")
    
    # Sprawdź czy kolor się zmienił
    color_changed = not np.array_equal(original_target_rgb, dominant_color)
    
    # Oblicz różnicę
    color_distance = np.sqrt(np.sum((np.array(original_target_rgb) - np.array(dominant_color))**2))
    
    return {
        'color_changed': color_changed,
        'color_distance': color_distance,
        'original_rgb': original_target_rgb,
        'result_rgb': dominant_color.tolist(),
        'unique_colors_count': len(unique_colors)
    }

def test_color_focus_functionality():
    """
    Główny test - sprawdza czy Color Focus ma wpływ na mapowanie kolorów
    """
    print("=" * 60)
    print("TEST AUTOMATYCZNY COLOR FOCUS")
    print("=" * 60)
    
    # Konfiguracja testu
    target_hsv = [25, 50, 70]  # Pomarańczowy odcień
    focus_ranges = [
        {
            "target_hsv": target_hsv,
            "range_h": 30,
            "range_s": 50, 
            "range_v": 60,
            "boost_factor": 10.0  # Bardzo duży boost dla widocznego efektu
        }
    ]
    
    # Stwórz tymczasowy folder
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Ścieżki plików
        master_path = os.path.join(temp_dir, "master_test.png")
        target_path = os.path.join(temp_dir, "target_test.png")
        result_without_focus_path = os.path.join(temp_dir, "result_without_focus.png")
        result_with_focus_path = os.path.join(temp_dir, "result_with_focus.png")
        
        # Stwórz obrazy testowe
        print("Tworzenie obrazów testowych...")
        master_image = create_test_master_image()
        target_image = create_test_target_image(target_hsv)
        target_rgb = create_target_hsv_color(target_hsv)
        
        master_image.save(master_path)
        target_image.save(target_path)
        
        # Test 1: Bez Color Focus
        print("\n--- TEST 1: BEZ COLOR FOCUS ---")
        algorithm1 = PaletteMappingAlgorithm()
        success1 = algorithm1.process_images(
            master_path=master_path,
            target_path=target_path,
            output_path=result_without_focus_path,
            use_color_focus=False,
            num_colors=8,
            distance_metric="weighted_hsv"
        )
        
        if success1:
            result1 = analyze_result_image(result_without_focus_path, target_rgb, "bez Color Focus")
            print(f"Sukces: {success1}")
            print(f"Kolor się zmienił: {result1['color_changed']}")
            print(f"Odległość kolorów: {result1['color_distance']:.2f}")
            print(f"Ilość unikalnych kolorów: {result1['unique_colors_count']}")
        else:
            print("BŁĄD: Test bez Color Focus nie powiódł się!")
            return False
        
        # Test 2: Z Color Focus
        print("\n--- TEST 2: Z COLOR FOCUS ---")
        algorithm2 = PaletteMappingAlgorithm()
        success2 = algorithm2.process_images(
            master_path=master_path,
            target_path=target_path,
            output_path=result_with_focus_path,
            use_color_focus=True,
            focus_ranges=focus_ranges,
            num_colors=8,
            distance_metric="weighted_hsv"
        )
        
        if success2:
            result2 = analyze_result_image(result_with_focus_path, target_rgb, "z Color Focus")
            print(f"Sukces: {success2}")
            print(f"Kolor się zmienił: {result2['color_changed']}")
            print(f"Odległość kolorów: {result2['color_distance']:.2f}")
            print(f"Ilość unikalnych kolorów: {result2['unique_colors_count']}")
        else:
            print("BŁĄD: Test z Color Focus nie powiódł się!")
            return False
        
        # Porównanie wyników
        print("\n--- PORÓWNANIE WYNIKÓW ---")
        print(f"Bez Color Focus - RGB wyniku: {result1['result_rgb']}")
        print(f"Z Color Focus - RGB wyniku: {result2['result_rgb']}")
        
        # Sprawdź czy wyniki są różne
        results_different = not np.array_equal(result1['result_rgb'], result2['result_rgb'])
        print(f"Wyniki są różne: {results_different}")
        
        if results_different:
            difference = np.sqrt(np.sum((np.array(result1['result_rgb']) - np.array(result2['result_rgb']))**2))
            print(f"Różnica między wynikami: {difference:.2f}")
            
            # WNIOSEK
            print("\n" + "=" * 60)
            if difference > 10:  # Arbitralna granica dla widocznej różnicy
                print("✅ COLOR FOCUS DZIAŁA! Widoczna różnica w wynikach.")
                return True
            else:
                print("⚠️  COLOR FOCUS ma minimalny wpływ. Różnica: {:.2f}".format(difference))
                return True
        else:
            print("\n" + "=" * 60)
            print("❌ COLOR FOCUS NIE DZIAŁA! Wyniki identyczne.")
            return False
            
    except Exception as e:
        print(f"BŁĄD TESTU: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Wyczyść pliki tymczasowe
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    success = test_color_focus_functionality()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
