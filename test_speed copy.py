#!/usr/bin/env python3
# Test script dla color matching endpoint

import requests
import os
import sys
import time

# Dodaj ścieżkę do modułu app
sys.path.append('.')

from app.processing.color_matching import simple_palette_mapping

def test_speed():
    """Test speed z obrazami z folderu source"""
    
    # Sprawdź folder source
    source_folder = "source"
    if not os.path.exists(source_folder):
        print(f"❌ Brak folderu: {source_folder}")
        return
    
    # Znajdź obrazy w folderze source
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        image_files.extend([f for f in os.listdir(source_folder) if f.lower().endswith(ext)])
    
    if len(image_files) < 2:
        print(f"❌ Potrzeba przynajmniej 2 obrazów w folderze {source_folder}")
        print(f"   Znalezione: {image_files}")
        return
    
    # Wybierz pierwsze 2 obrazy
    master_path = os.path.join(source_folder, image_files[0])
    target_path = os.path.join(source_folder, image_files[1])
    
    print(f"🚀 SPEED TEST - METHOD 1 OPTIMIZED")
    print(f"Master: {master_path}")
    print(f"Target: {target_path}")
    print(f"Colors: 8")
    print("-" * 50)
    
    try:
        # Test nowej optimized version
        start_time = time.time()
        result_path = simple_palette_mapping(master_path, target_path, k_colors=8)
        total_time = time.time() - start_time
        
        print("-" * 50)
        print(f"🎯 FINAL RESULT:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Result file: {result_path}")
        
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024*1024)  # MB
            print(f"   File size: {file_size:.1f}MB")
            print("✅ SUCCESS! File created.")
        else:
            print("❌ File not created!")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_speed()
