#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POZIOM 1: Podstawowy test trzech metod color matching
Cel: <5 sekund na 1MP, wszystkie metody działają bez błędów
"""

import time
import os
import requests
import shutil
from pathlib import Path
from PIL import Image # Added for dummy image creation

# Konfiguracja
SERVER_URL = "http://127.0.0.1:5000"
TEST_IMAGES_DIR = "test_images"
RESULTS_DIR = "test_results"

def setup_test_environment():
    """Prepare the test environment."""
    # Create directories if they don't exist
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Create dummy test images if they don't exist
    dummy_image_path_png = "test_image.png"
    dummy_image_path_tif = "test_simple.tif"

    if not os.path.exists(dummy_image_path_png):
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(dummy_image_path_png)
        print(f"[INFO] Created dummy image: {dummy_image_path_png}")

    if not os.path.exists(dummy_image_path_tif):
        img = Image.new('RGB', (100, 100), color = 'blue')
        img.save(dummy_image_path_tif)
        print(f"[INFO] Created dummy image: {dummy_image_path_tif}")
    
    return [dummy_image_path_png, dummy_image_path_tif]

def test_method(method_num, master_path, target_path, k_colors=16, distance_metric=None, use_dithering=False, preserve_luminance=False, is_preview=False):
    """Test pojedynczej metody"""
    print(f"\n[TEST] Testing Method {method_num}...")
    
    start_time = time.time()
    
    try:
        # Prepare files
        files = {
            'master_image': open(master_path, 'rb'),
            'target_image': open(target_path, 'rb')
        }
        
        data = {
            'method': str(method_num),
            'k': k_colors,
            'use_dithering': str(use_dithering).lower(),
            'preserve_luminance': str(preserve_luminance).lower()
        }
        if distance_metric:
            data['distance_metric'] = distance_metric

        url = f"{SERVER_URL}/api/colormatch"
        if is_preview:
            url = f"{SERVER_URL}/api/colormatch/preview"
            
        # Send request
        response = requests.post(url, files=files, data=data)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Close file handles
        files['master_image'].close()
        files['target_image'].close()

        # Check response
        if response.status_code == 200:
            result = response.text.strip()
            if result.startswith("success"):
                parts = result.split(",")
                if len(parts) >= 3:
                    result_filename = parts[2]
                    print(f"[PASS] Method {method_num}: SUCCESS")
                    print(f"   Time: {execution_time:.2f}s")
                    print(f"   Result: {result_filename}")
                    return True, execution_time
                else:
                    print(f"[FAIL] Method {method_num}: Invalid response format")
            else:
                print(f"[FAIL] Method {method_num}: {result}")
        else:
            print(f"[FAIL] Method {method_num}: HTTP {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Method {method_num}: Cannot connect to server")
        print("   Ensure the server is running: python run_server.py")
    except Exception as e:
        print(f"[FAIL] Method {method_num}: Error - {str(e)}")
    
    return False, 0

def check_server():
    """Check if the server is running"""
    import socket
    try:
        # Check if port is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 5000))
        sock.close()
        
        if result == 0:
            print("[OK] Port 5000 is open")
            return True
        else:
            print(f"[ERROR] Port 5000 is not responding (code: {result})")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking port: {e}")
        return False

def main():
    """Main test function"""
    print("LEVEL 1: Basic Color Matching Methods Test")
    print("=" * 50)
    
    # Check server
    if not check_server():
        print("[ERROR] Server is not running!")
        print("Start the server: python run_server.py")
        return

    print("[OK] Server is running")
    
    # Prepare environment
    test_files = setup_test_environment()
    if not test_files:
        return

    master_file, target_file = test_files
    print(f"[INFO] Master: {master_file}")
    print(f"[INFO] Target: {target_file}")
    
    # Test all methods
    methods_to_test = [
        (1, "Simple Palette Mapping (RGB K-means)", {}, False),
        (2, "Basic Statistical Transfer (LAB)", {}, False),
        (3, "Simple Histogram Matching (Luminance)", {}, False),
        (1, "Palette Mapping (LAB, Dithering, Preserve Luminance)", {'distance_metric': 'lab', 'use_dithering': True, 'preserve_luminance': True}, False),
        (1, "Palette Mapping Preview (LAB, Dithering)", {'distance_metric': 'lab', 'use_dithering': True}, True)
    ]
    
    results = []
    total_time = 0
    
    for method_num, method_name, params, is_preview in methods_to_test:
        print(f"\n[INFO] {method_name}")
        success, exec_time = test_method(method_num, master_file, target_file, **params, is_preview=is_preview)
        results.append((method_num, method_name, success, exec_time))
        total_time += exec_time

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    successful_methods = 0
    for method_num, method_name, success, exec_time in results:
        status = "[PASS]" if success else "[FAIL]"
        time_status = "[FAST]" if exec_time < 5.0 else "[SLOW]"
        print(f"Method {method_num}: {method_name}: {status} ({exec_time:.2f}s) {time_status}")
        if success:
            successful_methods += 1
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Success: {successful_methods}/{len(methods_to_test)} methods")
    
    # Success criteria
    if successful_methods == len(methods_to_test):
        print("\n[SUCCESS] LEVEL 1: PASSED!")
        print("All methods work without errors")
        if total_time < 25.0:  # Adjusted total time for more tests
            print("[BONUS] Performance within limits!")
    else:
        print("\n[FAILED] LEVEL 1: FAILED")
        print("Not all methods work correctly")

if __name__ == "__main__":
    main()
