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

# Konfiguracja
SERVER_URL = "http://127.0.0.1:5000"
TEST_IMAGES_DIR = "test_images"
RESULTS_DIR = "test_results"

def setup_test_environment():
    """Przygotuj środowisko testowe"""
    # Stwórz katalogi jeśli nie istnieją
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Sprawdź czy mamy obrazy testowe
    test_files = ["test_image.png", "test_simple.tif"]
    available_files = []
    
    for file in test_files:
        if os.path.exists(file):
            available_files.append(file)
    
    if len(available_files) < 2:
        print("[ERROR] Potrzebne co najmniej 2 obrazy testowe")
        print(f"Dostępne: {available_files}")
        return None
    
    return available_files[:2]  # Użyj pierwszych dwóch

def test_method(method_num, master_path, target_path, k_colors=8):
    """Test pojedynczej metody"""
    print(f"\n[TEST] Testowanie Metody {method_num}...")
    
    start_time = time.time()
    
    try:
        # Przygotuj pliki
        with open(master_path, 'rb') as f1, open(target_path, 'rb') as f2:
            files = {
                'master_image': f1,
                'target_image': f2
            }
            data = {
                'method': str(method_num),
                'k': k_colors
            }
            
            # Wyślij request
            response = requests.post(f"{SERVER_URL}/api/colormatch", files=files, data=data)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Sprawdź odpowiedź
        if response.status_code == 200:
            result = response.text.strip()
            if result.startswith("success"):
                parts = result.split(",")
                if len(parts) >= 3:
                    result_filename = parts[2]
                    print(f"[PASS] Metoda {method_num}: SUKCES")
                    print(f"   Czas: {execution_time:.2f}s")
                    print(f"   Wynik: {result_filename}")
                    return True, execution_time
                else:
                    print(f"[FAIL] Metoda {method_num}: Nieprawidłowy format odpowiedzi")
            else:
                print(f"[FAIL] Metoda {method_num}: {result}")
        else:
            print(f"[FAIL] Metoda {method_num}: HTTP {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Metoda {method_num}: Nie można połączyć z serwerem")
        print("   Upewnij się, że serwer działa: python run_server.py")
    except Exception as e:
        print(f"[FAIL] Metoda {method_num}: Błąd - {str(e)}")
    
    return False, 0

def check_server():
    """Sprawdź czy serwer działa"""
    import socket
    try:
        # Sprawdź czy port jest otwarty
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 5000))
        sock.close()
        
        if result == 0:
            print("[OK] Port 5000 jest otwarty")
            return True
        else:
            print(f"[ERROR] Port 5000 nie odpowiada (kod: {result})")
            return False
    except Exception as e:
        print(f"[ERROR] Błąd sprawdzania portu: {e}")
        return False

def main():
    """Główna funkcja testowa"""
    print("POZIOM 1: Test Podstawowych Metod Color Matching")
    print("=" * 50)
    
    # Sprawdź serwer
    if not check_server():
        print("[ERROR] Serwer nie działa!")
        print("Uruchom serwer: python run_server.py")
        return

    print("[OK] Serwer działa")
    
    # Przygotuj środowisko
    test_files = setup_test_environment()
    if not test_files:
        return

    master_file, target_file = test_files
    print(f"[INFO] Master: {master_file}")
    print(f"[INFO] Target: {target_file}")
    
    # Test wszystkich metod
    methods = [
        (1, "Simple Palette Mapping (RGB K-means)"),
        (2, "Basic Statistical Transfer (LAB)"),
        (3, "Simple Histogram Matching (Luminancja)")
    ]
    
    results = []
    total_time = 0
    
    for method_num, method_name in methods:
        print(f"\n[INFO] {method_name}")
        success, exec_time = test_method(method_num, master_file, target_file)
        results.append((method_num, method_name, success, exec_time))
        total_time += exec_time

    # Podsumowanie
    print("\n" + "=" * 50)
    print("PODSUMOWANIE TESTÓW")
    print("=" * 50)

    successful_methods = 0
    for method_num, method_name, success, exec_time in results:
        status = "[PASS]" if success else "[FAIL]"
        time_status = "[FAST]" if exec_time < 5.0 else "[SLOW]"
        print(f"Metoda {method_num}: {status} ({exec_time:.2f}s) {time_status}")
        if success:
            successful_methods += 1
    
    print(f"\nCałkowity czas: {total_time:.2f}s")
    print(f"Sukces: {successful_methods}/3 metod")
    
    # Kryterium sukcesu
    if successful_methods == 3:
        print("\n[SUCCESS] POZIOM 1: ZALICZONY!")
        print("Wszystkie metody działają bez błędów")
        if total_time < 15.0:  # 3 metody * 5s = 15s
            print("[BONUS] Wydajność w normie!")
    else:
        print("\n[FAILED] POZIOM 1: NIEZALICZONY")
        print("Nie wszystkie metody działają poprawnie")

if __name__ == "__main__":
    main()