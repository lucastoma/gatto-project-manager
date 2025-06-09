#!/usr/bin/env python3
# Prosty test curl dla color matching

import os
import subprocess
import sys

def test_curl():
    """Test curl command line dla color matching endpoint"""
    
    # Sprawdź czy są obrazy do testów
    source_folder = "source"
    if not os.path.exists(source_folder):
        print(f"❌ Brak folderu: {source_folder}")
        return
    
    # Znajdź obrazy
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        image_files.extend([f for f in os.listdir(source_folder) if f.lower().endswith(ext)])
    
    if len(image_files) < 2:
        print(f"❌ Potrzeba przynajmniej 2 obrazów w folderze {source_folder}")
        return
    
    master_path = os.path.join(source_folder, image_files[0])
    target_path = os.path.join(source_folder, image_files[1])
    
    print(f"🚀 CURL TEST")
    print(f"Master: {master_path}")
    print(f"Target: {target_path}")
    print("-" * 40)
    
    # Stwórz curl command
    curl_cmd = [
        'curl', '-s', '-X', 'POST',
        '-F', f'master_image=@{master_path}',
        '-F', f'target_image=@{target_path}',
        '-F', 'method=1',
        '-F', 'k=6',
        'http://127.0.0.1:5000/api/colormatch'
    ]
    
    try:
        print("📡 Wysyłam request...")
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=60)
        
        print(f"Return code: {result.returncode}")
        print(f"Response: {result.stdout}")
        
        if result.stderr:
            print(f"Error: {result.stderr}")
            
        # Parsuj odpowiedź
        if result.returncode == 0 and result.stdout:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3 and parts[0] == 'success':
                print(f"✅ SUCCESS!")
                print(f"Method: {parts[1]}")
                print(f"Result: {parts[2]}")
                
                # Sprawdź czy plik wynikowy istnieje
                result_path = f"results/{parts[2]}"
                if os.path.exists(result_path):
                    size_mb = os.path.getsize(result_path) / (1024*1024)
                    print(f"✅ File created: {result_path} ({size_mb:.1f}MB)")
                else:
                    print(f"❌ File not found: {result_path}")
            else:
                print(f"❌ Invalid response format")
        
    except subprocess.TimeoutExpired:
        print("❌ Request timeout (60s)")
    except FileNotFoundError:
        print("❌ curl command not found. Install curl.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_curl()
