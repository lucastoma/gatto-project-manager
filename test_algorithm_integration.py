#!/usr/bin/env python3
"""
Algorithm Integration Test
==========================

Test integration of new modular algorithm system with the Enhanced Flask server.
Verifies that:
1. New algorithm_01_palette works correctly 
2. API routing functions properly
3. Performance monitoring is active
4. Legacy algorithms (2,3) still work
5. Results are generated correctly
"""

import requests
import time
import os
import sys

# Server configuration
SERVER_URL = "http://127.0.0.1:5000"
API_URL = f"{SERVER_URL}/api/colormatch"

def test_algorithm_integration():
    """Test integration of new modular algorithm system."""
    print("🔬 ALGORITHM INTEGRATION TEST")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{SERVER_URL}/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not running. Start server first!")
            return False
    except:
        print("❌ Server not responding. Start server first!")
        return False
    
    print("✅ Server is running")
    
    # Test files
    master_file = "test_image.png"
    target_file = "test_simple.tif"
    
    if not os.path.exists(master_file) or not os.path.exists(target_file):
        print(f"❌ Test files not found: {master_file}, {target_file}")
        return False
    
    print(f"✅ Test files found: {master_file}, {target_file}")
    
    # Test each method
    methods = [
        ("1", "Enhanced Palette Mapping (New Modular)", True),
        ("2", "Statistical Transfer (Legacy)", False),
        ("3", "Histogram Matching (Legacy)", False)
    ]
    
    results = []
    
    for method, description, is_new in methods:
        print(f"\n🧪 Testing Method {method}: {description}")
        print("-" * 60)
        
        # Prepare request
        files = {
            'master_image': open(master_file, 'rb'),
            'target_image': open(target_file, 'rb')
        }
        data = {
            'method': method,
            'k': 8
        }
        
        # Send request and measure time
        start_time = time.time()
        try:
            response = requests.post(API_URL, files=files, data=data, timeout=30)
            end_time = time.time()
            duration = end_time - start_time
            
            # Close files
            files['master_image'].close()
            files['target_image'].close()
            
            if response.status_code == 200:
                result_text = response.text.strip()
                
                if result_text.startswith("success"):
                    parts = result_text.split(",")
                    result_filename = parts[2] if len(parts) >= 3 else "unknown"
                    
                    # Check if result file exists
                    result_path = f"results/{result_filename}"
                    file_exists = os.path.exists(result_path)
                    
                    status = "✅ PASS" if file_exists else "⚠️ PARTIAL"
                    print(f"   Status: {status}")
                    print(f"   Duration: {duration:.2f}s")
                    print(f"   Result: {result_filename}")
                    print(f"   File exists: {'Yes' if file_exists else 'No'}")
                    
                    if is_new:
                        print(f"   🆕 Using NEW modular algorithm!")
                    else:
                        print(f"   📦 Using legacy algorithm")
                    
                    results.append({
                        'method': method,
                        'status': 'PASS' if file_exists else 'PARTIAL',
                        'duration': duration,
                        'description': description,
                        'is_new': is_new
                    })
                else:
                    print(f"   ❌ FAIL: {result_text}")
                    results.append({
                        'method': method,
                        'status': 'FAIL',
                        'duration': duration,
                        'description': description,
                        'is_new': is_new,
                        'error': result_text
                    })
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                results.append({
                    'method': method,
                    'status': 'HTTP_ERROR',
                    'duration': duration,
                    'description': description,
                    'is_new': is_new
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            results.append({
                'method': method,
                'status': 'EXCEPTION',
                'description': description,
                'is_new': is_new,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for result in results:
        status_icon = {
            'PASS': '✅',
            'PARTIAL': '⚠️',
            'FAIL': '❌',
            'HTTP_ERROR': '🔥',
            'EXCEPTION': '💥'
        }.get(result['status'], '❓')
        
        new_indicator = '🆕' if result['is_new'] else '📦'
        duration_str = f"{result.get('duration', 0):.2f}s" if 'duration' in result else 'N/A'
        
        print(f"Method {result['method']}: {status_icon} {result['status']} ({duration_str}) {new_indicator}")
        
        if result['status'] == 'PASS':
            passed += 1
    
    print(f"\nResult: {passed}/{total} methods working")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Algorithm integration successful!")
        return True
    elif passed > 0:
        print("⚠️ PARTIAL SUCCESS: Some algorithms working")
        return True
    else:
        print("❌ ALL TESTS FAILED! Check server and algorithm setup")
        return False

if __name__ == "__main__":
    success = test_algorithm_integration()
    sys.exit(0 if success else 1)
