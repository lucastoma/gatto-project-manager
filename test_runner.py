#!/usr/bin/env python3
"""
Test Runner - Automatyczne uruchamianie testów z zarządzaniem serwerem

Użycie:
    python test_runner.py              # Uruchom wszystkie testy
    python test_runner.py --auto-start # Automatycznie uruchom serwer jeśli nie działa
    python test_runner.py --stop-after # Zatrzymaj serwer po testach
"""

import sys
import argparse
import time
from server_manager_enhanced import EnhancedServerManager

def run_tests_with_management(auto_start=False, stop_after=False):
    """Uruchom testy z zarządzaniem serwerem"""
    manager = EnhancedServerManager()
    server_was_running = manager.is_running()
    
    print("=== Test Runner ===")
    print(f"Auto-start: {auto_start}")
    print(f"Stop after: {stop_after}")
    print()
    
    # Sprawdź status serwera
    if server_was_running:
        print("[INFO] Serwer już działa")
    else:
        print("[INFO] Serwer nie działa")
        if auto_start:
            print("[INFO] Uruchamiam serwer automatycznie...")
            if not manager.start_server():
                print("[ERROR] Nie udało się uruchomić serwera")
                return False
        else:
            print("[ERROR] Serwer nie działa. Użyj --auto-start lub uruchom serwer ręcznie.")
            print("[INFO] Komenda: python server_manager.py start")
            return False
    
    print()
    
    # Uruchom testy
    print("=== Uruchamiam testy ===")
    success = manager.run_tests()
    
    # Zatrzymaj serwer jeśli trzeba
    if stop_after and (auto_start or not server_was_running):
        print("\n=== Zatrzymuję serwer ===")
        manager.stop_server()
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Test Runner z zarządzaniem serwerem')
    parser.add_argument('--auto-start', action='store_true', 
                       help='Automatycznie uruchom serwer jeśli nie działa')
    parser.add_argument('--stop-after', action='store_true',
                       help='Zatrzymaj serwer po testach')
    
    args = parser.parse_args()
    
    success = run_tests_with_management(
        auto_start=args.auto_start,
        stop_after=args.stop_after
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()