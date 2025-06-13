#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTERAKTYWNY SELEKTOR PLIKÓW KONFIGURACYJNYCH
Skrypt do wyboru i uruchamiania różnych konfiguracji .comb-scripts
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml

def load_config_info(config_path):
    """Wczytuje podstawowe informacje z pliku konfiguracyjnego."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        project_name = config.get('project_name', 'Nieznany projekt')
        output_file = config.get('output_file', 'Nieznany plik wyjściowy')
        groups_count = len(config.get('groups', []))
        
        return {
            'project_name': project_name,
            'output_file': output_file,
            'groups_count': groups_count,
            'valid': True
        }
    except Exception as e:
        return {
            'project_name': 'BŁĄD ODCZYTU',
            'output_file': f'Błąd: {str(e)}',
            'groups_count': 0,
            'valid': False
        }

def get_config_files():
    """Znajduje wszystkie pliki konfiguracyjne YAML w katalogu config-lists."""
    script_dir = Path(__file__).parent
    config_lists_dir = script_dir / 'config-lists'
    config_files = []
    
    # Sprawdź czy katalog config-lists istnieje
    if not config_lists_dir.exists():
        print(f"⚠️  Katalog config-lists nie istnieje: {config_lists_dir}")
        return config_files
    
    # Szukaj plików .yaml i .yml w katalogu config-lists
    for pattern in ['*.yaml', '*.yml']:
        for config_file in config_lists_dir.glob(pattern):
            if 'config' in config_file.name.lower():
                config_files.append(config_file)
    
    return sorted(config_files)

def display_config_list(config_files):
    """Wyświetla listę dostępnych plików konfiguracyjnych."""
    print("\n" + "="*80)
    print("📋 DOSTĘPNE PLIKI KONFIGURACYJNE")
    print("="*80)
    
    for i, config_file in enumerate(config_files, 1):
        info = load_config_info(config_file)
        
        print(f"\n[{i}] {config_file.name}")
        print(f"    📁 Ścieżka: {config_file}")
        print(f"    📝 Projekt: {info['project_name']}")
        print(f"    📄 Wyjście: {info['output_file']}")
        print(f"    📊 Grup: {info['groups_count']}")
        
        if not info['valid']:
            print(f"    ⚠️  Status: BŁĄD KONFIGURACJI")
        else:
            print(f"    ✅ Status: OK")
    
    print("\n" + "="*80)

def run_script_with_config(config_file):
    """Uruchamia skrypt .comb-scripts-v6.py z wybraną konfiguracją."""
    script_dir = Path(__file__).parent
    main_script = script_dir / '.comb-scripts-v6.py'
    export_dir = script_dir / 'export'
    
    if not main_script.exists():
        print(f"❌ BŁĄD: Nie znaleziono skryptu {main_script}")
        return False
    
    # Upewnij się, że katalog export istnieje
    export_dir.mkdir(exist_ok=True)
    
    try:
        print(f"\n🚀 Uruchamiam skrypt z konfiguracją: {config_file.name}")
        print(f"📝 Komenda: python {main_script.name} {str(config_file)} {str(export_dir)}")
        print("-" * 60)
        
        # Uruchom skrypt z pełną ścieżką do pliku konfiguracyjnego i katalogu export
        result = subprocess.run(
            [sys.executable, str(main_script), str(config_file), str(export_dir)],
            cwd=script_dir,
            capture_output=False,
            text=True
        )
        
        print("-" * 60)
        if result.returncode == 0:
            print("✅ Skrypt zakończony pomyślnie!")
            return True
        else:
            print(f"❌ Skrypt zakończony z błędem (kod: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ BŁĄD URUCHAMIANIA: {e}")
        return False

def main():
    """Główna funkcja programu."""
    print("🔧 INTERAKTYWNY SELEKTOR KONFIGURACJI COMB-SCRIPTS")
    
    # Znajdź pliki konfiguracyjne
    config_files = get_config_files()
    
    if not config_files:
        print("❌ Nie znaleziono żadnych plików konfiguracyjnych!")
        return
    
    while True:
        # Wyświetl listę
        display_config_list(config_files)
        
        # Opcje wyboru
        print("\n🎯 OPCJE:")
        for i in range(1, len(config_files) + 1):
            config_info = load_config_info(config_files[i-1])
            print(f"  {i} - {config_info['project_name']}")
        print(f"  0 - Wyjście")
        print(f"  r - Odśwież listę")
        
        # Pobierz wybór użytkownika
        try:
            choice = input("\n👉 Wybierz opcję: ").strip().lower()
            
            if choice == '0' or choice == 'q' or choice == 'quit':
                print("👋 Do widzenia!")
                break
            elif choice == 'r' or choice == 'refresh':
                config_files = get_config_files()
                continue
            else:
                choice_num = int(choice)
                if 1 <= choice_num <= len(config_files):
                    selected_config = config_files[choice_num - 1]
                    run_script_with_config(selected_config)
                    
                    # Zapytaj czy kontynuować
                    cont = input("\n❓ Chcesz wybrać inną konfigurację? (t/n): ").strip().lower()
                    if cont not in ['t', 'tak', 'y', 'yes']:
                        break
                else:
                    print(f"❌ Nieprawidłowy wybór: {choice}")
                    
        except ValueError:
            print(f"❌ Nieprawidłowy wybór: {choice}")
        except KeyboardInterrupt:
            print("\n\n👋 Przerwano przez użytkownika")
            break
        except Exception as e:
            print(f"❌ Nieoczekiwany błąd: {e}")

if __name__ == "__main__":
    main()