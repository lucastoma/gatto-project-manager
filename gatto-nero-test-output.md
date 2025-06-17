# Projekt: GattoNeroPhotoshop Test Project
## Katalog główny: `/home/lukasz/projects/gatto-project-manager`
## Łączna liczba unikalnych plików: 27
---
## Grupa: Python Scripts
**Opis:** Wszystkie skrypty Python w projekcie.
**Liczba plików w grupie:** 26

### Lista plików:
- `apps/collector/config-selector.py`
- `apps/collector/views.py`
- `apps/collector/__init__.py`
- `apps/collector/advanced_context_collector.py`
- `apps/collector/tests.py`
- `apps/collector/admin.py`
- `apps/collector/migrations/__init__.py`
- `apps/collector/models.py`
- `apps/collector/quick_context_collector.py`
- `apps/collector/management/commands/run_collector.py`
- `apps/collector/apps.py`
- `config/__init__.py`
- `config/urls.py`
- `config/wsgi.py`
- `config/asgi.py`
- `testing-testing/collector/config-selector.py`
- `testing-testing/collector/views.py`
- `testing-testing/collector/__init__.py`
- `testing-testing/collector/advanced_context_collector.py`
- `testing-testing/collector/tests.py`
- `testing-testing/collector/admin.py`
- `testing-testing/collector/migrations/__init__.py`
- `testing-testing/collector/models.py`
- `testing-testing/collector/quick_context_collector.py`
- `testing-testing/collector/management/commands/run_collector.py`
- `testing-testing/collector/apps.py`

### Zawartość plików:
#### Plik: `apps/collector/config-selector.py`
```py
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
    """Uruchamia skrypt advanced_context_collector.py z wybraną konfiguracją."""
    script_dir = Path(__file__).parent
    main_script = script_dir / 'advanced_context_collector.py'
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
```
#### Plik: `apps/collector/views.py`
```py
from django.shortcuts import render

# Create your views here.
```
#### Plik: `apps/collector/__init__.py`
```py

```
#### Plik: `apps/collector/advanced_context_collector.py`
```py
import os
import yaml
from pathlib import Path
import fnmatch
import re
import logging
import tempfile
import base64
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time
from repomix import RepoProcessor, RepomixConfig

# =================================================================================
# SCRIPT FOR FILE AGGREGATION WITH GROUPS AND EXCLUDE PATTERNS (REPOMIX INTEGRATION)
#
# Wersja: 9.0 (z obsługą formatu XML i Markdown)
# Opis: Skrypt wykorzystuje Repomix do przetwarzania plików, a następnie
#       generuje plik wyjściowy w formacie XML lub Markdown na podstawie
#       konfiguracji YAML.
# =================================================================================

DEFAULT_CONFIG_FILE = ".doc-gen/config-lists/.comb-scripts-config01.yaml"

def get_default_repomix_options():
    """Zwraca domyślne opcje konfiguracyjne dla Repomix."""
    return {
        "style": "xml",
        "remove_comments": False,
        "remove_empty_lines": False,
        "show_line_numbers": False,
        "calculate_tokens": True,
        "show_file_stats": True,
        "show_directory_structure": True,
        "top_files_length": 2,
        "copy_to_clipboard": False,
        "include_empty_directories": False,
        "compression": {
            "enabled": False,
            "keep_signatures": True,
            "keep_docstrings": True,
            "keep_interfaces": True,
        },
        "security_check": True,
    }

def get_workspace_root():
    """Zwraca ścieżkę do workspace root."""
    return Path(__file__).parent.parent.parent

def load_config(config_file_path):
    """Wczytuje konfigurację z pliku YAML."""
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"BŁĄD Wczytywania konfiguracji: {e}")
        return None

def process_group_with_repomix(group, workspace_root, processed_files_set, config):
    """
    Przetwarza grupę plików używając Python Repomix, z uwzględnieniem deduplikacji.
    Zwraca listę ścieżek do unikalnych plików oraz ich zawartość.
    """
    group_name = group.get("name", "Unnamed Group")
    patterns = group.get("patterns", [])
    exclude_patterns = group.get("exclude_patterns", [])
    paths = group.get("paths", [])
    logging.info(f"\nPrzetwarzanie grupy: {group_name}")
    group_files_content = []
    unique_files_in_group = []
    repomix_opts = get_default_repomix_options()
    if "repomix_global_options" in config:
        repomix_opts.update(config["repomix_global_options"])
    if "repomix_options" in group:
        group_opts = group["repomix_options"]
        if "compression" in group_opts:
            repomix_opts["compression"].update(group_opts["compression"])
            group_opts = group_opts.copy()
            del group_opts["compression"]
        repomix_opts.update(group_opts)
    for path_str in paths:
        target_path = (
            workspace_root / path_str
            if path_str not in ["all", ".", "**/*", "**"]
            else workspace_root
        )
        if not target_path.exists():
            logging.warning(f"  UWAGA: Ścieżka '{path_str}' nie istnieje i została pominięta.")
            continue
        try:
            repomix_config = RepomixConfig()
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".xml", encoding="utf-8"
            ) as temp_output_file:
                temp_output_path = workspace_root / temp_output_file.name
            repomix_config.output.file_path = temp_output_path
            repomix_config.output.style = "xml"  # Wymagane do wewnętrznego parsowania danych. Styl końcowy jest określany w sekcji 'output' pliku konfiguracyjnego.
            if patterns:
                repomix_config.include = patterns
            if exclude_patterns:
                repomix_config.ignore.custom_patterns = exclude_patterns
            if config.get("gitignore_file"):
                repomix_config.ignore.use_gitignore = True
            repomix_config.output.show_line_numbers = repomix_opts.get("show_line_numbers", False)
            repomix_config.output.calculate_tokens = repomix_opts.get("calculate_tokens", True)
            repomix_config.output.show_file_stats = repomix_opts.get("show_file_stats", True)
            repomix_config.output.show_directory_structure = repomix_opts.get("show_directory_structure", True)
            repomix_config.output.top_files_length = repomix_opts.get("top_files_length", 2)
            repomix_config.output.copy_to_clipboard = repomix_opts.get("copy_to_clipboard", False)
            repomix_config.output.include_empty_directories = repomix_opts.get("include_empty_directories", False)
            repomix_config.output.remove_comments = repomix_opts.get("remove_comments", False)
            repomix_config.output.remove_empty_lines = repomix_opts.get("remove_empty_lines", False)
            compression_opts = repomix_opts.get("compression", {})
            repomix_config.compression.enabled = compression_opts.get("enabled", False)
            repomix_config.compression.keep_signatures = compression_opts.get("keep_signatures", True)
            repomix_config.compression.keep_docstrings = compression_opts.get("keep_docstrings", True)
            repomix_config.compression.keep_interfaces = compression_opts.get("keep_interfaces", True)
            repomix_config.security.enable_security_check = repomix_opts.get("security_check", True)
            processor = RepoProcessor(str(target_path), config=repomix_config)
            result = processor.process()
            if result:
                stats_info = f"\n=== Statystyki dla grupy '{group_name}' - ścieżka '{path_str}' ===\n"
                if hasattr(result, 'total_files'):
                    stats_info += f"Łączna liczba plików: {result.total_files}\n"
                if hasattr(result, 'total_chars'):
                    stats_info += f"Łączna liczba znaków: {result.total_chars}\n"
                if hasattr(result, 'total_tokens'):
                    stats_info += f"Łączna liczba tokenów: {result.total_tokens}\n"
                if hasattr(result, 'file_char_counts') and result.file_char_counts:
                    stats_info += f"\nTop {repomix_opts.get('top_files_length', 2)} plików wg liczby znaków:\n"
                    sorted_files = sorted(result.file_char_counts.items(), key=lambda x: x[1], reverse=True)
                    for i, (file_path, char_count) in enumerate(sorted_files[:repomix_opts.get('top_files_length', 2)]):
                        stats_info += f"  {i+1}. {file_path}: {char_count} znaków\n"
                if hasattr(result, 'file_token_counts') and result.file_token_counts:
                    stats_info += f"\nTop {repomix_opts.get('top_files_length', 2)} plików wg liczby tokenów:\n"
                    sorted_files = sorted(result.file_token_counts.items(), key=lambda x: x[1], reverse=True)
                    for i, (file_path, token_count) in enumerate(sorted_files[:repomix_opts.get('top_files_length', 2)]):
                        stats_info += f"  {i+1}. {file_path}: {token_count} tokenów\n"
                if hasattr(result, 'file_tree') and result.file_tree:
                    stats_info += f"\nStruktura katalogów:\n{result.file_tree}\n"
                if hasattr(result, 'suspicious_files_results') and result.suspicious_files_results:
                    stats_info += f"\nPodejrzane pliki: {len(result.suspicious_files_results)}\n"
                export_dir = workspace_root / ".doc-gen" / "export"
                export_dir.mkdir(exist_ok=True)
                stats_file = export_dir / "repomix-stats.log"
                with open(stats_file, "a", encoding="utf-8") as f:
                    f.write(stats_info)
                logging.info(f"  Statystyki zapisane do: {stats_file}")
            if os.path.exists(temp_output_path):
                with open(temp_output_path, "r", encoding="utf-8") as f:
                    output_content = f.read()
                os.remove(temp_output_path)
                if repomix_opts.get("style", "xml") == "xml":
                    try:
                        root = ET.fromstring(output_content)
                        for file_elem in root.findall(".//file"):
                            file_path_elem = file_elem.find("path")
                            content_elem = file_elem.find("content")
                            if file_path_elem is not None and content_elem is not None:
                                file_path_in_repomix = file_path_elem.text
                                content = content_elem.text or ""
                                if content:
                                    try:
                                        decoded_content = base64.b64decode(content).decode("utf-8")
                                        content = decoded_content
                                    except:
                                        pass
                                full_file_path = workspace_root / file_path_in_repomix
                                if full_file_path not in processed_files_set:
                                    processed_files_set.add(full_file_path)
                                    unique_files_in_group.append(full_file_path)
                                    group_files_content.append(
                                        {"path": file_path_in_repomix, "content": content}
                                    )
                                else:
                                    logging.info(f"  Plik '{file_path_in_repomix}' już przetworzony, pomijam.")
                    except ET.ParseError as e:
                        logging.error(f"  BŁĄD parsowania XML dla grupy '{group_name}': {e}")
                else:
                    logging.warning(f"  Format '{repomix_opts.get('style')}' nie jest w pełni obsługiwany w tej wersji")
            else:
                logging.warning(f"  Repomix nie utworzył pliku wyjściowego dla grupy '{group_name}'.")
        except Exception as e:
            logging.error(f"BŁĄD przetwarzania grupy '{group_name}' z Python Repomix: {e}")
            continue
    logging.info(f"  Znaleziono {len(unique_files_in_group)} unikalnych plików w grupie '{group_name}'.")
    return unique_files_in_group, group_files_content

def create_final_xml(all_groups_data, workspace_root, output_file, project_name, processed_files_set):
    """Tworzy finalny plik w formacie XML."""
    logging.info("Rozpoczynam tworzenie pliku w formacie XML...")
    root_elem = ET.Element("AggregatedCodebase")
    project_elem = ET.SubElement(root_elem, "Project", name=project_name)
    ET.SubElement(project_elem, "WorkspaceRoot").text = str(workspace_root)
    ET.SubElement(project_elem, "TotalUniqueFiles").text = str(len(processed_files_set))
    for i, (group, files_list, group_content_data) in enumerate(all_groups_data, 1):
        group_elem = ET.SubElement(project_elem, "Group", name=group.get("name", f"Group {i}"))
        if desc := group.get("description"):
            ET.SubElement(group_elem, "Description").text = desc
        ET.SubElement(group_elem, "FileCount").text = str(len(files_list))
        files_list_elem = ET.SubElement(group_elem, "FilesList")
        for file_path_obj in files_list:
            file_elem = ET.SubElement(files_list_elem, "File")
            ET.SubElement(file_elem, "Path").text = file_path_obj.relative_to(workspace_root).as_posix()
            ET.SubElement(file_elem, "Name").text = file_path_obj.name
        content_elem = ET.SubElement(group_elem, "Content")
        for file_data in group_content_data:
            file_content_elem = ET.SubElement(content_elem, "FileContent")
            ET.SubElement(file_content_elem, "Path").text = file_data["path"]
            ET.SubElement(file_content_elem, "Content").text = file_data["content"]
    rough_string = ET.tostring(root_elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    final_xml_string = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    output_path = workspace_root / output_file
    try:
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(final_xml_string)
        logging.info(f"\nGotowe! Plik '{output_path.name}' został utworzony w: {output_path}")
    except Exception as e:
        logging.error(f"BŁĄD ZAPISU PLIKU: {e}")

def create_final_markdown(all_groups_data, workspace_root, output_file, project_name, processed_files_set):
    """Tworzy finalny plik w formacie Markdown."""
    logging.info("Rozpoczynam tworzenie pliku w formacie Markdown...")
    markdown_lines = [
        f"# Projekt: {project_name}",
        f'## Katalog główny: `{workspace_root}`',
        f"## Łączna liczba unikalnych plików: {len(processed_files_set)}",
        "---",
    ]
    for i, (group, files_list, group_content_data) in enumerate(all_groups_data, 1):
        group_name = group.get("name", f"Grupa {i}")
        markdown_lines.append(f"## Grupa: {group_name}")
        if desc := group.get("description"):
            markdown_lines.append(f"**Opis:** {desc}")
        markdown_lines.append(f"**Liczba plików w grupie:** {len(files_list)}")
        markdown_lines.append("\n### Lista plików:")
        for file_path_obj in files_list:
            relative_path = file_path_obj.relative_to(workspace_root).as_posix()
            markdown_lines.append(f"- `{relative_path}`")
        markdown_lines.append("\n### Zawartość plików:")
        for file_data in group_content_data:
            markdown_lines.append(f'#### Plik: `{file_data["path"]}`')
            lang = Path(file_data["path"]).suffix.lstrip(".") or "text"
            markdown_lines.append(f"```{lang}")
            markdown_lines.append(file_data["content"])
            markdown_lines.append("```")
        markdown_lines.append("---")
    final_markdown_string = "\n".join(markdown_lines)
    output_path = workspace_root / output_file
    try:
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(final_markdown_string)
        logging.info(f"\nGotowe! Plik '{output_path.name}' został utworzony w: {output_path}")
    except Exception as e:
        logging.error(f"BŁĄD ZAPISU PLIKU: {e}")

def main():
    """Główna funkcja skryptu."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    workspace_root = get_workspace_root()
    if len(sys.argv) > 1:
        config_file_path = Path(sys.argv[1])
        if not config_file_path.is_absolute():
            config_file_path = workspace_root / config_file_path
    else:
        config_file_path = workspace_root / DEFAULT_CONFIG_FILE
    logging.info(f"Używam pliku konfiguracyjnego: {config_file_path}")
    config = load_config(config_file_path)
    if not config:
        return
    project_name = config.get("project_name", "Unknown Project")
    output_config = config.get("output", {})
    output_style = output_config.get("style", "xml").lower()
    output_filename_base = output_config.get("filename", "output")
    if output_style in ["md", "markdown"]:
        output_style = "markdown"
        output_file = f"{output_filename_base}.md"
    elif output_style == "xml":
        output_file = f"{output_filename_base}.xml"
    else:
        logging.error(f"Nieobsługiwany format wyjściowy: '{output_style}'. Dozwolone: xml, markdown, md.")
        return
    logging.info(f"\nRozpoczynam agregację dla projektu: {project_name}")
    logging.info(f"Plik wyjściowy: {output_file}")
    export_dir = workspace_root / ".doc-gen" / "export"
    export_dir.mkdir(exist_ok=True)
    stats_file = export_dir / "repomix-stats.log"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"=== Statystyki Repomix dla projektu: {project_name} ===\n")
        f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
    logging.info(f"Plik statystyk: {stats_file}")
    all_groups_data = []
    processed_files_set = set()
    for i, group in enumerate(config.get("groups", [])):
        files_in_group, group_content_data = process_group_with_repomix(
            group, workspace_root, processed_files_set, config
        )
        all_groups_data.append((group, files_in_group, group_content_data))
    if output_style == "xml":
        create_final_xml(all_groups_data, workspace_root, output_file, project_name, processed_files_set)
    elif output_style == "markdown":
        create_final_markdown(all_groups_data, workspace_root, output_file, project_name, processed_files_set)

if __name__ == "__main__":
    main()
```
#### Plik: `apps/collector/tests.py`
```py
from django.test import TestCase

# Create your tests here.
```
#### Plik: `apps/collector/admin.py`
```py
from django.contrib import admin

# Register your models here.
```
#### Plik: `apps/collector/migrations/__init__.py`
```py

```
#### Plik: `apps/collector/models.py`
```py
from django.db import models

# Create your models here.
```
#### Plik: `apps/collector/quick_context_collector.py`
```py
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import yaml
from pathlib import Path
import os
import json # Added for history
import fnmatch

CONFIG_FILE_NAME = "config/context_filters.yaml"
HISTORY_FILE_NAME = "config/context_history.json"
MAX_HISTORY_ITEMS = 5

class QuickContextCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quick Context Collector")
        # Adjust initial size to accommodate new elements
        self.root.geometry("500x350")

        self.selected_directory = tk.StringVar()
        self.selected_filter_name = tk.StringVar()
        self.selected_history_entry = tk.StringVar() # For the new history combobox
        self.selected_exclude_pattern = tk.StringVar(value="None") # For the new exclude combobox
        self.save_to_central_dir = tk.BooleanVar(value=True) # For the new Checkbutton
        self.last_output_path = None # To store the path of the last generated file
        self.filters = {}
        self.history = []
        self.script_dir = Path(__file__).parent
        self.workspace_root = self.script_dir.parent # Define workspace_root for central export path
        self.history_file_path = self.script_dir / HISTORY_FILE_NAME

        # --- UI Elements ---
        # Directory Selection
        tk.Label(root, text="Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dir_entry = tk.Entry(root, textvariable=self.selected_directory, width=50)
        self.dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = tk.Button(root, text="Browse...", command=self.browse_directory)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        # History Selection
        tk.Label(root, text="History:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.history_combobox = ttk.Combobox(root, textvariable=self.selected_history_entry, state="readonly", width=47)
        self.history_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.history_combobox.bind("<<ComboboxSelected>>", self.on_history_selected)

        # Filter Selection
        tk.Label(root, text="Filter:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.filter_combobox = ttk.Combobox(root, textvariable=self.selected_filter_name, state="readonly", width=47)
        self.filter_combobox.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Exclude Pattern Selection
        tk.Label(root, text="Exclude:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.exclude_combobox = ttk.Combobox(root, textvariable=self.selected_exclude_pattern, state="readonly", width=47)
        self.exclude_combobox['values'] = ["None", "test", "legacy", "test & legacy"]
        self.exclude_combobox.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Save Location Checkbutton
        self.save_to_central_dir_checkbutton = tk.Checkbutton(root, text="Save to central export directory (.doc-gen/export)", variable=self.save_to_central_dir)
        self.save_to_central_dir_checkbutton.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Action Buttons Frame
        action_buttons_frame = tk.Frame(root)
        action_buttons_frame.grid(row=5, column=1, padx=5, pady=10, sticky="ew")
        action_buttons_frame.grid_columnconfigure(0, weight=1)
        action_buttons_frame.grid_columnconfigure(1, weight=1)

        self.collect_button = tk.Button(action_buttons_frame, text="Collect Context", command=self.collect_context, height=2, width=15)
        self.collect_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.copy_button = tk.Button(action_buttons_frame, text="Copy Output", command=self.copy_output_to_clipboard, height=2, width=15)
        self.copy_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Status Bar (optional)
        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Configure grid column weights for resizing
        root.grid_columnconfigure(1, weight=1)

        # Initial Load
        self.load_filters()
        self.load_history()

    def load_filters(self):
        try:
            config_path = self.script_dir / CONFIG_FILE_NAME
            with open(config_path, "r", encoding="utf-8") as f:
                self.filters = yaml.safe_load(f).get("filters", {})
                self.filter_combobox['values'] = list(self.filters.keys())
                self.update_status(f"Loaded {len(self.filters)} filters from {CONFIG_FILE_NAME}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load or parse {CONFIG_FILE_NAME}:\n{e}")
            self.root.quit()

    def browse_directory(self):
        # Start browsing from the workspace root directory
        initial_dir = self.workspace_root
        directory = filedialog.askdirectory(initialdir=initial_dir, title="Select a Directory")
        if directory:
            self.selected_directory.set(directory)

    def collect_context(self):
        directory_str = self.selected_directory.get()
        filter_name = self.selected_filter_name.get()

        if not directory_str or not filter_name:
            self.update_status("Please select a directory and a filter first.")
            return

        target_dir = Path(directory_str)
        if not target_dir.is_dir():
            self.update_status(f"Error: Directory not found at {target_dir}")
            return

        selected_filter_patterns = self.filters[filter_name].get("patterns", ["*.*"])
        base_output_filename = f"{target_dir.name}_context__{filter_name.replace(' (*.*)','').replace('*','all').replace('.','')}.txt"

        if self.save_to_central_dir.get():
            central_export_dir = self.workspace_root / ".doc-gen" / "export"
            central_export_dir.mkdir(parents=True, exist_ok=True)
            output_path = central_export_dir / base_output_filename
        else:
            output_path = target_dir / base_output_filename

        self.save_history(directory_str, filter_name)

        try:
            self.update_status(f"Collecting context... Filter: {filter_name}")

            exclude_option = self.selected_exclude_pattern.get()
            exclude_dirs = {'__pycache__', '.git', '.svn', 'node_modules', '.venv', 'venv'}
            if exclude_option == "test":
                exclude_dirs.update(['test', 'tests'])
            elif exclude_option == "legacy":
                exclude_dirs.add('legacy')
            elif exclude_option == "test & legacy":
                exclude_dirs.update(['test', 'tests', 'legacy'])

            found_files = []
            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if d.lower() not in exclude_dirs]
                
                for filename in files:
                    for pattern in selected_filter_patterns:
                        if fnmatch.fnmatch(filename, pattern):
                            found_files.append(Path(root) / filename)
                            break
            
            found_files = sorted(list(set(found_files)))

            if not found_files:
                self.update_status(f"No files found matching the filter in {target_dir.name}")
                return

            all_content = ""
            for file_path in found_files:
                try:
                    relative_path = file_path.relative_to(self.workspace_root)
                    all_content += f"--- START {relative_path} ---\n"
                    all_content += file_path.read_text(encoding='utf-8', errors='ignore')
                    all_content += f"\n--- END {relative_path} ---\n\n"
                except Exception as e:
                    all_content += f"--- ERROR reading {file_path}: {e} ---\n\n"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(all_content)
            self.last_output_path = output_path
            self.update_status(f"Successfully collected {len(found_files)} files to {output_path.name}")
            print(f"Successfully collected {len(found_files)} files to {output_path}")
        except Exception as e:
            self.last_output_path = None
            self.update_status(f"Error during collection: {e}")
            print(f"Error during collection: {e}")

    def format_history_entry_display(self, entry):
        dir_path = Path(entry.get("directory", "N/A"))
        filter_name = entry.get("filter_name", "N/A")
        return f"{dir_path.name}  |  {filter_name}"

    def update_history_combobox(self):
        display_entries = [self.format_history_entry_display(entry) for entry in self.history]
        self.history_combobox['values'] = display_entries

    def on_history_selected(self, event):
        selected_display_text = self.selected_history_entry.get()
        for entry in self.history:
            if self.format_history_entry_display(entry) == selected_display_text:
                if Path(entry.get("directory", "")).is_dir() and \
                   entry.get("filter_name", "") in self.filters:
                    self.selected_directory.set(entry["directory"])
                    self.selected_filter_name.set(entry["filter_name"])
                    self.update_status(f"Selected from history: {Path(entry['directory']).name} | {entry['filter_name']}")
                else:
                    self.update_status(f"Invalid history entry selected: {selected_display_text}")
                return

    def load_history(self):
        try:
            if self.history_file_path.exists():
                with open(self.history_file_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                    if not isinstance(self.history, list):
                        self.history = []
            else:
                 self.history = []
        except Exception as e:
            self.history = []
            self.update_status(f"Error loading history file: {e}. Starting with empty history.")
            print(f"Error loading history file: {e}")

        self.update_history_combobox()

        if self.history:
            most_recent = self.history[0]
            if Path(most_recent.get("directory", "")).is_dir() and \
               most_recent.get("filter_name", "") in self.filters:
                self.selected_directory.set(most_recent["directory"])
                self.selected_filter_name.set(most_recent["filter_name"])
                self.update_status(f"Loaded last used: {Path(most_recent['directory']).name} | {most_recent['filter_name']}")
                return
        
        self.update_status("No valid recent settings or history file not found. Using defaults.")

    def save_history(self, directory_str, filter_name_str):
        if not directory_str or not filter_name_str:
            return

        new_entry = {"directory": directory_str, "filter_name": filter_name_str}
        
        self.history = [entry for entry in self.history if not (entry.get("directory") == directory_str and entry.get("filter_name") == filter_name_str)]
        
        self.history.insert(0, new_entry)
        self.history = self.history[:MAX_HISTORY_ITEMS]

        try:
            with open(self.history_file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            self.update_history_combobox()
        except Exception as e:
            self.update_status(f"Error saving history: {e}")
            print(f"Error saving history: {e}")

    def copy_output_to_clipboard(self):
        if self.last_output_path and self.last_output_path.exists():
            try:
                with open(self.last_output_path, "r", encoding="utf-8") as f:
                    content_to_copy = f.read()
                
                self.root.clipboard_clear()
                self.root.clipboard_append(content_to_copy)
                self.update_status(f"Copied content of {self.last_output_path.name} to clipboard.")
                print(f"Copied content of {self.last_output_path.name} to clipboard.")
            except Exception as e:
                self.update_status(f"Error copying to clipboard: {e}")
                print(f"Error copying to clipboard: {e}")
        else:
            self.update_status("No output file generated yet or file not found.")
            print("No output file generated yet or file not found for copying.")

    def update_status(self, message):
        self.status_label.config(text=message)
        print(message)

if __name__ == "__main__":
    root = tk.Tk()
    app = QuickContextCollectorApp(root)
    root.mainloop()
```
#### Plik: `apps/collector/management/commands/run_collector.py`
```py

```
#### Plik: `apps/collector/apps.py`
```py
from django.apps import AppConfig


class CollectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'collector'
```
#### Plik: `config/__init__.py`
```py

```
#### Plik: `config/urls.py`
```py
"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
```
#### Plik: `config/wsgi.py`
```py
"""
WSGI config for config project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = get_wsgi_application()
```
#### Plik: `config/asgi.py`
```py
"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = get_asgi_application()
```
#### Plik: `testing-testing/collector/config-selector.py`
```py
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
    """Uruchamia skrypt advanced_context_collector.py z wybraną konfiguracją."""
    script_dir = Path(__file__).parent
    main_script = script_dir / 'advanced_context_collector.py'
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
```
#### Plik: `testing-testing/collector/views.py`
```py
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from .models import CollectorModel
from django.contrib.auth.decorators import login_required

# Enhanced views with authentication
@login_required
def collector_list(request):
    collectors = CollectorModel.objects.filter(owner=request.user)
    return render(request, 'collector/list.html', {'collectors': collectors})

@login_required
def collector_detail(request, pk):
    collector = get_object_or_404(CollectorModel, pk=pk, owner=request.user)
    return render(request, 'collector/detail.html', {'collector': collector})

def collector_api(request):
    collectors = list(CollectorModel.objects.values())
    return JsonResponse({'collectors': collectors, 'count': len(collectors)})
from django.http import JsonResponse
from .models import CollectorModel

# Create your views here.
def collector_list(request):
    collectors = CollectorModel.objects.all()
    return render(request, 'collector/list.html', {'collectors': collectors})

def collector_api(request):
    collectors = list(CollectorModel.objects.values())
    return JsonResponse({'collectors': collectors})
```
#### Plik: `testing-testing/collector/__init__.py`
```py

```
#### Plik: `testing-testing/collector/advanced_context_collector.py`
```py
import os
import yaml
from pathlib import Path
import fnmatch
import re
import logging
import tempfile
import base64
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time
from repomix import RepoProcessor, RepomixConfig

# =================================================================================
# SCRIPT FOR FILE AGGREGATION WITH GROUPS AND EXCLUDE PATTERNS (REPOMIX INTEGRATION)
#
# Wersja: 9.0 (z obsługą formatu XML i Markdown)
# Opis: Skrypt wykorzystuje Repomix do przetwarzania plików, a następnie
#       generuje plik wyjściowy w formacie XML lub Markdown na podstawie
#       konfiguracji YAML.
# =================================================================================

DEFAULT_CONFIG_FILE = ".doc-gen/config-lists/.comb-scripts-config01.yaml"

def get_default_repomix_options():
    """Zwraca domyślne opcje konfiguracyjne dla Repomix."""
    return {
        "style": "xml",
        "remove_comments": False,
        "remove_empty_lines": False,
        "show_line_numbers": False,
        "calculate_tokens": True,
        "show_file_stats": True,
        "show_directory_structure": True,
        "top_files_length": 2,
        "copy_to_clipboard": False,
        "include_empty_directories": False,
        "compression": {
            "enabled": False,
            "keep_signatures": True,
            "keep_docstrings": True,
            "keep_interfaces": True,
        },
        "security_check": True,
    }

def get_workspace_root():
    """Zwraca ścieżkę do workspace root."""
    return Path(__file__).parent.parent

def load_config(config_file_path):
    """Wczytuje konfigurację z pliku YAML."""
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"BŁĄD Wczytywania konfiguracji: {e}")
        return None

def process_group_with_repomix(group, workspace_root, processed_files_set, config):
    """
    Przetwarza grupę plików używając Python Repomix, z uwzględnieniem deduplikacji.
    Zwraca listę ścieżek do unikalnych plików oraz ich zawartość.
    """
    group_name = group.get("name", "Unnamed Group")
    patterns = group.get("patterns", [])
    exclude_patterns = group.get("exclude_patterns", [])
    paths = group.get("paths", [])
    logging.info(f"\nPrzetwarzanie grupy: {group_name}")
    group_files_content = []
    unique_files_in_group = []
    repomix_opts = get_default_repomix_options()
    if "repomix_global_options" in config:
        repomix_opts.update(config["repomix_global_options"])
    if "repomix_options" in group:
        group_opts = group["repomix_options"]
        if "compression" in group_opts:
            repomix_opts["compression"].update(group_opts["compression"])
            group_opts = group_opts.copy()
            del group_opts["compression"]
        repomix_opts.update(group_opts)
    for path_str in paths:
        target_path = (
            workspace_root / path_str
            if path_str not in ["all", ".", "**/*", "**"]
            else workspace_root
        )
        if not target_path.exists():
            logging.warning(f"  UWAGA: Ścieżka '{path_str}' nie istnieje i została pominięta.")
            continue
        try:
            repomix_config = RepomixConfig()
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".xml", encoding="utf-8"
            ) as temp_output_file:
                temp_output_path = workspace_root / temp_output_file.name
            repomix_config.output.file_path = temp_output_path
            repomix_config.output.style = "xml"  # Wymagane do wewnętrznego parsowania danych. Styl końcowy jest określany w sekcji 'output' pliku konfiguracyjnego.
            if patterns:
                repomix_config.include = patterns
            if exclude_patterns:
                repomix_config.ignore.custom_patterns = exclude_patterns
            if config.get("gitignore_file"):
                repomix_config.ignore.use_gitignore = True
            repomix_config.output.show_line_numbers = repomix_opts.get("show_line_numbers", False)
            repomix_config.output.calculate_tokens = repomix_opts.get("calculate_tokens", True)
            repomix_config.output.show_file_stats = repomix_opts.get("show_file_stats", True)
            repomix_config.output.show_directory_structure = repomix_opts.get("show_directory_structure", True)
            repomix_config.output.top_files_length = repomix_opts.get("top_files_length", 2)
            repomix_config.output.copy_to_clipboard = repomix_opts.get("copy_to_clipboard", False)
            repomix_config.output.include_empty_directories = repomix_opts.get("include_empty_directories", False)
            repomix_config.output.remove_comments = repomix_opts.get("remove_comments", False)
            repomix_config.output.remove_empty_lines = repomix_opts.get("remove_empty_lines", False)
            compression_opts = repomix_opts.get("compression", {})
            repomix_config.compression.enabled = compression_opts.get("enabled", False)
            repomix_config.compression.keep_signatures = compression_opts.get("keep_signatures", True)
            repomix_config.compression.keep_docstrings = compression_opts.get("keep_docstrings", True)
            repomix_config.compression.keep_interfaces = compression_opts.get("keep_interfaces", True)
            repomix_config.security.enable_security_check = repomix_opts.get("security_check", True)
            processor = RepoProcessor(str(target_path), config=repomix_config)
            result = processor.process()
            if result:
                stats_info = f"\n=== Statystyki dla grupy '{group_name}' - ścieżka '{path_str}' ===\n"
                if hasattr(result, 'total_files'):
                    stats_info += f"Łączna liczba plików: {result.total_files}\n"
                if hasattr(result, 'total_chars'):
                    stats_info += f"Łączna liczba znaków: {result.total_chars}\n"
                if hasattr(result, 'total_tokens'):
                    stats_info += f"Łączna liczba tokenów: {result.total_tokens}\n"
                if hasattr(result, 'file_char_counts') and result.file_char_counts:
                    stats_info += f"\nTop {repomix_opts.get('top_files_length', 2)} plików wg liczby znaków:\n"
                    sorted_files = sorted(result.file_char_counts.items(), key=lambda x: x[1], reverse=True)
                    for i, (file_path, char_count) in enumerate(sorted_files[:repomix_opts.get('top_files_length', 2)]):
                        stats_info += f"  {i+1}. {file_path}: {char_count} znaków\n"
                if hasattr(result, 'file_token_counts') and result.file_token_counts:
                    stats_info += f"\nTop {repomix_opts.get('top_files_length', 2)} plików wg liczby tokenów:\n"
                    sorted_files = sorted(result.file_token_counts.items(), key=lambda x: x[1], reverse=True)
                    for i, (file_path, token_count) in enumerate(sorted_files[:repomix_opts.get('top_files_length', 2)]):
                        stats_info += f"  {i+1}. {file_path}: {token_count} tokenów\n"
                if hasattr(result, 'file_tree') and result.file_tree:
                    stats_info += f"\nStruktura katalogów:\n{result.file_tree}\n"
                if hasattr(result, 'suspicious_files_results') and result.suspicious_files_results:
                    stats_info += f"\nPodejrzane pliki: {len(result.suspicious_files_results)}\n"
                export_dir = workspace_root / ".doc-gen" / "export"
                export_dir.mkdir(exist_ok=True)
                stats_file = export_dir / "repomix-stats.log"
                with open(stats_file, "a", encoding="utf-8") as f:
                    f.write(stats_info)
                logging.info(f"  Statystyki zapisane do: {stats_file}")
            if os.path.exists(temp_output_path):
                with open(temp_output_path, "r", encoding="utf-8") as f:
                    output_content = f.read()
                os.remove(temp_output_path)
                if repomix_opts.get("style", "xml") == "xml":
                    try:
                        root = ET.fromstring(output_content)
                        for file_elem in root.findall(".//file"):
                            file_path_elem = file_elem.find("path")
                            content_elem = file_elem.find("content")
                            if file_path_elem is not None and content_elem is not None:
                                file_path_in_repomix = file_path_elem.text
                                content = content_elem.text or ""
                                if content:
                                    try:
                                        decoded_content = base64.b64decode(content).decode("utf-8")
                                        content = decoded_content
                                    except:
                                        pass
                                full_file_path = workspace_root / file_path_in_repomix
                                if full_file_path not in processed_files_set:
                                    processed_files_set.add(full_file_path)
                                    unique_files_in_group.append(full_file_path)
                                    group_files_content.append(
                                        {"path": file_path_in_repomix, "content": content}
                                    )
                                else:
                                    logging.info(f"  Plik '{file_path_in_repomix}' już przetworzony, pomijam.")
                    except ET.ParseError as e:
                        logging.error(f"  BŁĄD parsowania XML dla grupy '{group_name}': {e}")
                else:
                    logging.warning(f"  Format '{repomix_opts.get('style')}' nie jest w pełni obsługiwany w tej wersji")
            else:
                logging.warning(f"  Repomix nie utworzył pliku wyjściowego dla grupy '{group_name}'.")
        except Exception as e:
            logging.error(f"BŁĄD przetwarzania grupy '{group_name}' z Python Repomix: {e}")
            continue
    logging.info(f"  Znaleziono {len(unique_files_in_group)} unikalnych plików w grupie '{group_name}'.")
    return unique_files_in_group, group_files_content

def create_final_xml(all_groups_data, workspace_root, output_file, project_name, processed_files_set):
    """Tworzy finalny plik w formacie XML."""
    logging.info("Rozpoczynam tworzenie pliku w formacie XML...")
    root_elem = ET.Element("AggregatedCodebase")
    project_elem = ET.SubElement(root_elem, "Project", name=project_name)
    ET.SubElement(project_elem, "WorkspaceRoot").text = str(workspace_root)
    ET.SubElement(project_elem, "TotalUniqueFiles").text = str(len(processed_files_set))
    for i, (group, files_list, group_content_data) in enumerate(all_groups_data, 1):
        group_elem = ET.SubElement(project_elem, "Group", name=group.get("name", f"Group {i}"))
        if desc := group.get("description"):
            ET.SubElement(group_elem, "Description").text = desc
        ET.SubElement(group_elem, "FileCount").text = str(len(files_list))
        files_list_elem = ET.SubElement(group_elem, "FilesList")
        for file_path_obj in files_list:
            file_elem = ET.SubElement(files_list_elem, "File")
            ET.SubElement(file_elem, "Path").text = file_path_obj.relative_to(workspace_root).as_posix()
            ET.SubElement(file_elem, "Name").text = file_path_obj.name
        content_elem = ET.SubElement(group_elem, "Content")
        for file_data in group_content_data:
            file_content_elem = ET.SubElement(content_elem, "FileContent")
            ET.SubElement(file_content_elem, "Path").text = file_data["path"]
            ET.SubElement(file_content_elem, "Content").text = file_data["content"]
    rough_string = ET.tostring(root_elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    final_xml_string = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    output_path = workspace_root / output_file
    try:
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(final_xml_string)
        logging.info(f"\nGotowe! Plik '{output_path.name}' został utworzony w: {output_path}")
    except Exception as e:
        logging.error(f"BŁĄD ZAPISU PLIKU: {e}")

def create_final_markdown(all_groups_data, workspace_root, output_file, project_name, processed_files_set):
    """Tworzy finalny plik w formacie Markdown."""
    logging.info("Rozpoczynam tworzenie pliku w formacie Markdown...")
    markdown_lines = [
        f"# Projekt: {project_name}",
        f'## Katalog główny: `{workspace_root}`',
        f"## Łączna liczba unikalnych plików: {len(processed_files_set)}",
        "---",
    ]
    for i, (group, files_list, group_content_data) in enumerate(all_groups_data, 1):
        group_name = group.get("name", f"Grupa {i}")
        markdown_lines.append(f"## Grupa: {group_name}")
        if desc := group.get("description"):
            markdown_lines.append(f"**Opis:** {desc}")
        markdown_lines.append(f"**Liczba plików w grupie:** {len(files_list)}")
        markdown_lines.append("\n### Lista plików:")
        for file_path_obj in files_list:
            relative_path = file_path_obj.relative_to(workspace_root).as_posix()
            markdown_lines.append(f"- `{relative_path}`")
        markdown_lines.append("\n### Zawartość plików:")
        for file_data in group_content_data:
            markdown_lines.append(f'#### Plik: `{file_data["path"]}`')
            lang = Path(file_data["path"]).suffix.lstrip(".") or "text"
            markdown_lines.append(f"```{lang}")
            markdown_lines.append(file_data["content"])
            markdown_lines.append("```")
        markdown_lines.append("---")
    final_markdown_string = "\n".join(markdown_lines)
    output_path = workspace_root / output_file
    try:
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(final_markdown_string)
        logging.info(f"\nGotowe! Plik '{output_path.name}' został utworzony w: {output_path}")
    except Exception as e:
        logging.error(f"BŁĄD ZAPISU PLIKU: {e}")

def main():
    """Główna funkcja skryptu."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    workspace_root = get_workspace_root()
    if len(sys.argv) > 1:
        config_file_path = Path(sys.argv[1])
        if not config_file_path.is_absolute():
            config_file_path = workspace_root / config_file_path
    else:
        config_file_path = workspace_root / DEFAULT_CONFIG_FILE
    logging.info(f"Używam pliku konfiguracyjnego: {config_file_path}")
    config = load_config(config_file_path)
    if not config:
        return
    project_name = config.get("project_name", "Unknown Project")
    output_config = config.get("output", {})
    output_style = output_config.get("style", "xml").lower()
    output_filename_base = output_config.get("filename", "output")
    if output_style in ["md", "markdown"]:
        output_style = "markdown"
        output_file = f"{output_filename_base}.md"
    elif output_style == "xml":
        output_file = f"{output_filename_base}.xml"
    else:
        logging.error(f"Nieobsługiwany format wyjściowy: '{output_style}'. Dozwolone: xml, markdown, md.")
        return
    logging.info(f"\nRozpoczynam agregację dla projektu: {project_name}")
    logging.info(f"Plik wyjściowy: {output_file}")
    export_dir = workspace_root / ".doc-gen" / "export"
    export_dir.mkdir(exist_ok=True)
    stats_file = export_dir / "repomix-stats.log"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"=== Statystyki Repomix dla projektu: {project_name} ===\n")
        f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
    logging.info(f"Plik statystyk: {stats_file}")
    all_groups_data = []
    processed_files_set = set()
    for i, group in enumerate(config.get("groups", [])):
        files_in_group, group_content_data = process_group_with_repomix(
            group, workspace_root, processed_files_set, config
        )
        all_groups_data.append((group, files_in_group, group_content_data))
    if output_style == "xml":
        create_final_xml(all_groups_data, workspace_root, output_file, project_name, processed_files_set)
    elif output_style == "markdown":
        create_final_markdown(all_groups_data, workspace_root, output_file, project_name, processed_files_set)

if __name__ == "__main__":
    main()
```
#### Plik: `testing-testing/collector/tests.py`
```py
from django.test import TestCase

# Create your tests here.
```
#### Plik: `testing-testing/collector/admin.py`
```py
from django.contrib import admin
from .models import CollectorModel, AnotherModel, ThirdModel

# Register all models here
admin.site.register(CollectorModel)
admin.site.register(AnotherModel)
admin.site.register(ThirdModel)

# Enhanced admin configuration
class CollectorModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at', 'status', 'priority']
    search_fields = ['name', 'description']
    list_filter = ['status', 'created_at']
from .models import CollectorModel, AnotherModel

# Register your models here.
admin.site.register(CollectorModel)
admin.site.register(AnotherModel)

# Additional admin configuration
class CollectorModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at', 'status']
    search_fields = ['name']
```
#### Plik: `testing-testing/collector/migrations/__init__.py`
```py

```
#### Plik: `testing-testing/collector/models.py`
```py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Enhanced models with better functionality
class CollectorModel(models.Model):
    name = models.CharField(max_length=100, help_text='Name of the collector')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.BooleanField(default=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='collectors')
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Collector'
        verbose_name_plural = 'Collectors'
    
    def __str__(self):
        return f'{self.name} ({self.owner.username})'

class AnotherModel(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    priority = models.IntegerField(default=1)

# Create your models here.
class CollectorModel(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.BooleanField(default=True)
    
    def __str__(self):
        return self.name

class AnotherModel(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
```
#### Plik: `testing-testing/collector/quick_context_collector.py`
```py
import tkinter as tk
from tkinter import filedialog, ttk
import yaml
from pathlib import Path
import os
import json # Added for history
import fnmatch

CONFIG_FILE_NAME = "config/context_filters.yaml"
HISTORY_FILE_NAME = "config/context_history.json"
MAX_HISTORY_ITEMS = 5

class QuickContextCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quick Context Collector")
        # Adjust initial size to accommodate new elements
        self.root.geometry("500x350")

        self.selected_directory = tk.StringVar()
        self.selected_filter_name = tk.StringVar()
        self.selected_history_entry = tk.StringVar() # For the new history combobox
        self.selected_exclude_pattern = tk.StringVar(value="None") # For the new exclude combobox
        self.save_to_central_dir = tk.BooleanVar(value=True) # For the new Checkbutton
        self.last_output_path = None # To store the path of the last generated file
        self.filters = {}
        self.history = []
        self.script_dir = Path(__file__).parent
        self.workspace_root = self.script_dir.parent # Define workspace_root for central export path
        self.history_file_path = self.script_dir / HISTORY_FILE_NAME

        # --- UI Elements ---
        # Directory Selection
        tk.Label(root, text="Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dir_entry = tk.Entry(root, textvariable=self.selected_directory, width=50)
        self.dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = tk.Button(root, text="Browse...", command=self.browse_directory)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        # History Selection
        tk.Label(root, text="History:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.history_combobox = ttk.Combobox(root, textvariable=self.selected_history_entry, state="readonly", width=47)
        self.history_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.history_combobox.bind("<<ComboboxSelected>>", self.on_history_selected)

        # Filter Selection
        tk.Label(root, text="Filter:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.filter_combobox = ttk.Combobox(root, textvariable=self.selected_filter_name, state="readonly", width=47)
        self.filter_combobox.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Exclude Pattern Selection
        tk.Label(root, text="Exclude:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.exclude_combobox = ttk.Combobox(root, textvariable=self.selected_exclude_pattern, state="readonly", width=47)
        self.exclude_combobox['values'] = ["None", "test", "legacy", "test & legacy"]
        self.exclude_combobox.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Save Location Checkbutton
        self.save_to_central_dir_checkbutton = tk.Checkbutton(root, text="Save to central export directory (.doc-gen/export)", variable=self.save_to_central_dir)
        self.save_to_central_dir_checkbutton.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Action Buttons Frame
        action_buttons_frame = tk.Frame(root)
        action_buttons_frame.grid(row=5, column=1, padx=5, pady=10, sticky="ew")
        action_buttons_frame.grid_columnconfigure(0, weight=1)
        action_buttons_frame.grid_columnconfigure(1, weight=1)

        self.collect_button = tk.Button(action_buttons_frame, text="Collect Context", command=self.collect_context, height=2, width=15)
        self.collect_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.copy_button = tk.Button(action_buttons_frame, text="Copy Output", command=self.copy_output_to_clipboard, height=2, width=15)
        self.copy_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Status Bar (optional)
        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Configure grid column weights for resizing
        root.grid_columnconfigure(1, weight=1)

        # Initial Load
        self.load_filters()
        self.load_history()

    def load_filters(self):
        try:
            config_path = self.script_dir / CONFIG_FILE_NAME
            with open(config_path, "r", encoding="utf-8") as f:
                self.filters = yaml.safe_load(f).get("filters", {})
                self.filter_combobox['values'] = list(self.filters.keys())
                self.update_status(f"Loaded {len(self.filters)} filters from {CONFIG_FILE_NAME}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load or parse {CONFIG_FILE_NAME}:\n{e}")
            self.root.quit()

    def browse_directory(self):
        # Start browsing from the workspace root directory
        initial_dir = self.workspace_root
        directory = filedialog.askdirectory(initialdir=initial_dir, title="Select a Directory")
        if directory:
            self.selected_directory.set(directory)

    def collect_context(self):
        directory_str = self.selected_directory.get()
        filter_name = self.selected_filter_name.get()

        if not directory_str or not filter_name:
            self.update_status("Please select a directory and a filter first.")
            return

        target_dir = Path(directory_str)
        if not target_dir.is_dir():
            self.update_status(f"Error: Directory not found at {target_dir}")
            return

        selected_filter_patterns = self.filters[filter_name].get("patterns", ["*.*"])
        base_output_filename = f"{target_dir.name}_context__{filter_name.replace(' (*.*)','').replace('*','all').replace('.','')}.txt"

        if self.save_to_central_dir.get():
            central_export_dir = self.workspace_root / ".doc-gen" / "export"
            central_export_dir.mkdir(parents=True, exist_ok=True)
            output_path = central_export_dir / base_output_filename
        else:
            output_path = target_dir / base_output_filename

        self.save_history(directory_str, filter_name)

        try:
            self.update_status(f"Collecting context... Filter: {filter_name}")

            exclude_option = self.selected_exclude_pattern.get()
            exclude_dirs = {'__pycache__', '.git', '.svn', 'node_modules', '.venv', 'venv'}
            if exclude_option == "test":
                exclude_dirs.update(['test', 'tests'])
            elif exclude_option == "legacy":
                exclude_dirs.add('legacy')
            elif exclude_option == "test & legacy":
                exclude_dirs.update(['test', 'tests', 'legacy'])

            found_files = []
            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if d.lower() not in exclude_dirs]
                
                for filename in files:
                    for pattern in selected_filter_patterns:
                        if fnmatch.fnmatch(filename, pattern):
                            found_files.append(Path(root) / filename)
                            break
            
            found_files = sorted(list(set(found_files)))

            if not found_files:
                self.update_status(f"No files found matching the filter in {target_dir.name}")
                return

            all_content = ""
            for file_path in found_files:
                try:
                    relative_path = file_path.relative_to(self.workspace_root)
                    all_content += f"--- START {relative_path} ---\n"
                    all_content += file_path.read_text(encoding='utf-8', errors='ignore')
                    all_content += f"\n--- END {relative_path} ---\n\n"
                except Exception as e:
                    all_content += f"--- ERROR reading {file_path}: {e} ---\n\n"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(all_content)
            self.last_output_path = output_path
            self.update_status(f"Successfully collected {len(found_files)} files to {output_path.name}")
            print(f"Successfully collected {len(found_files)} files to {output_path}")
        except Exception as e:
            self.last_output_path = None
            self.update_status(f"Error during collection: {e}")
            print(f"Error during collection: {e}")

    def format_history_entry_display(self, entry):
        dir_path = Path(entry.get("directory", "N/A"))
        filter_name = entry.get("filter_name", "N/A")
        return f"{dir_path.name}  |  {filter_name}"

    def update_history_combobox(self):
        display_entries = [self.format_history_entry_display(entry) for entry in self.history]
        self.history_combobox['values'] = display_entries

    def on_history_selected(self, event):
        selected_display_text = self.selected_history_entry.get()
        for entry in self.history:
            if self.format_history_entry_display(entry) == selected_display_text:
                if Path(entry.get("directory", "")).is_dir() and \
                   entry.get("filter_name", "") in self.filters:
                    self.selected_directory.set(entry["directory"])
                    self.selected_filter_name.set(entry["filter_name"])
                    self.update_status(f"Selected from history: {Path(entry['directory']).name} | {entry['filter_name']}")
                else:
                    self.update_status(f"Invalid history entry selected: {selected_display_text}")
                return

    def load_history(self):
        try:
            if self.history_file_path.exists():
                with open(self.history_file_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                    if not isinstance(self.history, list):
                        self.history = []
            else:
                 self.history = []
        except Exception as e:
            self.history = []
            self.update_status(f"Error loading history file: {e}. Starting with empty history.")
            print(f"Error loading history file: {e}")

        self.update_history_combobox()

        if self.history:
            most_recent = self.history[0]
            if Path(most_recent.get("directory", "")).is_dir() and \
               most_recent.get("filter_name", "") in self.filters:
                self.selected_directory.set(most_recent["directory"])
                self.selected_filter_name.set(most_recent["filter_name"])
                self.update_status(f"Loaded last used: {Path(most_recent['directory']).name} | {most_recent['filter_name']}")
                return
        
        self.update_status("No valid recent settings or history file not found. Using defaults.")

    def save_history(self, directory_str, filter_name_str):
        if not directory_str or not filter_name_str:
            return

        new_entry = {"directory": directory_str, "filter_name": filter_name_str}
        
        self.history = [entry for entry in self.history if not (entry.get("directory") == directory_str and entry.get("filter_name") == filter_name_str)]
        
        self.history.insert(0, new_entry)
        self.history = self.history[:MAX_HISTORY_ITEMS]

        try:
            with open(self.history_file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            self.update_history_combobox()
        except Exception as e:
            self.update_status(f"Error saving history: {e}")
            print(f"Error saving history: {e}")

    def copy_output_to_clipboard(self):
        if self.last_output_path and self.last_output_path.exists():
            try:
                with open(self.last_output_path, "r", encoding="utf-8") as f:
                    content_to_copy = f.read()
                
                self.root.clipboard_clear()
                self.root.clipboard_append(content_to_copy)
                self.update_status(f"Copied content of {self.last_output_path.name} to clipboard.")
                print(f"Copied content of {self.last_output_path.name} to clipboard.")
            except Exception as e:
                self.update_status(f"Error copying to clipboard: {e}")
                print(f"Error copying to clipboard: {e}")
        else:
            self.update_status("No output file generated yet or file not found.")
            print("No output file generated yet or file not found for copying.")

    def update_status(self, message):
        self.status_label.config(text=message)
        print(message)

if __name__ == "__main__":
    root = tk.Tk()
    app = QuickContextCollectorApp(root)
    root.mainloop()
```
#### Plik: `testing-testing/collector/management/commands/run_collector.py`
```py
from django.core.management.base import BaseCommand
from collector.models import CollectorModel

class Command(BaseCommand):
    help = 'Run collector operations'
    
    def handle(self, *args, **options):
        self.stdout.write('Starting collector...')
        collectors = CollectorModel.objects.all()
        self.stdout.write(f'Found {collectors.count()} collectors')
        self.stdout.write('Collector finished successfully')
```
#### Plik: `testing-testing/collector/apps.py`
```py
from django.apps import AppConfig


class CollectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'collector'
```
---
## Grupa: Documentation Files
**Opis:** Pliki z dokumentacją w formacie Markdown i tekstowym.
**Liczba plików w grupie:** 1

### Lista plików:
- `.doc-gen/README.md`

### Zawartość plików:
#### Plik: `.doc-gen/README.md`
```md
# Skrypty Agregacji Plików - Dokumentacja

## Przegląd

Katalog `.doc-gen` zawiera skrypty do automatycznej agregacji plików projektowych w grupy tematyczne. Skrypty generują zbiorczy plik Markdown z zawartością wszystkich plików podzielonych na logiczne grupy.

## Pliki w katalogu

### Skrypty

- **`.comb-scripts-v2.py`** - Wersja 4.0 (stara, uniwersalna)
- **`.comb-scripts-v3.py`** - Wersja 5.0 (nowa, z grupami i YAML)

### Konfiguracja

- **`.comb-scripts-config01.yaml`** - Konfiguracja grup plików dla v3

### Dokumentacja

- **`README.md`** - Ten plik

## Jak używać nowej wersji (v3)

### 1. Wymagania

```bash
pip install PyYAML
```

### 2. Uruchomienie

```bash
# Z katalogu głównego projektu
python .doc-gen\.comb-scripts-v3.py
```

### 3. Wynik

Skrypt utworzy plik `.comb-scripts.md` w katalogu głównym projektu (workspace root).

## Konfiguracja grup (YAML)

Plik `.comb-scripts-config01.yaml` definiuje grupy plików:

```yaml
groups:
  - name: "Nazwa Grupy"
    description: "Opis grupy"
    patterns:
      - "*.py"      # wzorce plików
      - "*.jsx"
    paths:
      - "app/scripts"  # ścieżki do przeszukania
      - "all"          # specjalna wartość = cały workspace
    recursive: true     # czy szukać w podkatalogach
```

### Dostępne grupy (domyślnie)

1. **Dokumentacja Algorytmów** - pliki `*.md` z katalogów dokumentacji algorytmów
2. **Kod Python** - wszystkie pliki `*.py` w całym workspace
3. **Skrypty JSX** - pliki `*.jsx` ze skryptów Photoshop
4. **Konfiguracja i Dokumentacja** - pliki konfiguracyjne z katalogu głównego

## Kluczowe różnice v3 vs v2

### Workspace Root

- **v2**: Używa katalogu, w którym jest uruchamiany skrypt
- **v3**: Automatycznie ustawia workspace root na katalog wyżej niż lokalizacja skryptu

### Organizacja plików

- **v2**: Wszystkie pliki w jednej liście
- **v3**: Pliki podzielone na grupy tematyczne

### Konfiguracja

- **v2**: Konfiguracja w kodzie Python
- **v3**: Konfiguracja w zewnętrznym pliku YAML

### Struktura wyjścia

- **v2**: Prosta lista plików + zawartość
- **v3**: Spis grup + zawartość podzielona na grupy

## Przykład użycia

```bash
# Przejdź do katalogu projektu
cd d:\Unity\Projects\GattoNeroPhotoshop

# Uruchom nowy skrypt
python .doc-gen\.comb-scripts-v3.py

# Sprawdź wynik
type .comb-scripts.md
```

## Dostosowywanie

### Dodanie nowej grupy

Edytuj plik `.comb-scripts-config01.yaml`:

```yaml
groups:
  # ... istniejące grupy ...
  - name: "Moja Nowa Grupa"
    description: "Opis mojej grupy"
    patterns:
      - "*.txt"
      - "*.log"
    paths:
      - "logs"
      - "temp"
    recursive: true
```

### Zmiana nazwy pliku wyjściowego

W pliku YAML:

```yaml
output_file: ".moj-plik-wyjsciowy.md"
```

### Wykluczenie plików

Skrypt automatycznie respektuje reguły z `.gitignore`.

## Rozwiązywanie problemów

### Błąd: "No module named 'yaml'"

```bash
pip install PyYAML
```

### Błąd: "Nie znaleziono pliku konfiguracyjnego"

Upewnij się, że plik `.comb-scripts-config01.yaml` istnieje w katalogu `.doc-gen`.

### Puste grupy

Sprawdź czy ścieżki w konfiguracji YAML są poprawne względem workspace root.

## Migracja z v2 do v3

1. Zainstaluj PyYAML: `pip install PyYAML`
2. Dostosuj konfigurację YAML do swoich potrzeb
3. Uruchom v3: `python .doc-gen\.comb-scripts-v3.py`
4. Porównaj wyniki z v2
5. Po weryfikacji możesz usunąć v2

## Wsparcie

W przypadku problemów sprawdź:

1. Czy PyYAML jest zainstalowany
2. Czy plik konfiguracyjny YAML ma poprawną składnię
3. Czy ścieżki w konfiguracji istnieją
4. Czy masz uprawnienia do zapisu w katalogu docelowym
```
---