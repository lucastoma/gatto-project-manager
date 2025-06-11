import os
import yaml
from pathlib import Path
import fnmatch

# =================================================================================
# SCRIPT FOR FILE AGGREGATION WITH GROUPS AND EXCLUDE PATTERNS
#
# Wersja: 5.1 (z grupami, konfiguracją YAML i exclude_patterns)
# Opis: Skrypt wyszukuje pliki w grupach zdefiniowanych w pliku YAML,
#       filtruje je na podstawie .gitignore i exclude_patterns,
#       a następnie łączy ich zawartość w jeden zbiorczy plik .md z podziałem na grupy.
# =================================================================================

# Domyślna nazwa pliku konfiguracyjnego
DEFAULT_CONFIG_FILE = ".comb-scripts-config01.yaml"

def get_workspace_root():
    """Zwraca ścieżkę do workspace root (katalog wyżej niż lokalizacja skryptu)."""
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    return workspace_root

def load_config(config_file_path):
    """Wczytuje konfigurację z pliku YAML."""
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Wczytano konfigurację z: {config_file_path}")
        return config
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku konfiguracyjnego: {config_file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"BŁĄD: Nieprawidłowy format YAML: {e}")
        return None
    except Exception as e:
        print(f"BŁĄD: Nie można wczytać konfiguracji: {e}")
        return None

def load_gitignore_patterns(workspace_root, gitignore_file):
    """Wczytuje i przetwarza wzorce z pliku .gitignore."""
    gitignore_path = workspace_root / gitignore_file
    patterns = []
    if gitignore_path.exists():
        print(f"Znaleziono plik .gitignore: {gitignore_path}")
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    patterns.append(stripped_line)
    else:
        print(f"Plik .gitignore nie został znaleziony: {gitignore_path}")
    return patterns

def is_ignored(file_path, workspace_root, ignore_patterns):
    """Sprawdza, czy dany plik powinien być zignorowany na podstawie wzorców."""
    try:
        relative_path_str = str(file_path.relative_to(workspace_root).as_posix())
        for pattern in ignore_patterns:
            if pattern.endswith('/'):
                if f"/{pattern[:-1]}/" in f"/{relative_path_str}":
                    return True
            elif file_path.match(pattern):
                return True
    except ValueError:
        # Plik jest poza workspace_root
        return True
    return False

def matches_exclude_pattern(file_path, exclude_patterns):
    """Sprawdza, czy plik pasuje do wzorców wykluczenia."""
    if not exclude_patterns:
        return False
    
    file_name = file_path.name
    file_path_str = str(file_path)
    
    for pattern in exclude_patterns:
        # Sprawdź czy wzorzec pasuje do nazwy pliku
        if fnmatch.fnmatch(file_name.lower(), pattern.lower()):
            return True
        # Sprawdź czy wzorzec pasuje do pełnej ścieżki
        if fnmatch.fnmatch(file_path_str.lower(), f"*{pattern.lower()}*"):
            return True
    
    return False

def find_files_for_group(group, workspace_root, ignore_patterns):
    """Znajduje pliki dla konkretnej grupy."""
    group_name = group.get('name', 'Unnamed Group')
    patterns = group.get('patterns', [])
    exclude_patterns = group.get('exclude_patterns', [])
    paths = group.get('paths', [])
    recursive = group.get('recursive', True)
    
    print(f"\nPrzetwarzanie grupy: {group_name}")
    print(f"  Wzorce: {', '.join(patterns)}")
    if exclude_patterns:
        print(f"  Wykluczenia: {', '.join(exclude_patterns)}")
    print(f"  Ścieżki: {', '.join(paths)}")
    print(f"  Rekursywnie: {recursive}")
    
    all_found_files = []
    
    # Określ ścieżki do przeszukania
    search_paths = []
    for path_str in paths:
        if path_str == 'all':
            search_paths.append(workspace_root)
        elif path_str == '.':
            search_paths.append(workspace_root)
        elif path_str == '**/*' or path_str == '**':
            # Wzorzec **/* oznacza przeszukiwanie całego workspace rekursywnie
            search_paths.append(workspace_root)
        else:
            full_path = workspace_root / path_str
            if full_path.exists() and full_path.is_dir():
                search_paths.append(full_path)
            else:
                print(f"  UWAGA: Ścieżka '{path_str}' nie istnieje i została pominięta.")
    
    # Wyszukaj pliki
    for search_path in search_paths:
        for pattern in patterns:
            if recursive:
                found_files = search_path.glob(f'**/{pattern}')
            else:
                found_files = search_path.glob(pattern)
            all_found_files.extend(found_files)
    
    # Filtruj pliki
    files_to_process = []
    excluded_count = 0
    
    for file in all_found_files:
        if file.is_file():
            # Sprawdź .gitignore
            if is_ignored(file, workspace_root, ignore_patterns):
                continue
            
            # Sprawdź exclude_patterns
            if matches_exclude_pattern(file, exclude_patterns):
                excluded_count += 1
                continue
            
            files_to_process.append(file)
    
    unique_files = sorted(list(set(files_to_process)))
    print(f"  Znaleziono {len(unique_files)} plików (wykluczono {excluded_count})")
    
    return unique_files

def read_file_with_fallback_encoding(file_path):
    """Próbuje odczytać plik jako UTF-8, a jeśli się nie uda, jako windows-1250."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        print(f"  -> Plik '{file_path.name}' nie jest w UTF-8, próba odczytu jako windows-1250.")
        try:
            with open(file_path, 'r', encoding='windows-1250') as f:
                return f.read()
        except Exception as e:
            return f"BŁĄD ODCZYTU PLIKU {file_path.name}: {e}"
    except Exception as e:
        return f"NIEOCZEKIWANY BŁĄD ODCZYTU PLIKU {file_path.name}: {e}"

def generate_markdown_content(config, workspace_root, all_groups_files):
    """Generuje zawartość pliku Markdown."""
    project_name = config.get('project_name', 'Unknown Project')
    
    markdown_content = []
    markdown_content.append(f"# project name: {project_name}\n")
    markdown_content.append(f"WORKSPACE ROOT: {workspace_root}\n")
    markdown_content.append(f"SCRIPT LOCATION: {Path(__file__).parent}\n\n---\n")
    
    # Spis treści grup
    markdown_content.append("## Spis Grup\n")
    total_files = 0
    for i, (group, files) in enumerate(all_groups_files, 1):
        group_name = group.get('name', f'Grupa {i}')
        group_desc = group.get('description', '')
        file_count = len(files)
        total_files += file_count
        
        markdown_content.append(f"{i}. **{group_name}** ({file_count} plików)")
        if group_desc:
            markdown_content.append(f"   - {group_desc}")
        
        # Pokaż wzorce wykluczenia jeśli są
        exclude_patterns = group.get('exclude_patterns', [])
        if exclude_patterns:
            markdown_content.append(f"   - Wykluczenia: {', '.join(exclude_patterns)}")
        
        markdown_content.append("")
    
    markdown_content.append(f"**Łącznie plików: {total_files}**\n\n---\n")
    
    # Zawartość grup
    for i, (group, files) in enumerate(all_groups_files, 1):
        group_name = group.get('name', f'Grupa {i}')
        group_desc = group.get('description', '')
        
        markdown_content.append(f"## Grupa {i}: {group_name}\n")
        if group_desc:
            markdown_content.append(f"*{group_desc}*\n")
        
        if not files:
            markdown_content.append("*Brak plików w tej grupie.*\n\n---\n")
            continue
        
        # Lista plików w grupie
        markdown_content.append(f"### Lista plików ({len(files)})\n")
        for file in files:
            try:
                relative_path = file.relative_to(workspace_root)
                dir_path = str(relative_path.parent)
                dir_path = '' if dir_path == '.' else f"\\{dir_path}"
                markdown_content.append(f"- {file.name} ({dir_path})")
            except ValueError:
                markdown_content.append(f"- {file.name} (poza workspace)")
        markdown_content.append("")
        
        # Zawartość plików
        markdown_content.append("### Zawartość plików\n")
        for file in files:
            try:
                relative_path = file.relative_to(workspace_root).as_posix()
                markdown_content.append(f"#### {file.name} - ./{relative_path}\n")
            except ValueError:
                markdown_content.append(f"#### {file.name} - {file}\n")
            
            markdown_content.append('``````')
            content = read_file_with_fallback_encoding(file)
            markdown_content.append(content)
            markdown_content.append('``````\n')
        
        markdown_content.append("---\n")
    
    return "\n".join(markdown_content)

def main():
    """Główna funkcja skryptu."""
    import sys
    
    # Ustaw workspace root
    workspace_root = get_workspace_root()
    print(f"Workspace Root: {workspace_root}")
    print(f"Script Location: {Path(__file__).parent}")
    
    # Wczytaj konfigurację (sprawdź czy podano plik jako argument)
    if len(sys.argv) > 1:
        config_file_path = Path(sys.argv[1])  # Pełna ścieżka do pliku konfiguracyjnego
        if not config_file_path.is_absolute():
            config_file_path = Path(__file__).parent / config_file_path
    else:
        config_file_path = Path(__file__).parent / DEFAULT_CONFIG_FILE
    
    # Sprawdź czy podano katalog export jako drugi argument
    export_dir = None
    if len(sys.argv) > 2:
        export_dir = Path(sys.argv[2])
        print(f"Katalog export: {export_dir}")
    
    print(f"Używam pliku konfiguracyjnego: {config_file_path}")
    config = load_config(config_file_path)
    if not config:
        return
    
    project_name = config.get('project_name', 'Unknown Project')
    output_file = config.get('output_file', '.doc-gen/comb-scripts.md')
    gitignore_file = config.get('gitignore_file', '.gitignore')
    groups = config.get('groups', [])
    
    print(f"\nRozpoczynam agregację dla projektu: {project_name}")
    print(f"Plik wyjściowy: {output_file}")
    print(f"Liczba grup: {len(groups)}")
    
    # Wczytaj wzorce .gitignore
    ignore_patterns = load_gitignore_patterns(workspace_root, gitignore_file)
    
    # Przetwórz każdą grupę z wykluczaniem duplikatów
    all_groups_files = []
    already_processed_files = set()  # Zbiór już przetworzonych plików
    
    for i, group in enumerate(groups):
        files = find_files_for_group(group, workspace_root, ignore_patterns)
        
        # Wykluczamy pliki już przetworzone w poprzednich grupach
        unique_files = []
        duplicates_count = 0
        
        for file in files:
            if file not in already_processed_files:
                unique_files.append(file)
                already_processed_files.add(file)
            else:
                duplicates_count += 1
        
        print(f"  Grupa {i+1}: {len(unique_files)} unikalnych plików (wykluczono {duplicates_count} duplikatów)")
        all_groups_files.append((group, unique_files))
    
    # Generuj zawartość Markdown
    markdown_content = generate_markdown_content(config, workspace_root, all_groups_files)
    
    # Zapisz plik - użyj katalogu export jeśli został podany
    if export_dir:
        # Jeśli podano katalog export, zapisz tam plik
        export_dir.mkdir(parents=True, exist_ok=True)
        output_filename = Path(output_file).name  # Tylko nazwa pliku bez ścieżki
        output_path = export_dir / output_filename
        print(f"Zapisuję do katalogu export: {export_dir}")
    else:
        # Standardowa ścieżka względem workspace_root
        output_path = workspace_root / output_file
    
    try:
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write(markdown_content)
        print(f"\nGotowe! Plik '{output_path.name}' został utworzony w: {output_path}")
    except Exception as e:
        print(f"BŁĄD ZAPISU PLIKU: {e}")

if __name__ == "__main__":
    main()