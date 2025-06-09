import os
from pathlib import Path

# =================================================================================
# SCRIPT FOR FILE AGGREGATION
#
# Wersja: 4.0 (Uniwersalna)
# Opis: Skrypt wyszukuje pliki pasujące do podanych wzorców (np. *.md, *.py)
#       w określonych katalogach, filtruje je na podstawie .gitignore,
#       a następnie łączy ich zawartość w jeden zbiorczy plik .md.
# =================================================================================

# --- KONFIGURACJA ---
# W tej sekcji możesz dostosować działanie skryptu do swoich potrzeb.

# Nazwa projektu, która pojawi się w nagłówku pliku wyjściowego.
PROJECT_NAME = "Gatto Nero Ai Manager (PY+JSX)"

# Nazwa pliku, do którego zostanie zapisany wynik.
OUTPUT_FILE = ".comb-scripts.md"

# ---------------------------------------------------------------------------------
# WZORCE PLIKÓW (FILE_PATTERNS)
# ---------------------------------------------------------------------------------
# Zdefiniuj, jakie pliki mają być wyszukiwane. Możesz podać jeden lub wiele wzorców.
#
# PRZYKŁADY:
# - Tylko pliki Markdown:      FILE_PATTERNS = ['*.md']
# - Pliki Python i JSX:        FILE_PATTERNS = ['*.py', '*.jsx']
# - Tylko pliki tekstowe:      FILE_PATTERNS = ['*.txt']
#
FILE_PATTERNS = ['*.py', '*.jsx']
# ---------------------------------------------------------------------------------

# ŚCIEŻKI DO PRZESZUKANIA ($includePaths)
# Lista katalogów do przeszukania. Użyj ['all'] aby przeszukać wszystko.
INCLUDE_PATHS = ['all']

# Nazwa pliku .gitignore, używanego do wykluczeń.
GITIGNORE_FILE = ".gitignore"


# =================================================================================
# --- SILNIK SKRYPTU (zazwyczaj nie wymaga modyfikacji) ---
# =================================================================================

def load_gitignore_patterns(root_path):
    """Wczytuje i przetwarza wzorce z pliku .gitignore."""
    gitignore_path = root_path / GITIGNORE_FILE
    patterns = []
    if gitignore_path.exists():
        print("Znaleziono plik .gitignore. Stosuję reguły wykluczeń.")
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    patterns.append(stripped_line)
    else:
        print(".gitignore nie został znaleziony.")
    return patterns

def is_ignored(file_path, root_path, ignore_patterns):
    """Sprawdza, czy dany plik powinien być zignorowany na podstawie wzorców."""
    relative_path_str = str(file_path.relative_to(root_path).as_posix())
    for pattern in ignore_patterns:
        if pattern.endswith('/'):
            if f"/{pattern[:-1]}/" in f"/{relative_path_str}":
                return True
        elif file_path.match(pattern):
             return True
    return False

def find_files_to_process(root_path, ignore_patterns):
    """Znajduje wszystkie pliki do przetworzenia zgodnie z konfiguracją."""
    print(f"Wyszukiwanie plików pasujących do wzorców: {', '.join(FILE_PATTERNS)}...")
    all_found_files = []

    search_paths = []
    if 'all' in INCLUDE_PATHS:
        search_paths.append(root_path)
    else:
        for include_path in INCLUDE_PATHS:
            full_search_path = root_path / include_path
            if full_search_path.is_dir():
                search_paths.append(full_search_path)
            else:
                print(f"UWAGA: Ścieżka '{include_path}' nie istnieje i została pominięta.")

    for path in search_paths:
        for pattern in FILE_PATTERNS:
            # Używamy **/{pattern}, aby szukać rekursywnie
            all_found_files.extend(path.glob(f'**/{pattern}'))
    
    files_to_process = []
    for file in all_found_files:
        if file.name == OUTPUT_FILE or is_ignored(file, root_path, ignore_patterns):
            continue
        files_to_process.append(file)
            
    unique_files = sorted(list(set(files_to_process)))
    print(f"Znaleziono {len(unique_files)} unikalnych plików do przetworzenia.")
    return unique_files

def read_file_with_fallback_encoding(file_path):
    """Probuje odczytac plik jako UTF-8, a jesli sie nie uda, jako windows-1250."""
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

def main():
    """Główna funkcja skryptu."""
    root_path = Path.cwd()
    print(f"Rozpoczynam agregację dla projektu: {PROJECT_NAME}")

    ignore_patterns = load_gitignore_patterns(root_path)
    files_to_process = find_files_to_process(root_path, ignore_patterns)
    
    markdown_content = []
    markdown_content.append(f"# project name: {PROJECT_NAME}\n\nROOT: {root_path}\n\n---\n")
    markdown_content.append("## file tree list\n")
    if files_to_process:
        markdown_content.append(f"### Found Files ({len(files_to_process)})")
        for file in files_to_process:
            dir_path = str(file.parent.relative_to(root_path))
            dir_path = '' if dir_path == '.' else f"\\{dir_path}"
            markdown_content.append(f"- {file.name} ({dir_path})")
        markdown_content.append("")
    markdown_content.append("---\n")

    markdown_content.append("## file content\n")
    for file in files_to_process:
        relative_path = file.relative_to(root_path).as_posix()
        markdown_content.append(f"### {file.name} - ./{relative_path}\n")
        markdown_content.append('``````')
        content = read_file_with_fallback_encoding(file)
        markdown_content.append(content)
        markdown_content.append('``````\n')

    final_content = "\n".join(markdown_content)
    try:
        # Zapisujemy z 'utf-8-sig' dla najlepszej kompatybilnosci z programami na Windows
        with open(root_path / OUTPUT_FILE, 'w', encoding='utf-8-sig') as f:
            f.write(final_content)
        print(f"Gotowe! Plik '{OUTPUT_FILE}' został utworzony.")
    except Exception as e:
        print(f"BŁĄD ZAPISU PLIKU: {e}")

if __name__ == "__main__":
    main()