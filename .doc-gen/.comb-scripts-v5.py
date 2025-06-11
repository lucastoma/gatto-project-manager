import os
import yaml
from pathlib import Path
import fnmatch
import re  # Dodajemy import dla wyrażeń regularnych
from repomix import RepoProcessor  # Import RepoProcessor

# =================================================================================
# SCRIPT FOR FILE AGGREGATION WITH GROUPS AND EXCLUDE PATTERNS (REPOMIX INTEGRATION)
#
# Wersja: 7.0 (z pełną integracją Repomix)
# Opis: Skrypt wykorzystuje Repomix do przetwarzania plików z zachowaniem
#       grupowania i wykluczeń zdefiniowanych w YAML, oraz deduplikacji plików.
# =================================================================================

DEFAULT_CONFIG_FILE = ".comb-scripts-config01.yaml"
TEMP_REPOMIX_OUTPUT_FILE = "temp_repomix_output.md"  # Tymczasowy plik wyjściowy Repomix


def get_workspace_root():
    """Zwraca ścieżkę do workspace root."""
    return Path(__file__).parent.parent


def load_config(config_file_path):
    """Wczytuje konfigurację z pliku YAML."""
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"BŁĄD Wczytywania konfiguracji: {e}")
        return None


def process_group_with_repomix(group, workspace_root, processed_files_set):
    """
    Przetwarza grupę plików używając Repomix, z uwzględnieniem deduplikacji.
    Zwraca listę ścieżek do unikalnych plików oraz ich zawartość.
    """
    group_name = group.get("name", "Unnamed Group")
    patterns = group.get("patterns", [])
    exclude_patterns = group.get("exclude_patterns", [])
    paths = group.get("paths", [])

    print(f"\nPrzetwarzanie grupy: {group_name}")

    group_files_content = []
    unique_files_in_group = []

    for path_str in paths:
        target_path = (
            workspace_root / path_str
            if path_str not in ["all", ".", "**/*", "**"]
            else workspace_root
        )

        if not target_path.exists():
            print(f"  UWAGA: Ścieżka '{path_str}' nie istnieje i została pominięta.")
            continue

        # Przygotowanie parametrów dla RepoProcessor
        # Repomix używa glob patterns, więc łączymy je w stringi
        include_patterns_str = ",".join(patterns)
        # Dodajemy wzorce wykluczeń z konfiguracji grupy
        ignore_patterns_str = ",".join(exclude_patterns)

        try:
            processor = RepoProcessor(
                directory=str(target_path),
                include_patterns=include_patterns_str,
                ignore_patterns=ignore_patterns_str,
                output_file_path=TEMP_REPOMIX_OUTPUT_FILE,
                style="markdown",  # Możesz zmienić na "xml" jeśli potrzebujesz strukturyzowanego wyjścia
                # Dodatkowe parametry Repomix, jeśli są potrzebne z konfiguracji YAML
                # np. compress=group.get('compress', False),
                # show_line_numbers=group.get('show_line_numbers', False),
                # calculate_tokens=group.get('calculate_tokens', False),
                # security_check=group.get('security_check', True),
            )

            result = processor.process()

            # Repomix generuje plik z całą strukturą i zawartością.
            # Musimy teraz wyodrębnić poszczególne pliki i sprawdzić deduplikację.
            with open(TEMP_REPOMIX_OUTPUT_FILE, "r", encoding="utf-8") as f:
                repomix_output = f.read()
            os.remove(TEMP_REPOMIX_OUTPUT_FILE)  # Usuwamy tymczasowy plik

            # Parsowanie wyjścia Repomix, aby odzyskać poszczególne pliki
            # To jest uproszczone parsowanie Markdown, może wymagać dostosowania
            # w zależności od dokładnego formatu wyjścia Repomix.
            # Lepszym rozwiązaniem byłoby użycie stylu "xml" i parsowanie XML.

            # Szukamy nagłówków plików w wyjściu Repomix Markdown
            # Format: ## File: ścieżka/do/pliku.py
            file_sections = re.split(
                r"## File: (.+?)\n```(?:python|text|json|yaml|bash|markdown)?\n",
                repomix_output,
            )

            # Pierwszy element splitu to zazwyczaj nagłówek Repomix, ignorujemy go
            for i in range(1, len(file_sections), 2):
                file_path_in_repomix = file_sections[i].strip()
                content_block = file_sections[i + 1]

                # Usuwamy końcowe ```
                if content_block.endswith("\n```\n"):
                    content = content_block[:-5]
                else:
                    content = content_block

                # Tworzymy obiekt Path dla ścieżki pliku
                # Zakładamy, że ścieżka w Repomix jest względna do workspace_root
                full_file_path = workspace_root / file_path_in_repomix

                if full_file_path not in processed_files_set:
                    processed_files_set.add(full_file_path)
                    unique_files_in_group.append(full_file_path)
                    group_files_content.append(
                        f"#### {full_file_path.name} - ./{file_path_in_repomix}\n```\n{content}\n```\n"
                    )
                else:
                    print(f"  Plik '{file_path_in_repomix}' już przetworzony, pomijam.")

        except Exception as e:
            print(f"BŁĄD przetwarzania grupy '{group_name}' z Repomix: {e}")
            # W przypadku błędu, możemy wrócić do oryginalnej logiki lub pominąć grupę
            # Na potrzeby tego zadania, po prostu kontynuujemy
            continue

    print(
        f"  Znaleziono {len(unique_files_in_group)} unikalnych plików w grupie '{group_name}'."
    )
    return unique_files_in_group, "\n".join(group_files_content)


def generate_output(config, workspace_root, all_groups_data):
    """Generuje finalny plik Markdown."""
    output = [
        f"# {config.get('project_name', 'Unknown Project')}",
        f"WORKSPACE ROOT: {workspace_root}",
        "---\n## Spis Grup\n",
    ]

    total_unique_files = 0
    for i, (group, files_list, _) in enumerate(all_groups_data, 1):
        file_count = len(files_list)
        total_unique_files += file_count
        output.append(
            f"{i}. **{group.get('name', f'Grupa {i}')}** ({file_count} plików)"
        )
        if desc := group.get("description"):
            output.append(f"   - {desc}")
        output.append("")

    output.append(f"**Łącznie unikalnych plików: {total_unique_files}**\n\n---\n")

    # Zawartość grup
    for i, (group, files_list, group_content_str) in enumerate(all_groups_data, 1):
        output.append(f"## Grupa {i}: {group.get('name', f'Grupa {i}')}\n")
        if desc := group.get("description"):
            output.append(f"*{desc}*\n")

        if not files_list:
            output.append("*Brak plików w tej grupie.*\n\n---\n")
            continue

        output.append(f"### Lista plików ({len(files_list)})\n")
        for file in files_list:
            try:
                relative_path = file.relative_to(workspace_root).as_posix()
                dir_path = str(file.relative_to(workspace_root).parent)
                dir_path = "" if dir_path == "." else f"\\{dir_path}"
                output.append(f"- {file.name} ({dir_path})")
            except ValueError:
                output.append(f"- {file.name} (poza workspace)")
        output.append("")

        output.append("### Zawartość plików\n")
        output.append(
            group_content_str
        )  # Dodajemy już przetworzoną zawartość z Repomix
        output.append("---\n")

    return "\n".join(output)


def main():
    """Główna funkcja skryptu."""
    import sys

    workspace_root = get_workspace_root()

    # Wczytaj konfigurację (sprawdź czy podano plik jako argument)
    if len(sys.argv) > 1:
        config_file_path = Path(sys.argv[1])  # Pełna ścieżka do pliku konfiguracyjnego
        if not config_file_path.is_absolute():
            config_file_path = Path(__file__).parent / config_file_path
    else:
        config_file_path = workspace_root / DEFAULT_CONFIG_FILE # Zmieniamy ścieżkę na workspace_root

    print(f"Używam pliku konfiguracyjnego: {config_file_path}")
    config = load_config(config_file_path)
    if not config:
        return

    project_name = config.get("project_name", "Unknown Project")
    output_file = config.get("output_file", ".doc-gen/comb-scripts.md")

    print(f"\nRozpoczynam agregację dla projektu: {project_name}")
    print(f"Plik wyjściowy: {output_file}")

    # Przetwórz każdą grupę z wykluczaniem duplikatów
    all_groups_data = []
    processed_files_set = set()  # Zbiór już przetworzonych plików

    for i, group in enumerate(config.get("groups", [])):
        files_in_group, group_content_str = process_group_with_repomix(
            group, workspace_root, processed_files_set
        )
        all_groups_data.append((group, files_in_group, group_content_str))

    # Generuj zawartość Markdown
    markdown_content = generate_output(config, workspace_root, all_groups_data)

    # Zapisz plik
    output_path = workspace_root / output_file

    try:
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(markdown_content)
        print(f"\nGotowe! Plik '{output_path.name}' został utworzony w: {output_path}")
    except Exception as e:
        print(f"BŁĄD ZAPISU PLIKU: {e}")


if __name__ == "__main__":
    main()
