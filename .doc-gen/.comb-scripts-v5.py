import os
import yaml
from pathlib import Path
import fnmatch
import re  # Dodajemy import dla wyrażeń regularnych
import logging  # Added
import tempfile  # Added
import base64  # Added
import xml.etree.ElementTree as ET  # Added
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
        logging.error(f"BŁĄD Wczytywania konfiguracji: {e}")
        return None


def process_group_with_repomix(
    group, workspace_root, processed_files_set, config
):  # Added config
    """
    Przetwarza grupę plików używając Repomix, z uwzględnieniem deduplikacji.
    Zwraca listę ścieżek do unikalnych plików oraz ich zawartość.
    """
    group_name = group.get("name", "Unnamed Group")
    patterns = group.get("patterns", [])
    exclude_patterns = group.get("exclude_patterns", [])
    paths = group.get("paths", [])

    logging.info(f"\nPrzetwarzanie grupy: {group_name}")

    group_files_content = []
    unique_files_in_group = []

    # Get repomix options from group, with global defaults
    repomix_opts = config.get("repomix_global_options", {}).copy()
    repomix_opts.update(group.get("repomix_options", {}))

    # Ensure style is xml by default if not specified
    repomix_opts.setdefault("style", "xml")

    for path_str in paths:
        target_path = (
            workspace_root / path_str
            if path_str not in ["all", ".", "**/*", "**"]
            else workspace_root
        )

        if not target_path.exists():
            logging.warning(
                f"  UWAGA: Ścieżka '{path_str}' nie istnieje i została pominięta."
            )
            continue

        # Przygotowanie parametrów dla RepoProcessor
        # Repomix używa glob patterns, więc łączymy je w stringi
        include_patterns_str = ",".join(patterns)
        # Dodajemy wzorce wykluczeń z konfiguracji grupy
        ignore_patterns_str = ",".join(exclude_patterns)

        # Integrate .gitignore if specified
        if gitignore_file := config.get("gitignore_file"):
            gitignore_path = workspace_root / gitignore_file
            if gitignore_path.exists():
                try:
                    with open(gitignore_path, "r", encoding="utf-8") as f:
                        gitignore_patterns = [
                            line.strip()
                            for line in f
                            if line.strip() and not line.startswith("#")
                        ]
                    if gitignore_patterns:
                        ignore_patterns_str = ",".join(
                            filter(None, [ignore_patterns_str] + gitignore_patterns)
                        )
                except Exception as e:
                    logging.warning(
                        f"  Nie udało się wczytać pliku .gitignore '{gitignore_file}': {e}"
                    )

        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".xml", encoding="utf-8"
            ) as temp_output_file:
                temp_output_path = temp_output_file.name

                processor = RepoProcessor(
                    directory=str(target_path),
                    include_patterns=include_patterns_str,
                    ignore_patterns=ignore_patterns_str,
                    output_file_path=temp_output_path,
                    **repomix_opts,  # Pass all options dynamically
                )

                result = processor.process()

            # New XML parsing logic
            if result and os.path.exists(
                temp_output_path
            ):  # Check temp_output_path directly
                with open(temp_output_path, "r", encoding="utf-8") as f:
                    xml_output = f.read()
                os.remove(temp_output_path)  # Clean up temp file

                root = ET.fromstring(xml_output)
                for file_elem in root.findall(".//file"):
                    file_path_in_repomix = file_elem.find("path").text
                    encoded_content = file_elem.find("content").text

                    if encoded_content:
                        try:
                            content = base64.b64decode(encoded_content).decode("utf-8")
                        except Exception as decode_e:
                            logging.warning(
                                f"  BŁĄD dekodowania zawartości pliku '{file_path_in_repomix}': {decode_e}. Zawartość zostanie pusta."
                            )
                            content = ""
                    else:
                        content = ""

                    full_file_path = workspace_root / file_path_in_repomix

                    if full_file_path not in processed_files_set:
                        processed_files_set.add(full_file_path)
                        unique_files_in_group.append(full_file_path)
                        group_files_content.append(
                            {"path": file_path_in_repomix, "content": content}
                        )
                    else:
                        logging.info(
                            f"  Plik '{file_path_in_repomix}' już przetworzony, pomijam."
                        )
            else:
                logging.warning(
                    f"  Repomix nie zwrócił pliku wyjściowego dla grupy '{group_name}'."
                )

        except Exception as e:
            logging.error(f"BŁĄD przetwarzania grupy '{group_name}' z Repomix: {e}")
            # W przypadku błędu, możemy wrócić do oryginalnej logiki lub pominąć grupę
            # Na potrzeby tego zadania, po prostu kontynuujemy
            continue

    logging.info(
        f"  Znaleziono {len(unique_files_in_group)} unikalnych plików w grupie '{group_name}'."
    )
    return unique_files_in_group, group_files_content


def main():
    """Główna funkcja skryptu."""
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )  # Added

    workspace_root = get_workspace_root()

    # Wczytaj konfigurację (sprawdź czy podano plik jako argument)
    if len(sys.argv) > 1:
        config_file_path = Path(sys.argv[1])  # Pełna ścieżka do pliku konfiguracyjnego
        if not config_file_path.is_absolute():
            config_file_path = Path(__file__).parent / config_file_path
    else:
        config_file_path = (
            workspace_root / DEFAULT_CONFIG_FILE
        )  # Zmieniamy ścieżkę na workspace_root

    logging.info(f"Używam pliku konfiguracyjnego: {config_file_path}")
    config = load_config(config_file_path)
    if not config:
        return

    project_name = config.get("project_name", "Unknown Project")
    # output_file = config.get("output_file", ".doc-gen/comb-scripts.md") # Removed

    logging.info(f"\nRozpoczynam agregację dla projektu: {project_name}")
    # logging.info(f"Plik wyjściowy: {output_file}")

    # Przetwórz każdą grupę z wykluczaniem duplikatów
    all_groups_data = []
    processed_files_set = set()  # Zbiór już przetworzonych plików

    for i, group in enumerate(config.get("groups", [])):
        files_in_group, group_content_str = process_group_with_repomix(
            group, workspace_root, processed_files_set, config  # Added config
        )
        all_groups_data.append((group, files_in_group, group_content_str))

    # Generuj zawartość Markdown
    # markdown_content = generate_output(config, workspace_root, all_groups_data) # Removed

    # Zapisz plik
    # output_path = workspace_root / output_file # Removed

    # try:
    #     with open(output_path, "w", encoding="utf-8-sig") as f:
    #         f.write(markdown_content)
    #     logging.info(
    #         f"\nGotowe! Plik '{output_path.name}' został utworzony w: {output_path}"
    #     )
    # except Exception as e:
    #     logging.error(f"BŁĄD ZAPISU PLIKU: {e}")

    # Construct and print final XML output
    root_elem = ET.Element("AggregatedCodebase")
    project_elem = ET.SubElement(root_elem, "Project", name=project_name)
    ET.SubElement(project_elem, "WorkspaceRoot").text = str(workspace_root)
    ET.SubElement(project_elem, "TotalUniqueFiles").text = str(len(processed_files_set))

    for i, (group, files_list, group_content_data) in enumerate(all_groups_data, 1):
        group_elem = ET.SubElement(
            project_elem, "Group", name=group.get("name", f"Group {i}")
        )
        if desc := group.get("description"):
            ET.SubElement(group_elem, "Description").text = desc
        ET.SubElement(group_elem, "FileCount").text = str(len(files_list))

        files_list_elem = ET.SubElement(group_elem, "FilesList")
        for file_path_obj in files_list:
            file_elem = ET.SubElement(files_list_elem, "File")
            ET.SubElement(file_elem, "Path").text = file_path_obj.relative_to(
                workspace_root
            ).as_posix()
            ET.SubElement(file_elem, "Name").text = file_path_obj.name

        content_elem = ET.SubElement(group_elem, "Content")
        for file_data in group_content_data:
            file_content_elem = ET.SubElement(content_elem, "FileContent")
            ET.SubElement(file_content_elem, "Path").text = file_data["path"]
            ET.SubElement(file_content_elem, "Content").text = file_data["content"]

    # Print the final XML to stdout
    final_xml_string = ET.tostring(
        root_elem, encoding="utf-8", pretty_print=True
    ).decode("utf-8")
    print(final_xml_string)  # Print to stdout for AI consumption
    logging.info("\nAggregated XML output generated and printed to stdout.")


if __name__ == "__main__":
    main()
