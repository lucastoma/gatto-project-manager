import os
import yaml
from pathlib import Path
import fnmatch
import re  # Dodajemy import dla wyrażeń regularnych
import logging  # Added
import tempfile  # Added
import base64  # Added
import xml.etree.ElementTree as ET  # Added
from xml.dom import minidom  # Added for pretty printing
import time  # Added
import os  # Added for file operations
from repomix import RepoProcessor, RepomixConfig  # Import Python repomix library

# =================================================================================
# SCRIPT FOR FILE AGGREGATION WITH GROUPS AND EXCLUDE PATTERNS (REPOMIX INTEGRATION)
#
# Wersja: 8.0 (z pełną konfiguracją parametrów Repomix)
# Opis: Skrypt wykorzystuje Repomix do przetwarzania plików z zachowaniem
#       grupowania i wykluczeń zdefiniowanych w YAML, oraz deduplikacji plików.
# =================================================================================

DEFAULT_CONFIG_FILE = ".comb-scripts-config01.yaml"
TEMP_REPOMIX_OUTPUT_FILE = "temp_repomix_output.md"  # Tymczasowy plik wyjściowy Repomix


def get_default_repomix_options():
    """Zwraca domyślne opcje konfiguracyjne dla Repomix."""
    return {
        # Output options
        "style": "xml",  # Format wyjściowy: xml, markdown, plain
        "remove_comments": False,  # Usuwanie komentarzy z kodu
        "remove_empty_lines": False,  # Usuwanie pustych linii
        "show_line_numbers": False,  # Pokazywanie numerów linii
        "calculate_tokens": True,  # Obliczanie liczby tokenów
        "show_file_stats": True,  # Pokazywanie statystyk plików
        "show_directory_structure": True,  # Pokazywanie struktury katalogów
        "top_files_length": 2,  # Liczba top plików w statystykach
        "copy_to_clipboard": False,  # Kopiowanie do schowka
        "include_empty_directories": False,  # Dołączanie pustych katalogów
        
        # Compression options
        "compression": {
            "enabled": False,  # Włączenie kompresji
            "keep_signatures": True,  # Zachowanie sygnatur funkcji
            "keep_docstrings": True,  # Zachowanie docstringów
            "keep_interfaces": True,  # Zachowanie interfejsów
        },
        
        # Security options
        "security_check": True,  # Sprawdzanie bezpieczeństwa
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


def process_group_with_repomix(
    group, workspace_root, processed_files_set, config
):  # Added config
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

    # Get repomix options from defaults, then global config, then group config
    repomix_opts = get_default_repomix_options()
    
    # Update with global options from config file
    if "repomix_global_options" in config:
        repomix_opts.update(config["repomix_global_options"])
    
    # Update with group-specific options
    if "repomix_options" in group:
        group_opts = group["repomix_options"]
        # Handle nested compression options properly
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
            logging.warning(
                f"  UWAGA: Ścieżka '{path_str}' nie istnieje i została pominięta."
            )
            continue

        try:
            # Create RepomixConfig object
            repomix_config = RepomixConfig()
            
            # Configure output to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".xml", encoding="utf-8"
            ) as temp_output_file:
                temp_output_path = temp_output_file.name
            
            repomix_config.output.file_path = temp_output_path
            repomix_config.output.style = repomix_opts.get("style", "xml")
            
            # Configure include patterns
            if patterns:
                repomix_config.include = patterns
            
            # Configure exclude patterns
            if exclude_patterns:
                repomix_config.ignore.custom_patterns = exclude_patterns
            
            # Integrate .gitignore if specified
            if config.get("gitignore_file"):
                repomix_config.ignore.use_gitignore = True
            
            # Map output options
            repomix_config.output.show_line_numbers = repomix_opts.get("show_line_numbers", False)
            repomix_config.output.calculate_tokens = repomix_opts.get("calculate_tokens", True)
            repomix_config.output.show_file_stats = repomix_opts.get("show_file_stats", True)
            repomix_config.output.show_directory_structure = repomix_opts.get("show_directory_structure", True)
            repomix_config.output.top_files_length = repomix_opts.get("top_files_length", 2)
            repomix_config.output.copy_to_clipboard = repomix_opts.get("copy_to_clipboard", False)
            repomix_config.output.include_empty_directories = repomix_opts.get("include_empty_directories", False)
            repomix_config.output.remove_comments = repomix_opts.get("remove_comments", False)
            repomix_config.output.remove_empty_lines = repomix_opts.get("remove_empty_lines", False)
            
            # Map compression options
            compression_opts = repomix_opts.get("compression", {})
            repomix_config.compression.enabled = compression_opts.get("enabled", False)
            repomix_config.compression.keep_signatures = compression_opts.get("keep_signatures", True)
            repomix_config.compression.keep_docstrings = compression_opts.get("keep_docstrings", True)
            repomix_config.compression.keep_interfaces = compression_opts.get("keep_interfaces", True)
            
            # Map security options
            repomix_config.security.enable_security_check = repomix_opts.get("security_check", True)

            # Create processor and process
            processor = RepoProcessor(str(target_path), config=repomix_config)
            result = processor.process()
            
            # Collect statistics from result
            if result:
                stats_info = f"\n=== Statystyki dla grupy '{group_name}' - ścieżka '{path_str}' ===\n"
                if hasattr(result, 'total_files'):
                    stats_info += f"Łączna liczba plików: {result.total_files}\n"
                if hasattr(result, 'total_chars'):
                    stats_info += f"Łączna liczba znaków: {result.total_chars}\n"
                if hasattr(result, 'total_tokens'):
                    stats_info += f"Łączna liczba tokenów: {result.total_tokens}\n"
                
                # Add file statistics if available
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
                
                # Save stats to export directory
                export_dir = workspace_root / ".doc-gen" / "export"
                export_dir.mkdir(exist_ok=True)
                stats_file = export_dir / "repomix-stats.log"
                
                with open(stats_file, "a", encoding="utf-8") as f:
                    f.write(stats_info)
                
                logging.info(f"  Statystyki zapisane do: {stats_file}")

            # Read and parse the output file
            if os.path.exists(temp_output_path):
                with open(temp_output_path, "r", encoding="utf-8") as f:
                    output_content = f.read()
                os.remove(temp_output_path)  # Clean up temp file

                # Parse XML output if style is xml
                if repomix_opts.get("style", "xml") == "xml":
                    try:
                        root = ET.fromstring(output_content)
                        for file_elem in root.findall(".//file"):
                            file_path_elem = file_elem.find("path")
                            content_elem = file_elem.find("content")
                            
                            if file_path_elem is not None and content_elem is not None:
                                file_path_in_repomix = file_path_elem.text
                                content = content_elem.text or ""
                                
                                # Try to decode if it's base64 encoded
                                if content:
                                    try:
                                        decoded_content = base64.b64decode(content).decode("utf-8")
                                        content = decoded_content
                                    except:
                                        # If decoding fails, use content as is
                                        pass

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
                    except ET.ParseError as e:
                        logging.error(f"  BŁĄD parsowania XML dla grupy '{group_name}': {e}")
                else:
                    # For non-XML formats, we need to parse differently
                    # This is a simplified approach - you might need to adjust based on actual output format
                    logging.warning(f"  Format '{repomix_opts.get('style')}' nie jest w pełni obsługiwany w tej wersji")
            else:
                logging.warning(
                    f"  Repomix nie utworzył pliku wyjściowego dla grupy '{group_name}'."
                )

        except Exception as e:
            logging.error(f"BŁĄD przetwarzania grupy '{group_name}' z Python Repomix: {e}")
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
    output_file = config.get(
        "output_file", ".doc-gen/comb-scripts-output.xml"
    )  # Re-enabled, changed default to .xml

    logging.info(f"\nRozpoczynam agregację dla projektu: {project_name}")
    logging.info(f"Plik wyjściowy: {output_file}")  # Re-enabled

    # Initialize stats log file
    export_dir = workspace_root / ".doc-gen" / "export"
    export_dir.mkdir(exist_ok=True)
    stats_file = export_dir / "repomix-stats.log"
    
    # Clear previous stats
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"=== Statystyki Repomix dla projektu: {project_name} ===\n")
        f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
    
    logging.info(f"Plik statystyk: {stats_file}")

    # Przetwórz każdą grupę z wykluczaniem duplikatów
    all_groups_data = []
    processed_files_set = set()  # Zbiór już przetworzonych plików

    for i, group in enumerate(config.get("groups", [])):
        files_in_group, group_content_str = process_group_with_repomix(
            group, workspace_root, processed_files_set, config  # Added config
        )
        all_groups_data.append((group, files_in_group, group_content_str))

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

    # Generate XML string and then pretty print it
    rough_string = ET.tostring(root_elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    final_xml_string = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode(
        "utf-8"
    )

    output_path = workspace_root / output_file
    try:
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(final_xml_string)
        logging.info(
            f"\nGotowe! Plik '{output_path.name}' został utworzony w: {output_path}"
        )
    except Exception as e:
        logging.error(f"BŁĄD ZAPISU PLIKU: {e}")


if __name__ == "__main__":
    main()
