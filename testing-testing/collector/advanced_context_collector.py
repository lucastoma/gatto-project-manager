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
