$ErrorActionPreference = "Stop"

# Skrypt do testowania wszystkich (lub większości) narzędzi serwera MCP Filesystem
# Zakłada, że serwer MCP jest już uruchomiony i nasłuchuje.
# Zakłada, że ten skrypt jest uruchamiany z katalogu mcp-test.

$TestToolScript = ".\test-tool.ps1"

Function Invoke-Tool {
    param (
        [string]$ToolName,
        [string]$ParamsJson,
        [string]$TestDescription
    )
    Write-Host "`n--- Test: $TestDescription ($ToolName) ---"
    Write-Host "Params: $ParamsJson"
    try {
        powershell -File $TestToolScript -ToolName $ToolName -ParamsJson $ParamsJson
        Write-Host "Test '$TestDescription' zakończony."
    } catch {
        Write-Error "Błąd podczas testu '$TestDescription': $_"
        # Można dodać opcję przerwania całego skryptu w razie błędu
    }
}

Write-Host "=== Rozpoczynam kompleksowe testowanie narzędzi MCP Filesystem ==="

# --- Przygotowanie środowiska testowego ---
$TempDir = "temp-test-dir"
$TempFile = "temp-test-file.txt"
$MovedFile = "$TempDir/moved.txt"

$TreeTestRoot = "tree_test_root"

# Usuń, jeśli istnieją z poprzednich testów
if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }
if (Test-Path $TempFile) { Remove-Item -Force $TempFile }
if (Test-Path $TreeTestRoot) { Remove-Item -Recurse -Force $TreeTestRoot }

# --- Tworzenie struktury dla directory_tree ---
New-Item -ItemType Directory -Path $TreeTestRoot -Force | Out-Null
New-Item -ItemType File -Path "$TreeTestRoot/file1.txt" -Value "content1" -Force | Out-Null
New-Item -ItemType Directory -Path "$TreeTestRoot/subdir1" -Force | Out-Null
New-Item -ItemType File -Path "$TreeTestRoot/subdir1/file2.txt" -Value "content2" -Force | Out-Null
New-Item -ItemType Directory -Path "$TreeTestRoot/subdir1/empty_subdir" -Force | Out-Null
New-Item -ItemType File -Path "$TreeTestRoot/file3.txt" -Value "content3" -Force | Out-Null

# --- Testy --- 

Invoke-Tool -ToolName "handshake" -ParamsJson '{}' -TestDescription "Handshake z serwerem"

Invoke-Tool -ToolName "list_tools" -ParamsJson '{}' -TestDescription "Listowanie dostępnych narzędzi"

Invoke-Tool -ToolName "list_allowed_directories" -ParamsJson '{}' -TestDescription "Listowanie dozwolonych katalogów"

Invoke-Tool -ToolName "read_file" -ParamsJson '{"path": "test_dir/test.txt"}' -TestDescription "Odczyt istniejącego pliku"

Invoke-Tool -ToolName "write_file" -ParamsJson ('{"path":"' + $TempFile + '","content":"Initial content"}') -TestDescription "Zapis nowego pliku"

Invoke-Tool -ToolName "edit_file" -ParamsJson ('{"path":"' + $TempFile + '","edits":[{"oldText":"Initial content","newText":"Edited content"}]}') -TestDescription "Edycja pliku"

Invoke-Tool -ToolName "create_directory" -ParamsJson ('{"path":"' + $TempDir + '"}') -TestDescription "Tworzenie nowego katalogu"

Invoke-Tool -ToolName "list_directory" -ParamsJson '{"path": "."}' -TestDescription "Listowanie zawartości bieżącego katalogu"

Invoke-Tool -ToolName "get_file_info" -ParamsJson ('{"path":"' + $TempFile + '"}') -TestDescription "Pobieranie informacji o pliku"

Invoke-Tool -ToolName "read_multiple_files" -ParamsJson '{"paths":["test-file.txt", "non-existent-file.txt", "test_dir/test.txt"], "encoding":"auto"}' -TestDescription "Odczyt wielu plików (dwa istnieją, jeden nie)"

Invoke-Tool -ToolName "search_files" -ParamsJson '{"path":".","pattern":"*.txt"}' -TestDescription "Wyszukiwanie plików *.txt"

Invoke-Tool -ToolName "move_file" -ParamsJson ('{"source":"' + $TempFile + '","destination":"' + $MovedFile + '"}') -TestDescription "Przenoszenie pliku"

Invoke-Tool -ToolName "delete_file" -ParamsJson ('{"path":"' + $MovedFile + '"}') -TestDescription "Usuwanie pliku"

Invoke-Tool -ToolName "delete_directory" -ParamsJson ('{"path":"' + $TempDir + '"}') -TestDescription "Usuwanie katalogu"

Invoke-Tool -ToolName "directory_tree" -ParamsJson ('{"path":"' + $TreeTestRoot + '"}') -TestDescription "Pobieranie drzewa katalogów dla '$TreeTestRoot'"

# --- Sprzątanie po directory_tree ---
if (Test-Path $TreeTestRoot) { Remove-Item -Recurse -Force $TreeTestRoot }

Write-Host "`n=== Wszystkie zaplanowane testy zakończone. ==="
