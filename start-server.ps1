$ErrorActionPreference = "Stop"

# Skrypt do budowania i uruchamiania serwera MCP Filesystem

Write-Host "Przechodzę do katalogu serwera: gatto_filesystem_v2"
Push-Location -Path ".\gatto_filesystem_v2" -ErrorAction Stop

Write-Host "Buduję serwer (npm run build)..."
npm run build

Write-Host "Wracam do katalogu głównego projektu"
Pop-Location

$allowedDir = "D:/projects/gatto-ps-ai-link1/mcp-test"
Write-Host "Uruchamiam serwer MCP Filesystem..."
Write-Host "Dozwolony katalog: $allowedDir"

# Uruchom serwer w nowym oknie PowerShell, aby nie blokować bieżącego
# Start-Process powershell -ArgumentList "-NoExit -Command node .\gatto_filesystem_v2\dist\server\index.js '$allowedDir'"

# Ustaw poziom logowania na debug
$env:LOG_LEVEL = "debug"
Write-Host "Poziom logowania ustawiony na: $env:LOG_LEVEL"

# Alternatywnie, uruchom w tym samym oknie (blokuje)
node .\gatto_filesystem_v2\dist\server\index.js $allowedDir

Write-Host "Serwer uruchomiony (jeśli nie było błędów)."