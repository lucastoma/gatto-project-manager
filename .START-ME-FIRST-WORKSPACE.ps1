# Plik: start.ps1
# Uruchamiaj w terminalu PowerShell

# Zatrzymuje skrypt w przypadku błędu
$ErrorActionPreference = "Stop"

echo "Krok 1: Budowanie obrazu Dockera 'gatto-repomix'..."
docker build -t gatto-repomix .
echo "Obraz został zbudowany pomyślnie."
echo "" # Pusta linia dla czytelności

echo "Krok 2: Uruchamianie kontenera MCP Server..."

# Montuje bieżący katalog jako /app i przekazuje /app/Workspace jako ścieżkę dla MCP
docker run -d --rm --name mcp_server -v "$($pwd.Path):/app" gatto-repomix --mcp /app

echo "Gotowe! Kontener 'mcp_server' został uruchomiony w tle."
echo "Możesz sprawdzić jego status poleceniem 'docker ps'."

