@echo off
REM Buduje obraz Dockera z nazwą gatto-repomix
docker build -t gatto-repomix .

REM Uruchamia kontener repomix jako MCP server z montowaniem katalogu
docker run -i --rm -v %cd%:/workspace gatto-repomix --mcp