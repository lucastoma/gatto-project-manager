#!/bin/bash

# Ostre logowanie MCP filesystem server
# Utworzenie katalogu dla logów
mkdir -p /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/logs

# Uruchomienie serwera z logowaniem
echo "=== MCP Filesystem Server Log $(date) ===" >> /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/logs/mcp-debug.log

# Przekieruj stderr i stdout
exec 2>> /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/logs/mcp-debug.log
exec 1>> /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/logs/mcp-debug.log

echo "Starting MCP server with full logging..."
echo "Arguments: $@"
echo "Node version: $(node --version)"
echo "Working directory: $(pwd)"
echo "Environment PATH: $PATH"

# Uruchom serwer
node /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/dist/server/index.js "$@"
