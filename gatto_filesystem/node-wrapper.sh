#!/bin/bash
# MCP Node.js wrapper script

# Załaduj NVM jeśli dostępne
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    source "$NVM_DIR/nvm.sh"
fi

# Użyj Node.js z pełną ścieżką jako fallback
NODE_PATH="/home/lukasz/.nvm/versions/node/v22.16.0/bin/node"

# Sprawdź czy node jest dostępne w PATH
if command -v node >/dev/null 2>&1; then
    exec node "$@"
elif [ -x "$NODE_PATH" ]; then
    exec "$NODE_PATH" "$@"
else
    echo "Error: Node.js not found" >&2
    exit 1
fi
