#!/bin/bash

# MCP Filesystem Server Wrapper
# Ensures Node.js is available and runs the MCP server

# Try to use NVM if available
if [ -f "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
    nvm use 22.16.0 2>/dev/null || true
fi

# Fallback to system node or direct path
if command -v node >/dev/null 2>&1; then
    exec node "$@"
elif [ -f "$HOME/.nvm/versions/node/v22.16.0/bin/node" ]; then
    exec "$HOME/.nvm/versions/node/v22.16.0/bin/node" "$@"
else
    echo "Error: Node.js not found" >&2
    exit 1
fi
