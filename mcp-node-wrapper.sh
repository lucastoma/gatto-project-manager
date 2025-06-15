#!/bin/bash
# MCP Node.js Wrapper Script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_PATH="$SCRIPT_DIR/.mcp-local-node/node-v20.10.0-linux-x64/bin/node"

# Check if Node.js exists
if [ ! -f "$NODE_PATH" ]; then
    echo "Error: Node.js not found at $NODE_PATH" >&2
    exit 1
fi

# Make sure it's executable
chmod +x "$NODE_PATH"

# Execute Node.js with all arguments
exec "$NODE_PATH" "$@"
