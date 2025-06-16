#!/bin/bash

# Test the fixed server_stats and list_allowed_directories tools
echo "Testing fixed tools..."

# Create test input
cat > /tmp/test_fixed_tools.json << 'EOF'
{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "server_stats", "arguments": {}}}
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_allowed_directories", "arguments": {}}}
EOF

echo "Testing server_stats and list_allowed_directories..."
cd /home/lukasz/projects/gatto-ps-ai/gatto_filesystem
timeout 10s cat /tmp/test_fixed_tools.json | node dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\nChecking brutal debug log..."
tail -20 /tmp/mcp-brutal-debug.log
