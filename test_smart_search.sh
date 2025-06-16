#!/bin/bash

# Test smart_search with basic query
echo "Testing smart_search tool..."

# Create test input for MCP server
cat > /tmp/test_smart_search_input.json << 'EOF'
{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "smart_search", "arguments": {"query": "basic_transfer", "search_mode": "text", "base_paths": ["/home/lukasz/projects/gatto-ps-ai/app/algorithms"], "max_results": 5}}}
EOF

# Run the test
echo "Sending requests to MCP server..."
cd /home/lukasz/projects/gatto-ps-ai/gatto_filesystem
timeout 10s cat /tmp/test_smart_search_input.json | node dist/server/index.js

echo -e "\n\nChecking brutal debug log..."
tail -20 /tmp/mcp-brutal-debug.log
