#!/bin/bash

# Test smart_search with queries that should find results
echo "Testing smart_search with different queries..."

# Create test input for MCP server
cat > /tmp/test_smart_search_comprehensive.json << 'EOF'
{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "smart_search", "arguments": {"query": "def transfer", "search_mode": "text", "base_paths": ["/home/lukasz/projects/gatto-ps-ai/app/algorithms"], "max_results": 10}}}
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "smart_search", "arguments": {"query": "class", "search_mode": "symbol", "base_paths": ["/home/lukasz/projects/gatto-ps-ai/app/algorithms"], "max_results": 5}}}
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "smart_search", "arguments": {"query": "algorithm", "search_mode": "hybrid", "base_paths": ["/home/lukasz/projects/gatto-ps-ai/app"], "max_results": 8}}}
EOF

# Run the test
echo "Sending comprehensive search requests..."
cd /home/lukasz/projects/gatto-ps-ai/gatto_filesystem
timeout 15s cat /tmp/test_smart_search_comprehensive.json | node dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\nChecking recent brutal debug log..."
tail -50 /tmp/mcp-brutal-debug.log | grep -A 5 -B 2 "smart_search"
