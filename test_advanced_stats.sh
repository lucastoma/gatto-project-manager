#!/bin/bash

# Test the enhanced server_stats with advanced parameter
echo "Testing server_stats with advanced parameter..."

# Create test input
cat > /tmp/test_advanced_stats.json << 'EOF'
{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "server_stats", "arguments": {"advanced": false}}}
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "server_stats", "arguments": {"advanced": true}}}
EOF

echo "Testing basic and advanced server_stats..."
cd /home/lukasz/projects/gatto-ps-ai/gatto_filesystem
timeout 15s cat /tmp/test_advanced_stats.json | node dist/server/index.js /home/lukasz/projects/gatto-ps-ai
