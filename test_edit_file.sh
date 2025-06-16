#!/bin/bash

# Test edit_file functionality comprehensively
echo "Testing edit_file functionality..."

# Create test input for various edit scenarios
cat > /tmp/test_edit_file.json << 'EOF'
{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "import os", "newText": "import os\n# Test comment added"}], "dry_run": true}}}
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "def basic_transfer_not_exists", "newText": "def basic_transfer_replaced"}], "dry_run": true}}}
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "def basic_lab_transfer", "newText": "def basic_lab_transfer_enhanced"}], "dry_run": true}}}
EOF

echo "Running edit_file tests..."
cd /home/lukasz/projects/gatto-ps-ai/gatto_filesystem
timeout 15s cat /tmp/test_edit_file.json | node dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\n=== Checking brutal debug log for edit_file activity ==="
tail -50 /tmp/mcp-brutal-debug.log | grep -A 5 -B 5 "edit_file"
