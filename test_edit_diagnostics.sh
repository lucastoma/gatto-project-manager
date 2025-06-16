#!/bin/bash

# Test edit_file with various scenarios to demonstrate improved diagnostics
echo "Testing edit_file with improved diagnostics..."

# Create test input for edit_file scenarios
cat > /tmp/test_edit_scenarios.json << 'EOF'
{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "def basic_transfer(", "newText": "def basic_transfer_enhanced("}], "dry_run": true}}}
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "import os", "newText": "import os\nimport sys"}], "dry_run": true}}}
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "nonexistent_text_that_will_fail", "newText": "replacement"}], "dry_run": true}}}
EOF

echo "Testing various edit scenarios..."
cd /home/lukasz/projects/gatto-ps-ai/gatto_filesystem
timeout 15s cat /tmp/test_edit_scenarios.json | node dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\nChecking brutal debug log for edit results..."
tail -100 /tmp/mcp-brutal-debug.log | grep -A 5 -B 2 "edit_file.*result"
