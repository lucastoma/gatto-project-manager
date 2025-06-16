#!/bin/bash

# Test edit_file WITHOUT dry_run to see actual changes
echo "Testing edit_file WITHOUT dry_run..."

# Create test input for actual edit
cat > /tmp/test_edit_real.json << 'EOF'
{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}}}
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "import os", "newText": "import os\n# Test comment added by edit_file"}], "dry_run": false}}}
EOF

echo "Before edit - checking current import lines:"
head -10 /home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py | grep -n "import"

echo -e "\nRunning REAL edit (no dry_run)..."
cd /home/lukasz/projects/gatto-ps-ai/gatto_filesystem
timeout 10s cat /tmp/test_edit_real.json | node dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\nAfter edit - checking if comment was added:"
head -15 /home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py | grep -A 2 -B 2 "import os"
