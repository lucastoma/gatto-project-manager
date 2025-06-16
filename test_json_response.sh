#!/bin/bash

echo "=== Testing MCP JSON-RPC Response ==="

response=$(echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "import os", "newText": "import os  # SUCCESS TEST"}], "dryRun": true}}}' | node /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/dist/server/index.js /home/lukasz/projects/gatto-ps-ai 2>/dev/null)

echo "FULL RESPONSE:"
echo "$response"
echo "========================"

if [ -z "$response" ]; then
    echo "ERROR: No JSON response received!"
else
    echo "SUCCESS: JSON response received!"
    echo "$response" | jq . 2>/dev/null || echo "Response is not valid JSON"
fi
