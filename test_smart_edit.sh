#!/bin/bash

# Test nowego systemu edit_file z różnymi progami similarity

echo "=== Testing MCP edit_file with Smart Similarity Matching ==="

# Test 1: EXACT MATCH (100%)
echo "Test 1: EXACT MATCH"
echo '{"method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "from app.core.development_logger import get_logger", "newText": "from app.core.development_logger import get_logger  # Exact match test"}]}}}' | node /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\n=========================\n"

# Test 2: HIGH SIMILARITY (98-100%) - should auto-edit
echo "Test 2: HIGH SIMILARITY - minor typo"
echo '{"method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "from app.core.performance_profiler import get_profler", "newText": "from app.core.performance_profiler import get_profiler  # Fixed typo"}]}}}' | node /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\n=========================\n"

# Test 3: MEDIUM SIMILARITY (85-97%) - should require force_edit
echo "Test 3: MEDIUM SIMILARITY - without force_edit"
echo '{"method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "def create_lab_transfer_processor():", "newText": "def create_lab_transfer_algorithm():"}]}}}' | node /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\n=========================\n"

# Test 4: MEDIUM SIMILARITY with force_edit=true
echo "Test 4: MEDIUM SIMILARITY - with force_edit=true"
echo '{"method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "def create_lab_transfer_processor():", "newText": "def create_lab_transfer_algorithm():"}], "force_edit": true}}}' | node /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\n=========================\n"

# Test 5: LOW SIMILARITY (60-84%) - diagnostics only
echo "Test 5: LOW SIMILARITY"
echo '{"method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "def completely_different_function_name():", "newText": "def new_function():"}]}}}' | node /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\n=========================\n"

# Test 6: INSIGNIFICANT (<60%) - "CZEGO KURWA SZUKASZ"
echo "Test 6: INSIGNIFICANT SIMILARITY"
echo '{"method": "tools/call", "params": {"name": "edit_file", "arguments": {"path": "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py", "edits": [{"oldText": "This text definitely does not exist in the file at all", "newText": "replacement"}]}}}' | node /home/lukasz/projects/gatto-ps-ai/gatto_filesystem/dist/server/index.js /home/lukasz/projects/gatto-ps-ai

echo -e "\n\n=== Test completed ==="
