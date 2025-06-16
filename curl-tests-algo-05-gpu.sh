#!/bin/bash

# curl-tests-algo-05-gpu.sh - Test script for algorithm_05_lab_transfer with forced GPU

echo "Starting GPU tests for algorithm_05_lab_transfer"

# Unset environment variable to allow GPU usage
unset LAB_TRANSFER_USE_GPU

# Test 1: Basic LAB transfer with default parameters
curl -X POST -F "master_image=@/home/lukasz/projects/gatto-ps-ai/source/master.jpg" -F "target_image=@/home/lukasz/projects/gatto-ps-ai/source/target.jpg" -F "method=5" http://localhost:5000/api/colormatch -o result_gpu_basic.jpg
echo "Test 1 (Basic LAB Transfer GPU): Done"

# Test 2: LAB transfer with custom channels (only 'b')
curl -X POST -F "master_image=@/home/lukasz/projects/gatto-ps-ai/source/master.jpg" -F "target_image=@/home/lukasz/projects/gatto-ps-ai/source/target.jpg" -F "method=5" -F "channels=b" http://localhost:5000/api/colormatch -o result_gpu_channel_b.jpg
echo "Test 2 (Channel 'b' only GPU): Done"

# Test 3: LAB transfer with gpu method
curl -X POST -F "master_image=@/home/lukasz/projects/gatto-ps-ai/source/master.jpg" -F "target_image=@/home/lukasz/projects/gatto-ps-ai/source/target.jpg" -F "method=5" -F "processing_method=gpu" http://localhost:5000/api/colormatch -o result_gpu_forced.jpg
echo "Test 3 (Forced GPU): Done"

echo "GPU tests for algorithm_05_lab_transfer completed. Check output files: result_gpu_*.jpg"
