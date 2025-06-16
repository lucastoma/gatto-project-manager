#!/bin/bash

# curl-tests-algo-05-cpu.sh - Test script for algorithm_05_lab_transfer with forced CPU

echo "Starting CPU tests for algorithm_05_lab_transfer"

# Set environment variable to force CPU usage
export LAB_TRANSFER_USE_GPU=0

# Test 1: Basic LAB transfer with default parameters
curl -X POST -F "master_image=@/home/lukasz/projects/gatto-ps-ai/source/master.jpg" -F "target_image=@/home/lukasz/projects/gatto-ps-ai/source/target.jpg" -F "method=5" http://localhost:5000/api/colormatch -o result_cpu_basic.jpg
echo "Test 1 (Basic LAB Transfer CPU): Done"

# Test 2: LAB transfer with custom channels (only 'a')
curl -X POST -F "master_image=@/home/lukasz/projects/gatto-ps-ai/source/master.jpg" -F "target_image=@/home/lukasz/projects/gatto-ps-ai/source/target.jpg" -F "method=5" -F "channels=a" http://localhost:5000/api/colormatch -o result_cpu_channel_a.jpg
echo "Test 2 (Channel 'a' only CPU): Done"

# Test 3: LAB transfer with hybrid method forced to CPU
curl -X POST -F "master_image=@/home/lukasz/projects/gatto-ps-ai/source/master.jpg" -F "target_image=@/home/lukasz/projects/gatto-ps-ai/source/target.jpg" -F "method=5" -F "processing_method=cpu" http://localhost:5000/api/colormatch -o result_cpu_hybrid.jpg
echo "Test 3 (Hybrid forced CPU): Done"

echo "CPU tests for algorithm_05_lab_transfer completed. Check output files: result_cpu_*.jpg"
