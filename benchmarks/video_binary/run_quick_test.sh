#!/bin/bash

# Quick test script to verify the benchmark setup
# Runs a short training session with reduced epochs for testing
#
# Usage: ./run_quick_test.sh [output_directory]
#   output_directory: Optional path where results will be saved (default: ./test_results)

# Set output directory from parameter or use default
OUTPUT_DIR="${1:-./test_results}"

echo "=================================="
echo "Video Classification Quick Test"
echo "=================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Test with just 2 models and reduced epochs
python train_benchmark.py \
    --data_dir ../../erdes \
    --split macula_detached_vs_intact \
    --save_dir "${OUTPUT_DIR}" \
    --models resnet3d i3d \
    --num_frames 16 \
    --img_size 224 \
    --batch_size 4 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --num_workers 2

echo ""
echo "=================================="
echo "Generating comparison report..."
echo "=================================="
echo ""

# Generate comparison visualizations
python compare_results.py \
    --results_dir "${OUTPUT_DIR}" \
    --results_file benchmark_results.json

echo ""
echo "=================================="
echo "Quick test complete!"
echo "Check ${OUTPUT_DIR}/ for outputs"
echo "=================================="
