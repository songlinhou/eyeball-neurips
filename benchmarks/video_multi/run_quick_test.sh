#!/bin/bash

# Quick test script for multi-class video classification benchmark
# Runs a fast test with reduced epochs and single model

set -e

echo "=========================================="
echo "Multi-Class Video Benchmark - Quick Test"
echo "=========================================="

# Step 1: Prepare splits
echo ""
echo "Step 1: Preparing train/test splits..."
python prepare_splits.py \
    --metadata ../../erdes/metadata.csv \
    --output_dir ./splits \
    --train_size 300 \
    --test_size 100 \
    --seed 42

# Step 2: Run quick benchmark (single model, few epochs)
echo ""
echo "Step 2: Running quick benchmark (ResNet3D, 5 epochs)..."
python train_benchmark.py \
    --data_dir ../../erdes \
    --splits_dir ./splits \
    --save_dir ./results_quick_test \
    --models resnet3d \
    --num_epochs 5 \
    --batch_size 8 \
    --num_frames 16 \
    --num_workers 4

echo ""
echo "=========================================="
echo "Quick test completed!"
echo "Results saved to: ./results_quick_test"
echo "=========================================="
