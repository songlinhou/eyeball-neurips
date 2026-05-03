#!/bin/bash

# Full benchmark script for multi-class video classification
# Trains all models with full configuration

set -e

echo "================================================"
echo "Multi-Class Video Benchmark - Full Evaluation"
echo "================================================"

# Step 1: Prepare splits (if not already done)
if [ ! -d "./splits" ]; then
    echo ""
    echo "Step 1: Preparing train/test splits..."
    python prepare_splits.py \
        --metadata ../../erdes/metadata.csv \
        --output_dir ./splits \
        --train_size 300 \
        --test_size 100 \
        --seed 42
else
    echo ""
    echo "Step 1: Splits already exist, skipping..."
fi

# Step 2: Run full benchmark
echo ""
echo "Step 2: Running full benchmark (all models, 10 epochs)..."
echo ""
echo "Models to benchmark:"
echo "  - ResNet3D (baseline)"
echo "  - I3D (Inflated 3D ConvNet)"
echo "  - SlowFast (dual-pathway)"
echo "  - X3D (efficient)"
echo "  - MViT (vision transformer)"
echo "  - VideoMAE (masked autoencoder)"
echo "  - TimeSformer (space-time attention)"
echo "  - C3D (classic 3D CNN)"
echo "  - Explainable (PROPOSED - with attention & optical flow)"
echo ""

python train_benchmark.py \
    --data_dir ../../erdes \
    --splits_dir ./splits \
    --save_dir ./results \
    --models explainable resnet3d i3d slowfast x3d mvit videomae timesformer c3d \
    --num_epochs 10 \
    --batch_size 16 \
    --num_frames 32 \
    --img_size 224 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --num_workers 4

echo ""
echo "================================================"
echo "Full benchmark completed!"
echo "Results saved to: ./results"
echo "================================================"
echo ""
echo "To view results:"
echo "  - Training logs: ./results/logs/"
echo "  - Model weights: ./results/models/"
echo "  - Plots: ./results/plots/"
echo "  - Summary: ./results/benchmark_results.json"
