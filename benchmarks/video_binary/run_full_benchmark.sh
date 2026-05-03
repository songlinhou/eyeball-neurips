#!/bin/bash

# Full benchmark script to train and compare all video classification models
# This will take several hours to complete depending on hardware
#
# Usage: ./run_full_benchmark.sh [output_directory]
#   output_directory: Optional path where results will be saved 
#                     (default: ./results_YYYYMMDD_HHMMSS)

echo "=========================================="
echo "Full Video Classification Benchmark"
echo "=========================================="
echo ""
echo "This will train 8 different models:"
echo "  - ResNet3D (baseline)"
echo "  - I3D (Inflated 3D ConvNet)"
echo "  - SlowFast (dual-pathway)"
echo "  - X3D (efficient)"
echo "  - MViT (vision transformer)"
echo "  - VideoMAE (masked autoencoder)"
echo "  - TimeSformer (space-time attention)"
echo "  - C3D (classic 3D CNN)"
echo ""
echo "Estimated time: 4-8 hours (depending on GPU)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Set output directory from parameter or create timestamped default
if [ -n "$1" ]; then
    RESULTS_DIR="$1"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RESULTS_DIR="./results_${TIMESTAMP}"
fi

echo ""
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Run full benchmark with all models
python train_benchmark.py \
    --data_dir ../../erdes \
    --split macula_detached_vs_intact \
    --save_dir "${RESULTS_DIR}" \
    --num_frames 32 \
    --img_size 224 \
    --batch_size 8 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --num_workers 4 \
    --pretrained

echo ""
echo "=========================================="
echo "Training complete! Generating reports..."
echo "=========================================="
echo ""

# Generate comprehensive comparison
python compare_results.py \
    --results_dir "${RESULTS_DIR}" \
    --results_file benchmark_results.json

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Key files:"
echo "  - ${RESULTS_DIR}/BENCHMARK_REPORT.md"
echo "  - ${RESULTS_DIR}/comparison_table.csv"
echo "  - ${RESULTS_DIR}/metrics_comparison.png"
echo "  - ${RESULTS_DIR}/efficiency_analysis.png"
echo ""
echo "View the full report:"
echo "  cat ${RESULTS_DIR}/BENCHMARK_REPORT.md"
echo ""
