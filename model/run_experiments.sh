#!/bin/bash

# Ocular Ultrasound Classifier - Experiment Runner
# This script runs all experiments with proper environment setup

echo "=========================================="
echo "Ocular Ultrasound Classifier Experiments"
echo "=========================================="
echo ""

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null
then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Create necessary directories
SAVE_DIR="/content/drive/MyDrive/EyeballProject/classifier_experiment"
mkdir -p "$SAVE_DIR"
mkdir -p "$SAVE_DIR/models"
mkdir -p "$SAVE_DIR/checkpoints"
mkdir -p "$SAVE_DIR/logs"
mkdir -p "$SAVE_DIR/plots"
mkdir -p "$SAVE_DIR/results"
mkdir -p "$SAVE_DIR/explainability"

echo "Save directory: $SAVE_DIR"
echo ""

# Check if Google Drive is mounted (for Colab)
if [ -d "/content/drive" ]; then
    echo "✓ Google Drive detected"
else
    echo "⚠ Google Drive not detected - results will be saved locally"
fi
echo ""

# Run experiments
echo "Starting experiments..."
echo "This will take several hours. Progress will be logged."
echo ""

cd /content/improved_classifier

# Run all experiments
python run_experiments.py 2>&1 | tee "$SAVE_DIR/experiment_run.log"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiments completed successfully!"
    echo ""
    echo "Results saved to: $SAVE_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Check EXPERIMENT_REPORT.md for summary"
    echo "2. Review experiment_summary.csv for metrics"
    echo "3. Run visualize_explainability.py for interpretability"
else
    echo "✗ Experiments failed with exit code: $EXIT_CODE"
    echo "Check the log file: $SAVE_DIR/experiment_run.log"
fi
echo "=========================================="
