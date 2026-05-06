#!/bin/bash

# Script to run attention comparison analysis
# Usage: bash run_attention_analysis.sh

CHECKPOINT="/content/drive/MyDrive/EyeballProject/multi_class_and_llm_manual_split/checkpoints/multiclass/best_model_weights.pth"
DATA_DIR="/content/eyeball-neurips/erdes"
SAVE_DIR="attention_analysis_results"
NUM_SAMPLES=50  # Use -1 to analyze all records in the dataset

echo "Running attention comparison analysis..."
echo "Checkpoint: $CHECKPOINT"
echo "Data directory: $DATA_DIR"
echo "Save directory: $SAVE_DIR"
echo "Number of samples: $NUM_SAMPLES"
echo ""

python compare_attention_on_dataset.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --num_samples $NUM_SAMPLES \
    --device cuda

echo ""
echo "Analysis complete! Results saved to $SAVE_DIR"
