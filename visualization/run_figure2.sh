#!/bin/bash
#
# Generate Figure 2: Attention Visualization Examples
#
# Usage:
#   bash run_figure2.sh
#

set -e

# Configuration
MODEL_CHECKPOINT="/content/drive/MyDrive/EyeballProject/multi_class_and_llm_manual_split/checkpoints/multiclass/best_model_weights.pth"
DATA_CSV="../benchmarks/input/balanced_split_desc.csv"
VIDEO_BASE_DIR="/content/eyeball-neurips/erdes"
OUTPUT_DIR="./figures"
NUM_EXAMPLES=6
DEVICE="cuda"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Generating Figure 2"
echo "=========================================="
echo "Model: $MODEL_CHECKPOINT"
echo "Data: $DATA_CSV"
echo "Output: $OUTPUT_DIR"
echo "Examples: $NUM_EXAMPLES"
echo "=========================================="
echo ""

# Run the script
python generate_figure2_attention.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --data_csv "$DATA_CSV" \
    --video_base_dir "$VIDEO_BASE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_examples "$NUM_EXAMPLES" \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "Done! Check $OUTPUT_DIR for outputs"
echo "=========================================="
