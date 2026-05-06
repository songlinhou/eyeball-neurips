#!/bin/bash
# ============================================================================
# Train VLM with Uniformly Sampled Frames (No Classifier, No Heatmaps)
# ============================================================================
#
# This script trains Qwen 2.5 VL using uniformly sampled frames,
# without classifier-based frame selection or heatmap overlays.
#
# USAGE:
#   bash run_training_uniform.sh [OPTIONS]
#
# OPTIONS:
#   --resume              Resume VLM training from last checkpoint (if interrupted)
#   --vlm-dir DIR         Custom output directory for VLM (default: ./checkpoints/vlm_uniform)
#   --num-frames N        Number of frames to uniformly sample (default: 5)
#
# EXAMPLES:
#   # Train with default settings (5 uniformly sampled frames)
#   bash run_training_uniform.sh
#
#   # Resume interrupted training
#   bash run_training_uniform.sh --resume
#
#   # Use 8 frames instead of 5
#   bash run_training_uniform.sh --num-frames 8
#
# FEATURES:
#   ✓ Uniform frame sampling (no classifier needed)
#   ✓ Original frames only (no heatmap overlays)
#   ✓ No contrastive learning
#   ✓ Automatic checkpoint saving every 100 steps
#   ✓ TensorBoard logging
#
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "VLM Training - Uniform Sampling"
echo "=========================================="
echo ""

# Parse arguments
RESUME_VLM=false
CUSTOM_VLM_DIR=""
NUM_FRAMES=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_VLM=true
            shift
            ;;
        --vlm-dir)
            CUSTOM_VLM_DIR="$2"
            shift 2
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_training_uniform.sh [--resume] [--vlm-dir DIR] [--num-frames N]"
            exit 1
            ;;
    esac
done

# Configuration
TRAIN_CSV="../benchmarks/input/balanced_split_desc_train.csv"
TEST_CSV="../benchmarks/input/balanced_split_desc_test.csv"
DATA_ROOT="../erdes"

# Use custom directory if provided, otherwise use default
if [ -n "$CUSTOM_VLM_DIR" ]; then
    VLM_OUTPUT="$CUSTOM_VLM_DIR"
else
    VLM_OUTPUT="./checkpoints/vlm_uniform"
fi

echo "=========================================="
echo "VLM Training with Uniform Sampling"
echo "=========================================="
echo "Train CSV: $TRAIN_CSV"
echo "Test CSV: $TEST_CSV"
echo "Output: $VLM_OUTPUT"
echo "Frames per video: $NUM_FRAMES"

if [ "$RESUME_VLM" = true ]; then
    echo "Mode: RESUME from last checkpoint (if available)"
else
    echo "Mode: TRAIN from scratch"
fi
echo ""

# Build VLM training command
VLM_CMD="python train_llm_uniform.py \
    --train_csv $TRAIN_CSV \
    --test_csv $TEST_CSV \
    --data_root $DATA_ROOT \
    --output_dir $VLM_OUTPUT \
    --num_frames $NUM_FRAMES \
    --img_size 224 \
    --vlm_model Qwen/Qwen2-VL-7B-Instruct \
    --use_4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --vlm_epochs 10 \
    --vlm_batch_size 2 \
    --vlm_lr 2e-5 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100"

# Add skip_data_preparation flag if resuming (to use cached data)
if [ "$RESUME_VLM" = true ]; then
    VLM_CMD="$VLM_CMD --skip_data_preparation"
fi

# Execute VLM training
eval $VLM_CMD

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "🤖 VLM Results:"
echo "  - Final model: $VLM_OUTPUT/vlm_checkpoints/final_model/"
echo "  - Best model: $VLM_OUTPUT/vlm_checkpoints/best_model/"
echo "  - Training data: $VLM_OUTPUT/vlm_data/"
echo ""
echo "📝 Training Configuration:"
echo "  ✓ Uniform frame sampling ($NUM_FRAMES frames per video)"
echo "  ✓ No classifier-based frame selection"
echo "  ✓ No heatmap overlays"
echo "  ✓ No contrastive learning"
echo "  ✓ Checkpoint auto-resumption"
echo ""
echo "To resume interrupted training:"
echo "  bash run_training_uniform.sh --resume"
echo ""
echo "To force re-prepare VLM data:"
echo "  python train_llm_uniform.py --force_prepare ..."
echo "=========================================="
