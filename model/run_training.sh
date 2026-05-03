#!/bin/bash
# ============================================================================
# Complete Training Pipeline for ERDES Medical Video Diagnosis
# ============================================================================
#
# This script runs the full training pipeline:
#   1. Train multi-class video classifier (ExplainableOpticalFlowResNet3D)
#   2. Prepare VLM training data with ground-truth summaries from CSV
#   3. Finetune Qwen 2.5 VL with FAVG (Faithfulness-Aware Visual Grounding)
#
# USAGE:
#   bash run_training.sh [OPTIONS]
#
# OPTIONS:
#   --resume    Resume VLM training from last checkpoint (if interrupted)
#   --no-vlm    Skip VLM training (classifier only)
#
# EXAMPLES:
#   # Full training pipeline (classifier + VLM)
#   bash run_training.sh
#
#   # Resume interrupted VLM training
#   bash run_training.sh --resume
#
#   # Train classifier only
#   bash run_training.sh --no-vlm
#
# FEATURES:
#   ✓ Automatic checkpoint saving every 100 steps
#   ✓ Automatic resumption from last checkpoint
#   ✓ Ground-truth clinical summaries from CSV
#   ✓ Structured diagnosis_text output
#   ✓ FAVG contrastive learning for faithfulness
#   ✓ TensorBoard logging
#
# REQUIREMENTS:
#   - CSV file: ../benchmarks/input/balanced_split_desc.csv
#   - Video data: ../erdes/clips/
#   - GPU with 16GB+ VRAM (for VLM training)
#
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "ERDES Medical Video Diagnosis Training"
echo "=========================================="
echo ""

# Parse arguments
RESUME_VLM=false
SKIP_VLM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_VLM=true
            shift
            ;;
        --no-vlm)
            SKIP_VLM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_training.sh [--resume] [--no-vlm]"
            exit 1
            ;;
    esac
done

# Configuration
CSV_PATH="../benchmarks/input/balanced_split_desc.csv"
DATA_ROOT="../erdes"
MULTICLASS_OUTPUT="./checkpoints/multiclass_$(date +%Y%m%d_%H%M%S)"
VLM_OUTPUT="./checkpoints/vlm_finetuned"  # Fixed directory for checkpoint resumption

# Stage 1: Train Multi-Class Classifier
echo "=========================================="
echo "Stage 1: Training Multi-Class Classifier"
echo "=========================================="
echo "Output: $MULTICLASS_OUTPUT"
echo ""

python train_multiclass.py \
    --csv_path "$CSV_PATH" \
    --data_root "$DATA_ROOT" \
    --output_dir "$MULTICLASS_OUTPUT" \
    --test_size 0.2 \
    --random_state 42 \
    --num_diagnostic_classes 2 \
    --num_subtype_classes 4 \
    --pretrained \
    --dropout 0.3 \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --num_workers 4 \
    --num_frames 32 \
    --img_size 224

echo ""
echo "✓ Classifier training complete!"
echo "Best model: $MULTICLASS_OUTPUT/best_model_weights.pth"
echo ""

# Check if classifier training was successful
if [ ! -f "$MULTICLASS_OUTPUT/best_model_weights.pth" ]; then
    echo "Error: Classifier training failed - best_model_weights.pth not found"
    exit 1
fi

# Skip VLM training if requested
if [ "$SKIP_VLM" = true ]; then
    echo ""
    echo "=========================================="
    echo "Skipping VLM Training (--no-vlm flag)"
    echo "=========================================="
    echo "Classifier checkpoint: $MULTICLASS_OUTPUT/best_model_weights.pth"
    echo "=========================================="
    exit 0
fi

# Stage 2 & 3: Prepare VLM Data and Train VLM
echo "=========================================="
echo "Stage 2 & 3: VLM Data Preparation & Training"
echo "=========================================="
echo "Using classifier: $MULTICLASS_OUTPUT/best_model_weights.pth"
echo "CSV: $CSV_PATH"
echo "Output: $VLM_OUTPUT"

if [ "$RESUME_VLM" = true ]; then
    echo "Mode: RESUME from last checkpoint (if available)"
else
    echo "Mode: TRAIN from scratch"
fi
echo ""

# Build VLM training command using train_llm.py
# This script handles both data preparation and training
VLM_CMD="python train_llm.py \
    --classifier_checkpoint $MULTICLASS_OUTPUT/best_model_weights.pth \
    --csv_path $CSV_PATH \
    --data_root $DATA_ROOT \
    --output_dir $VLM_OUTPUT \
    --test_size 0.2 \
    --random_state 42 \
    --num_diagnostic_classes 2 \
    --num_subtype_classes 4 \
    --top_k_frames 5 \
    --use_contrastive \
    --num_frames 32 \
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
echo "Training Pipeline Complete!"
echo "=========================================="
echo ""
echo "📊 Classifier Results:"
echo "  - Checkpoint: $MULTICLASS_OUTPUT/best_model_weights.pth"
echo "  - Reports: $MULTICLASS_OUTPUT/best_*_report.txt"
echo "  - History: $MULTICLASS_OUTPUT/history.json"
echo ""
echo "🤖 VLM Results:"
echo "  - Final model: $VLM_OUTPUT/vlm_checkpoints/final_model/"
echo "  - Best model: $VLM_OUTPUT/vlm_checkpoints/best_model/"
echo "  - Training data: $VLM_OUTPUT/vlm_data/"
echo "  - Split cache: $VLM_OUTPUT/train_test_splits.json"
echo ""
echo "📝 Training Features:"
echo "  ✓ Automatic data caching (splits + VLM samples)"
echo "  ✓ Important frame extraction with attention"
echo "  ✓ Heatmap overlay generation"
echo "  ✓ FAVG contrastive learning enabled"
echo "  ✓ Checkpoint auto-resumption"
echo ""
echo "To resume interrupted training:"
echo "  bash run_training.sh --resume"
echo ""
echo "To force re-prepare VLM data:"
echo "  python train_llm.py --force_prepare ..."
echo "=========================================="
