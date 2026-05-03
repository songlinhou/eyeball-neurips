#!/bin/bash
# Complete training pipeline for ERDES medical video diagnosis
# Usage: bash run_training.sh

set -e  # Exit on error

echo "=========================================="
echo "ERDES Medical Video Diagnosis Training"
echo "=========================================="
echo ""

# Configuration
CSV_PATH="../benchmarks/input/balanced_split_desc.csv"
DATA_ROOT="../erdes"
MULTICLASS_OUTPUT="./checkpoints/multiclass_$(date +%Y%m%d_%H%M%S)"
VLM_OUTPUT="./checkpoints/vlm_$(date +%Y%m%d_%H%M%S)"

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
    --frame_size 224

echo ""
echo "✓ Classifier training complete!"
echo "Best model: $MULTICLASS_OUTPUT/best_model_weights.pth"
echo ""

# Check if classifier training was successful
if [ ! -f "$MULTICLASS_OUTPUT/best_model_weights.pth" ]; then
    echo "Error: Classifier training failed - best_model_weights.pth not found"
    exit 1
fi

# Stage 2: Train VLM
echo "=========================================="
echo "Stage 2: Training VLM"
echo "=========================================="
echo "Using classifier: $MULTICLASS_OUTPUT/best_model_weights.pth"
echo "Output: $VLM_OUTPUT"
echo ""

python train_llm.py \
    --classifier_checkpoint "$MULTICLASS_OUTPUT/best_model_weights.pth" \
    --csv_path "$CSV_PATH" \
    --data_root "$DATA_ROOT" \
    --output_dir "$VLM_OUTPUT" \
    --test_size 0.2 \
    --random_state 42 \
    --num_diagnostic_classes 2 \
    --num_subtype_classes 4 \
    --top_k_frames 5 \
    --use_contrastive \
    --num_frames 32 \
    --frame_size 224 \
    --vlm_model "Qwen/Qwen2-VL-7B-Instruct" \
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
    --save_steps 100

echo ""
echo "=========================================="
echo "Training Pipeline Complete!"
echo "=========================================="
echo "Classifier checkpoint: $MULTICLASS_OUTPUT/best_model_weights.pth"
echo "VLM checkpoint: $VLM_OUTPUT/vlm_checkpoints/best_model"
echo ""
echo "To evaluate:"
echo "  - Classifier reports: $MULTICLASS_OUTPUT/best_*_report.txt"
echo "  - Training history: $MULTICLASS_OUTPUT/history.json"
echo "  - VLM checkpoints: $VLM_OUTPUT/vlm_checkpoints/"
echo "=========================================="
