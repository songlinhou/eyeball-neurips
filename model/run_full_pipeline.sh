#!/bin/bash
# ============================================================================
# Complete Training and Evaluation Pipeline for ERDES Medical Video Diagnosis
# ============================================================================
#
# This script runs the full pipeline:
#   1. Train multi-class video classifier (ExplainableOpticalFlowResNet3D)
#   2. Prepare VLM training data with ground-truth summaries from CSV
#   3. Finetune Qwen 2.5 VL with FAVG (Faithfulness-Aware Visual Grounding)
#   4. Run test set inference and evaluation
#
# USAGE:
#   bash run_full_pipeline.sh [OPTIONS]
#
# OPTIONS:
#   --resume          Resume VLM training from last checkpoint (if interrupted)
#   --no-vlm          Skip VLM training (classifier only)
#   --no-heatmap      Use original frames instead of heatmap overlays (disables contrastive learning)
#   --no-vlm-eval     Skip VLM for test set evaluation (default: use VLM)
#   --skip-training   Skip training, only run evaluation
#   --num-workers N   Number of parallel workers for evaluation (default: 1)
#
# EXAMPLES:
#   # Full pipeline (train + evaluate with VLM)
#   bash run_full_pipeline.sh
#
#   # Train and evaluate without VLM
#   bash run_full_pipeline.sh --no-vlm --no-vlm-eval
#
#   # Resume VLM training and evaluate
#   bash run_full_pipeline.sh --resume
#
#   # Skip training, only evaluate with VLM
#   bash run_full_pipeline.sh --skip-training
#
#   # Parallel evaluation with 4 workers
#   bash run_full_pipeline.sh --skip-training --num-workers 4
#
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "ERDES Full Training & Evaluation Pipeline"
echo "=========================================="
echo ""

# Parse arguments
RESUME_VLM=true
SKIP_VLM=false
NO_HEATMAP=false
USE_VLM_EVAL=true
SKIP_TRAINING=false
NUM_WORKERS=1

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
        --no-heatmap)
            NO_HEATMAP=true
            shift
            ;;
        --no-vlm-eval)
            USE_VLM_EVAL=false
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_full_pipeline.sh [--resume] [--no-vlm] [--no-heatmap] [--no-vlm-eval] [--skip-training] [--num-workers N]"
            exit 1
            ;;
    esac
done

# Configuration
TRAIN_CSV="../benchmarks/input/balanced_split_desc_train.csv"
TEST_CSV="../benchmarks/input/balanced_split_desc_test.csv"
DATA_ROOT="../erdes"
MULTICLASS_CHECKPOINT="/content/drive/MyDrive/EyeballProject/multi_class_and_llm_manual_split/checkpoints/multiclass/best_model_weights.pth"
VLM_CHECKPOINT="/content/drive/MyDrive/EyeballProject/multi_class_and_llm_manual_split/checkpoints/vlm_finetuned_base/vlm_checkpoints/final_model"
EVAL_OUTPUT_DIR="/content/drive/MyDrive/EyeballProject/multi_class_and_llm_manual_split/test_set_inference_results_base"

# Extract base directories for training
CLASSIFIER_DIR=$(dirname "$MULTICLASS_CHECKPOINT")
VLM_DIR=$(dirname $(dirname "$VLM_CHECKPOINT"))

# ============================================================================
# STAGE 1-3: TRAINING
# ============================================================================

if [ "$SKIP_TRAINING" = false ]; then
    echo "=========================================="
    echo "PHASE 1: TRAINING"
    echo "=========================================="
    echo ""
    
    # Build training command
    TRAIN_CMD="bash run_training.sh --classifier-dir \"$CLASSIFIER_DIR\" --vlm-dir \"$VLM_DIR\""
    
    if [ "$RESUME_VLM" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --resume"
    fi
    
    if [ "$SKIP_VLM" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --no-vlm"
    fi
    
    if [ "$NO_HEATMAP" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --no-heatmap"
    fi
    
    echo "Running: $TRAIN_CMD"
    echo ""
    
    # Execute training
    eval $TRAIN_CMD
    
    echo ""
    echo "=========================================="
    echo "Training Phase Complete!"
    echo "=========================================="
    echo ""
else
    echo "=========================================="
    echo "Skipping Training Phase (--skip-training)"
    echo "=========================================="
    echo ""
fi

# ============================================================================
# STAGE 4: TEST SET EVALUATION
# ============================================================================

echo "=========================================="
echo "PHASE 2: TEST SET EVALUATION"
echo "=========================================="
echo ""

# Check if classifier checkpoint exists
if [ ! -f "$MULTICLASS_CHECKPOINT" ]; then
    echo "Error: Classifier checkpoint not found: $MULTICLASS_CHECKPOINT"
    echo ""
    echo "Please run training first or check the checkpoint path"
    exit 1
fi

# Build evaluation command
EVAL_CMD="bash run_test_set_inference.sh \
    --data-csv \"$TEST_CSV\" \
    --classifier \"$MULTICLASS_CHECKPOINT\" \
    --output-dir \"$EVAL_OUTPUT_DIR\" \
    --num-workers $NUM_WORKERS"

# Add VLM if requested
if [ "$USE_VLM_EVAL" = true ]; then
    if [ ! -d "$VLM_CHECKPOINT" ]; then
        echo "Warning: VLM checkpoint not found: $VLM_CHECKPOINT"
        echo "Running classifier-only evaluation"
        echo ""
    else
        EVAL_CMD="$EVAL_CMD --with-vlm --vlm \"$VLM_CHECKPOINT\""
    fi
fi

echo "Running test set evaluation..."
echo ""

# Execute evaluation
eval $EVAL_CMD

echo ""
echo "=========================================="
echo "Evaluation Phase Complete!"
echo "=========================================="
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "=========================================="
echo "FULL PIPELINE COMPLETE!"
echo "=========================================="
echo ""

if [ "$SKIP_TRAINING" = false ]; then
    echo "📊 Training Results:"
    echo "  - Classifier: $MULTICLASS_CHECKPOINT"
    if [ "$SKIP_VLM" = false ]; then
        echo "  - VLM Model: $VLM_CHECKPOINT"
    fi
    echo ""
fi

echo "📈 Evaluation Results:"
echo "  - Output Directory: $EVAL_OUTPUT_DIR"
echo "  - Predictions: ${EVAL_OUTPUT_DIR}/test_set_predictions.csv"
echo "  - Metrics: ${EVAL_OUTPUT_DIR}/test_set_metrics.json"
echo "  - Confusion Matrices: ${EVAL_OUTPUT_DIR}/confusion_matrices/"
echo ""

echo "📝 Pipeline Summary:"
if [ "$SKIP_TRAINING" = false ]; then
    echo "  ✓ Training completed"
fi
echo "  ✓ Test set evaluation completed"
if [ "$USE_VLM_EVAL" = true ] && [ -d "$VLM_CHECKPOINT" ]; then
    echo "  ✓ VLM-based clinical reasoning included"
fi
echo ""

echo "To view detailed results:"
echo "  cat ${EVAL_OUTPUT_DIR}/test_set_metrics.json"
echo ""
echo "To re-run evaluation only:"
echo "  bash run_full_pipeline.sh --skip-training"
echo ""
echo "=========================================="
