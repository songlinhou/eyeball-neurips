#!/bin/bash
# ============================================================================
# Batch Inference Script for ERDES Test Set
# ============================================================================
#
# This script runs inference on the entire test set using trained models.
# It processes all videos in the test split and generates comprehensive results.
#
# USAGE:
#   bash run_test_set_inference.sh [OPTIONS]
#
# EXAMPLES:
#   # Classifier-only inference on test set
#   bash run_test_set_inference.sh
#
#   # Full pipeline with VLM
#   bash run_test_set_inference.sh --with-vlm
#
#   # Custom data and checkpoint paths
#   bash run_test_set_inference.sh --data-csv /path/to/test.csv --classifier /path/to/model.pth
#
#   # Parallel processing with 4 workers
#   bash run_test_set_inference.sh --num-workers 4
#
# ============================================================================

set -e  # Exit on error

# Default configuration
DATA_CSV="../benchmarks/input/balanced_split_desc_test.csv"
CLASSIFIER_CHECKPOINT="/content/drive/MyDrive/EyeballProject/multi_class_and_llm_manual_split/checkpoints/multiclass/best_model_weights.pth"
VLM_CHECKPOINT="/content/drive/MyDrive/EyeballProject/multi_class_and_llm_manual_split/checkpoints/vlm_finetuned/vlm_checkpoints/final_model"
OUTPUT_DIR="/content/drive/MyDrive/EyeballProject/multi_class_and_llm_manual_split/test_set_inference_results"
VIDEO_BASE_DIR="/content/eyeball-neurips/erdes"
USE_VLM=true
NUM_WORKERS=1
BATCH_SIZE=1
SAVE_ATTENTION=true
SAVE_VISUALIZATIONS=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-vlm)
            USE_VLM=true
            shift
            ;;
        --data-csv)
            DATA_CSV="$2"
            shift 2
            ;;
        --classifier)
            CLASSIFIER_CHECKPOINT="$2"
            shift 2
            ;;
        --vlm)
            VLM_CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --video-base-dir)
            VIDEO_BASE_DIR="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-attention)
            SAVE_ATTENTION=false
            shift
            ;;
        --no-visualizations)
            SAVE_VISUALIZATIONS=false
            shift
            ;;
        --help)
            echo "Usage: bash run_test_set_inference.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --with-vlm                Use VLM for clinical reasoning"
            echo "  --data-csv PATH           Path to test CSV file (default: ../benchmarks/input/balanced_split_desc_test.csv)"
            echo "  --classifier PATH         Path to classifier checkpoint"
            echo "  --vlm PATH                Path to VLM checkpoint"
            echo "  --output-dir DIR          Output directory for results (default: ./test_set_inference_results)"
            echo "  --video-base-dir DIR      Base directory for video files (if CSV paths are relative)"
            echo "  --num-workers N           Number of parallel workers (default: 1)"
            echo "  --batch-size N            Batch size for processing (default: 1)"
            echo "  --no-attention            Skip saving attention maps"
            echo "  --no-visualizations       Skip generating visualizations"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash run_test_set_inference.sh"
            echo "  bash run_test_set_inference.sh --with-vlm --num-workers 4"
            echo "  bash run_test_set_inference.sh --data-csv /path/to/test.csv --output-dir ./results"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if data CSV exists
if [ ! -f "$DATA_CSV" ]; then
    echo "Error: Data CSV not found: $DATA_CSV"
    echo ""
    echo "Expected location: $DATA_CSV"
    echo ""
    echo "Please provide the correct path using --data-csv option"
    exit 1
fi

# Check if classifier checkpoint exists
if [ ! -f "$CLASSIFIER_CHECKPOINT" ]; then
    echo "Error: Classifier checkpoint not found: $CLASSIFIER_CHECKPOINT"
    echo ""
    echo "Expected location: $CLASSIFIER_CHECKPOINT"
    echo ""
    echo "Did you run training? Use:"
    echo "  bash run_training.sh"
    exit 1
fi

echo "=========================================="
echo "ERDES Test Set Batch Inference"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Data CSV: $DATA_CSV"
echo "  Classifier: $CLASSIFIER_CHECKPOINT"

# Build command
CMD="python run_test_set_inference.py \
    --data_csv \"$DATA_CSV\" \
    --classifier_checkpoint \"$CLASSIFIER_CHECKPOINT\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE"

# Add video base dir if provided
if [ -n "$VIDEO_BASE_DIR" ]; then
    CMD="$CMD --video_base_dir \"$VIDEO_BASE_DIR\""
fi

# Add VLM if requested
if [ "$USE_VLM" = true ]; then
    if [ ! -d "$VLM_CHECKPOINT" ]; then
        echo ""
        echo "Warning: VLM checkpoint not found: $VLM_CHECKPOINT"
        echo "Running classifier-only inference"
        echo ""
        echo "To use VLM, first complete training with:"
        echo "  bash run_training.sh"
    else
        echo "  VLM: $VLM_CHECKPOINT"
        CMD="$CMD --vlm_checkpoint \"$VLM_CHECKPOINT\""
    fi
else
    echo "  VLM: Not used (add --with-vlm to enable)"
fi

# Add optional flags
if [ "$SAVE_ATTENTION" = false ]; then
    CMD="$CMD --no_save_attention"
fi

if [ "$SAVE_VISUALIZATIONS" = false ]; then
    CMD="$CMD --no_save_visualizations"
fi

echo "  Output: $OUTPUT_DIR"
echo "  Workers: $NUM_WORKERS"
echo "  Batch size: $BATCH_SIZE"
echo "  Save attention: $SAVE_ATTENTION"
echo "  Save visualizations: $SAVE_VISUALIZATIONS"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run batch inference
echo "Starting batch inference on test set..."
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Batch Inference Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary files:"
echo "  - ${OUTPUT_DIR}/test_set_predictions.csv"
echo "  - ${OUTPUT_DIR}/test_set_metrics.json"
echo "  - ${OUTPUT_DIR}/confusion_matrices/"
echo ""
