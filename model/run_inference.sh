#!/bin/bash
# ============================================================================
# Inference Script for ERDES Medical Video Diagnosis
# ============================================================================
#
# This script runs inference on a video using trained models.
#
# USAGE:
#   bash run_inference.sh <video_path> [OPTIONS]
#
# EXAMPLES:
#   # Classifier-only inference
#   bash run_inference.sh /path/to/video.mp4
#
#   # Full pipeline with VLM
#   bash run_inference.sh /path/to/video.mp4 --with-vlm
#
#   # Custom output directory
#   bash run_inference.sh /path/to/video.mp4 --output-dir ./my_results
#
# ============================================================================

set -e  # Exit on error

# Default configuration
CLASSIFIER_CHECKPOINT="./checkpoints/multiclass/best_model_weights.pth"
VLM_CHECKPOINT="./checkpoints/vlm_finetuned/vlm_checkpoints/final_model"
OUTPUT_DIR="./inference_output"
USE_VLM=false

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Error: No video path provided"
    echo ""
    echo "Usage: bash run_inference.sh <video_path> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --with-vlm              Use VLM for clinical reasoning"
    echo "  --classifier PATH       Path to classifier checkpoint"
    echo "  --vlm PATH              Path to VLM checkpoint"
    echo "  --output-dir DIR        Output directory for results"
    echo ""
    echo "Examples:"
    echo "  bash run_inference.sh /path/to/video.mp4"
    echo "  bash run_inference.sh /path/to/video.mp4 --with-vlm"
    exit 1
fi

VIDEO_PATH="$1"
shift

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-vlm)
            USE_VLM=true
            shift
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
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
echo "ERDES Medical Video Diagnosis - Inference"
echo "=========================================="
echo ""
echo "Video: $VIDEO_PATH"
echo "Classifier: $CLASSIFIER_CHECKPOINT"

# Build command
CMD="python run_inference.py \
    --video_path \"$VIDEO_PATH\" \
    --classifier_checkpoint \"$CLASSIFIER_CHECKPOINT\" \
    --output_dir \"$OUTPUT_DIR\""

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
        echo "VLM: $VLM_CHECKPOINT"
        CMD="$CMD --vlm_checkpoint \"$VLM_CHECKPOINT\""
    fi
else
    echo "VLM: Not used (add --with-vlm to enable)"
fi

echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run inference
eval $CMD

echo ""
echo "=========================================="
echo "Inference Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
