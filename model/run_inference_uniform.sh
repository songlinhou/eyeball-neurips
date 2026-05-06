#!/bin/bash
# ============================================================================
# Run Inference with Uniformly-Trained VLM Model
# ============================================================================
#
# This script runs inference on the test set using a VLM model trained with
# uniformly sampled frames (no classifier, no heatmaps).
#
# USAGE:
#   bash run_inference_uniform.sh [OPTIONS]
#
# OPTIONS:
#   --vlm-checkpoint PATH    Path to VLM checkpoint (required)
#   --num-frames N          Number of frames to sample (default: 5, should match training)
#   --output-dir DIR        Output directory (default: ./results/uniform_inference)
#
# EXAMPLES:
#   # Basic usage (use best model)
#   bash run_inference_uniform.sh \
#     --vlm-checkpoint ./checkpoints/vlm_uniform/vlm_checkpoints/best_model
#
#   # Use final model instead of best
#   bash run_inference_uniform.sh \
#     --vlm-checkpoint ./checkpoints/vlm_uniform/vlm_checkpoints/final_model
#
#   # Custom number of frames (must match training)
#   bash run_inference_uniform.sh \
#     --vlm-checkpoint ./checkpoints/vlm_uniform/vlm_checkpoints/best_model \
#     --num-frames 8
#
# OUTPUT:
#   - uniform_predictions.csv  (for GPT-as-judge comparison)
#   - uniform_predictions.json (detailed results)
#   - inference_summary.json   (statistics)
#
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "VLM Inference - Uniform Sampling"
echo "=========================================="
echo ""

# Default values
VLM_CHECKPOINT=""
NUM_FRAMES=5
OUTPUT_DIR="./results/uniform_inference"
TEST_CSV="../benchmarks/input/balanced_split_desc_test.csv"
DATA_ROOT="../erdes"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vlm-checkpoint)
            VLM_CHECKPOINT="$2"
            shift 2
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --test-csv)
            TEST_CSV="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_inference_uniform.sh --vlm-checkpoint PATH [--num-frames N] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$VLM_CHECKPOINT" ]; then
    echo "Error: --vlm-checkpoint is required"
    echo ""
    echo "Usage: bash run_inference_uniform.sh --vlm-checkpoint PATH"
    echo ""
    echo "Example:"
    echo "  bash run_inference_uniform.sh \\"
    echo "    --vlm-checkpoint ./checkpoints/vlm_uniform/vlm_checkpoints/best_model"
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "$VLM_CHECKPOINT" ]; then
    echo "Error: VLM checkpoint directory not found: $VLM_CHECKPOINT"
    exit 1
fi

echo "Configuration:"
echo "  VLM Checkpoint: $VLM_CHECKPOINT"
echo "  Test CSV: $TEST_CSV"
echo "  Data Root: $DATA_ROOT"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Frames per video: $NUM_FRAMES"
echo ""

# Run inference
python run_inference_uniform.py \
    --vlm_checkpoint "$VLM_CHECKPOINT" \
    --test_csv "$TEST_CSV" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_frames "$NUM_FRAMES" \
    --img_size 224 \
    --max_new_tokens 512

echo ""
echo "=========================================="
echo "Inference Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo ""
echo "1. Compare with other models using GPT-as-judge:"
echo "   cd ../benchmarks"
echo "   python llm_as_judge.py \\"
echo "     --predictions-1 ../model/$OUTPUT_DIR/uniform_predictions.csv \\"
echo "     --predictions-2 output/FAVG_pred.csv \\"
echo "     --model-1-name 'Uniform Sampling' \\"
echo "     --model-2-name 'FAVG Model'"
echo ""
echo "2. Or compare with Base model:"
echo "   python llm_as_judge.py \\"
echo "     --predictions-1 ../model/$OUTPUT_DIR/uniform_predictions.csv \\"
echo "     --predictions-2 output/base_pred.csv \\"
echo "     --model-1-name 'Uniform Sampling' \\"
echo "     --model-2-name 'Base Model'"
echo ""
echo "=========================================="
