# ERDES Inference Guide

This guide explains how to run inference on videos using your trained models.

## Quick Start

### Option 1: Using the Shell Script (Easiest)

```bash
# Classifier-only inference (predictions + attention maps)
bash run_inference.sh /path/to/your/video.mp4

# Full pipeline with VLM (predictions + clinical reasoning)
bash run_inference.sh /path/to/your/video.mp4 --with-vlm

# Custom output directory
bash run_inference.sh /path/to/your/video.mp4 --output-dir ./my_results
```

### Option 2: Using Python Directly

```bash
# Classifier-only inference
python run_inference.py \
    --video_path /path/to/your/video.mp4 \
    --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth

# Full pipeline with VLM
python run_inference.py \
    --video_path /path/to/your/video.mp4 \
    --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
    --vlm_checkpoint ./checkpoints/vlm_finetuned/vlm_checkpoints/final_model
```

## What You Get

After running inference, you'll get:

1. **JSON Results** (`{video_name}_results.json`):
   - Diagnostic prediction (RD vs Non-RD)
   - Subtype prediction (Normal, PVD, Macula Intact, Macula Detached)
   - Confidence scores for all predictions
   - Frame importance scores
   - Clinical reasoning (if VLM is used)

2. **Frame Importance Plot** (`{video_name}_frame_importance.png`):
   - Bar chart showing which frames the model focused on
   - Helps understand temporal attention

3. **Extracted Frames & Heatmaps** (in temp directory, if VLM is used):
   - Top-K most important frames
   - Attention heatmaps overlaid on frames

## Example Output

```json
{
  "video_path": "/path/to/video.mp4",
  "video_name": "video",
  "predictions": {
    "diagnostic": "RD",
    "diagnostic_class": 1,
    "diagnostic_confidence": 0.95,
    "diagnostic_probs": [0.05, 0.95],
    "subtype": "Macula Detached",
    "subtype_class": 3,
    "subtype_confidence": 0.87,
    "subtype_probs": [0.02, 0.05, 0.06, 0.87]
  },
  "frame_importance": [0.02, 0.15, 0.08, ..., 0.22],
  "clinical_reasoning": "Based on the highlighted regions showing..."
}
```

## Model Checkpoints

After running `bash run_training.sh`, your models are saved at:

- **Classifier**: `./checkpoints/multiclass/best_model_weights.pth`
- **VLM (final)**: `./checkpoints/vlm_finetuned/vlm_checkpoints/final_model/`
- **VLM (best)**: `./checkpoints/vlm_finetuned/vlm_checkpoints/best_model/`

## Advanced Usage

### Custom Model Configuration

```bash
python run_inference.py \
    --video_path /path/to/video.mp4 \
    --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
    --num_frames 64 \
    --img_size 256 \
    --top_k_frames 8 \
    --device cuda
```

### Parameters

- `--video_path`: Path to input video file (required)
- `--classifier_checkpoint`: Path to trained classifier (required)
- `--vlm_checkpoint`: Path to finetuned VLM (optional)
- `--output_dir`: Output directory (default: `./inference_output`)
- `--num_frames`: Number of frames to sample (default: 32)
- `--img_size`: Image size for preprocessing (default: 224)
- `--top_k_frames`: Number of important frames for VLM (default: 5)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)

### Batch Inference

To run inference on multiple videos:

```bash
#!/bin/bash
for video in /path/to/videos/*.mp4; do
    echo "Processing: $video"
    bash run_inference.sh "$video" --with-vlm
done
```

## Pipeline Modes

### 1. Classifier-Only Mode

**What it does:**
- Loads video and samples frames
- Runs multi-class classification
- Extracts frame importance scores
- Generates attention maps

**Use when:**
- You only need predictions
- VLM is not trained yet
- Fast inference is needed

**Command:**
```bash
bash run_inference.sh /path/to/video.mp4
```

### 2. Full Pipeline Mode (Classifier + VLM)

**What it does:**
- Everything from classifier-only mode
- Extracts top-K important frames
- Generates attention heatmaps
- Runs VLM to generate clinical reasoning

**Use when:**
- You need explainable predictions
- Clinical reasoning is required
- VLM has been trained

**Command:**
```bash
bash run_inference.sh /path/to/video.mp4 --with-vlm
```

## Troubleshooting

### Error: Classifier checkpoint not found

Make sure you've run training first:
```bash
bash run_training.sh
```

The classifier will be saved to `./checkpoints/multiclass/best_model_weights.pth`

### Error: VLM checkpoint not found

If you want to use VLM, make sure training completed successfully. The VLM checkpoint should be at:
```
./checkpoints/vlm_finetuned/vlm_checkpoints/final_model/
```

If training was interrupted, you can resume with:
```bash
bash run_training.sh --resume
```

### CUDA Out of Memory

Try these solutions:

1. Use CPU instead:
   ```bash
   python run_inference.py --video_path /path/to/video.mp4 \
       --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
       --device cpu
   ```

2. Reduce number of frames:
   ```bash
   python run_inference.py --video_path /path/to/video.mp4 \
       --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
       --num_frames 16
   ```

3. For VLM, use the 4-bit quantized version during training

### Video Format Issues

The script supports most common video formats (mp4, avi, mov, etc.). If you encounter issues:

1. Check if OpenCV can read your video:
   ```python
   import cv2
   cap = cv2.VideoCapture('/path/to/video.mp4')
   print(cap.isOpened())
   ```

2. Convert to a standard format:
   ```bash
   ffmpeg -i input.avi -c:v libx264 -crf 23 output.mp4
   ```

## Integration with Your Workflow

### Python API

You can also use the inference functions directly in your Python code:

```python
from run_inference import load_video, run_classifier_inference
from multiclass_model import create_multiclass_model
import torch

# Load model
model = create_multiclass_model(
    num_diagnostic_classes=2,
    num_subtype_classes=4,
    pretrained=False
)
checkpoint = torch.load('./checkpoints/multiclass/best_model_weights.pth')
model.load_state_dict(checkpoint)
model.eval()

# Load and process video
video_tensor = load_video('/path/to/video.mp4')

# Run inference
predictions, attention = run_classifier_inference(model, video_tensor)

print(f"Diagnostic: {predictions['diagnostic']} ({predictions['diagnostic_confidence']:.1%})")
print(f"Subtype: {predictions['subtype']} ({predictions['subtype_confidence']:.1%})")
```

### Using VLM Pipeline

```python
from vlm_pipeline import VLMDiagnosisPipeline

# Initialize pipeline
pipeline = VLMDiagnosisPipeline(
    classifier_checkpoint='./checkpoints/multiclass/best_model_weights.pth',
    num_diagnostic_classes=2,
    num_subtype_classes=4
)

# Setup VLM
pipeline.setup_vlm(
    model_name='./checkpoints/vlm_finetuned/vlm_checkpoints/final_model'
)

# Run diagnosis
diagnosis = pipeline.diagnose_video(
    video_tensor=video_tensor,
    video_id='my_video'
)

print(diagnosis['clinical_reasoning'])
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster inference
2. **Batch Processing**: Process multiple videos in parallel
3. **Frame Sampling**: Reduce `--num_frames` for faster processing
4. **Model Quantization**: Use 4-bit VLM for lower memory usage

## Next Steps

- See `VLM_README.md` for details on the VLM architecture
- See `TRAINING_GUIDE.md` for training details
- See `vlm_example.py` for more code examples
- See `VISUALIZATION_GUIDE.md` for attention visualization
