# Training Guide

Complete guide for training the multi-class classifier and VLM on ERDES dataset.

## Overview

The training pipeline consists of two stages:

1. **Stage 1**: Train multi-class classifier (diagnostic + subtype)
2. **Stage 2**: Train VLM using the trained classifier

## Prerequisites

```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install transformers peft accelerate bitsandbytes
pip install opencv-python pandas scikit-learn tqdm
pip install qwen-vl-utils  # For Qwen 2.5 VL
```

## Stage 1: Train Multi-Class Classifier

### Quick Start

```bash
cd /home/ray/research/eyeball-llm/eyeball-neurips/model

# Train with default settings
python train_multiclass.py \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/multiclass \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4
```

### Key Arguments

**Data**:
- `--csv_path`: Path to balanced_split_desc.csv
- `--data_root`: Root directory containing video clips
- `--test_size`: Fraction for test set (default: 0.2)
- `--random_state`: Random seed for reproducibility (default: 42)

**Model**:
- `--num_diagnostic_classes`: Number of diagnostic classes (default: 2)
- `--num_subtype_classes`: Number of subtype classes (default: 4)
- `--pretrained`: Use pretrained R3D-18 weights (default: True)
- `--dropout`: Dropout rate (default: 0.3)

**Training**:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-5)

**Video Processing**:
- `--num_frames`: Number of frames to sample (default: 32)
- `--frame_size`: Frame size (default: 224)

### Output

The script creates:
```
checkpoints/multiclass/
├── config.json                    # Training configuration
├── best_model.pth                 # Best checkpoint (full)
├── best_model_weights.pth         # Best weights only ⭐
├── latest_checkpoint.pth          # Latest checkpoint
├── history.json                   # Training history
├── best_diagnostic_report.txt     # Diagnostic classification report
└── best_subtype_report.txt        # Subtype classification report
```

**Important**: Use `best_model_weights.pth` for Stage 2!

### Example Output

```
Epoch 25/50
------------------------------------------------------------
Epoch 25 [Train]: 100%|████████| 425/425 [05:23<00:00]
Test: 100%|████████████████████| 107/107 [00:52<00:00]

Epoch 25 Summary:
  Train - Loss: 0.2145, Diag Acc: 0.9234, Subtype Acc: 0.8567, Avg Acc: 0.8901
  Test  - Loss: 0.3012, Diag Acc: 0.9012, Subtype Acc: 0.8234, Avg Acc: 0.8623
  ✓ New best model saved! (Avg Acc: 0.8623)
```

### Monitoring Training

```python
# Load and visualize training history
import json
import matplotlib.pyplot as plt

with open('checkpoints/multiclass/history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

# Plot diagnostic accuracy
plt.subplot(1, 3, 1)
plt.plot(history['train_diag_acc'], label='Train')
plt.plot(history['test_diag_acc'], label='Test')
plt.title('Diagnostic Accuracy')
plt.legend()

# Plot subtype accuracy
plt.subplot(1, 3, 2)
plt.plot(history['train_subtype_acc'], label='Train')
plt.plot(history['test_subtype_acc'], label='Test')
plt.title('Subtype Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 3, 3)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['test_loss'], label='Test')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
```

## Stage 2: Train VLM

### Quick Start

```bash
cd /home/ray/research/eyeball-llm/eyeball-neurips/model

# Train VLM using the trained classifier
python train_llm.py \
    --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/vlm \
    --vlm_epochs 10 \
    --vlm_batch_size 2 \
    --use_contrastive
```

### Key Arguments

**Classifier**:
- `--classifier_checkpoint`: Path to trained classifier weights ⭐ **REQUIRED**
- `--num_diagnostic_classes`: Must match Stage 1 (default: 2)
- `--num_subtype_classes`: Must match Stage 1 (default: 4)

**Data Preparation**:
- `--skip_data_preparation`: Skip if data already prepared
- `--top_k_frames`: Number of important frames (default: 5)
- `--use_contrastive`: Create contrastive samples (default: True)

**VLM Model**:
- `--vlm_model`: Model name (default: 'Qwen/Qwen2-VL-7B-Instruct')
- `--use_4bit`: Use 4-bit quantization (default: True)

**LoRA**:
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.05)

**Training**:
- `--vlm_epochs`: Number of epochs (default: 10)
- `--vlm_batch_size`: Batch size (default: 2)
- `--vlm_lr`: Learning rate (default: 2e-5)

### Pipeline Steps

The script performs three steps:

**Step 1: Data Preparation**
- Loads trained classifier
- Extracts important frames from each video
- Generates attention heatmaps
- Creates text prompts with predictions
- Optionally creates contrastive samples

**Step 2: VLM Setup**
- Loads Qwen 2.5 VL model
- Applies 4-bit quantization
- Configures LoRA adapters

**Step 3: VLM Training**
- Finetunes VLM on prepared data
- Saves checkpoints during training
- Evaluates on test set

### Output

```
checkpoints/vlm/
├── config.json                    # Training configuration
├── vlm_data/                      # Prepared VLM data
│   ├── train/
│   │   ├── {video_id}/
│   │   │   ├── frame_0.png
│   │   │   ├── heatmap_0.png
│   │   │   └── {video_id}_metadata.json
│   │   └── all_samples.json
│   └── test/
│       └── ...
└── vlm_checkpoints/               # VLM model checkpoints
    ├── checkpoint-100/
    ├── checkpoint-200/
    ├── best_model/                # Best model ⭐
    └── trainer_state.json
```

### Example Output

```
==============================================================
VLM Training Pipeline
==============================================================
Classifier Checkpoint: ./checkpoints/multiclass/best_model_weights.pth
CSV Path: ../benchmarks/input/balanced_split_desc.csv
VLM Model: Qwen/Qwen2-VL-7B-Instruct
==============================================================

Loading pretrained classifier...
Loaded checkpoint from epoch 25
Test accuracy: 0.8623
✓ Classifier loaded successfully!

==============================================================
Step 1: Preparing VLM Training Data
==============================================================
Processing videos: 100%|████████| 4306/4306 [2:15:30<00:00]
Prepared 8612 samples (with contrastive)

==============================================================
Step 2: Setting up VLM Model
==============================================================
Loading Qwen2-VL-7B-Instruct...
Applying 4-bit quantization...
Configuring LoRA (r=16, alpha=32)...
✓ VLM model setup complete!

==============================================================
Step 3: Training VLM
==============================================================
Epoch 1/10: 100%|████████| 4306/4306 [3:45:12<00:00]
Eval: 100%|████████| 1077/1077 [0:52:30<00:00]
...
```

## Complete Example Workflow

### 1. Train Classifier

```bash
# Train for 50 epochs
python train_multiclass.py \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/exp1_multiclass \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --pretrained

# Check results
cat checkpoints/exp1_multiclass/best_diagnostic_report.txt
cat checkpoints/exp1_multiclass/best_subtype_report.txt
```

### 2. Train VLM

```bash
# Train VLM using the best classifier
python train_llm.py \
    --classifier_checkpoint ./checkpoints/exp1_multiclass/best_model_weights.pth \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/exp1_vlm \
    --vlm_epochs 10 \
    --vlm_batch_size 2 \
    --vlm_lr 2e-5 \
    --use_contrastive \
    --top_k_frames 5
```

### 3. Resume VLM Training (if interrupted)

```bash
# Skip data preparation if already done
python train_llm.py \
    --classifier_checkpoint ./checkpoints/exp1_multiclass/best_model_weights.pth \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/exp1_vlm \
    --skip_data_preparation \
    --vlm_epochs 20  # Continue for more epochs
```

## Data Splits

Both scripts use **stratified splitting** to ensure balanced classes:

```python
# Stratification by diagnostic + subtype combination
# Example distribution:
Train (80%):
  - non_rd + normal: 3272 samples
  - non_rd + pvd: 517 samples
  - rd + macula_intact: 346 samples
  - rd + macula_detached: 170 samples

Test (20%):
  - non_rd + normal: 819 samples
  - non_rd + pvd: 129 samples
  - rd + macula_intact: 87 samples
  - rd + macula_detached: 43 samples
```

## Hardware Requirements

### Stage 1 (Classifier)
- **GPU**: 8GB+ VRAM (e.g., RTX 3070, V100)
- **RAM**: 16GB+
- **Storage**: 50GB+ for dataset
- **Time**: ~5-6 hours for 50 epochs (batch_size=8)

### Stage 2 (VLM)
- **GPU**: 24GB+ VRAM (e.g., RTX 3090, A100)
  - With 4-bit quantization: 16GB+ VRAM
- **RAM**: 32GB+
- **Storage**: 100GB+ (for prepared data + checkpoints)
- **Time**: 
  - Data preparation: ~2-3 hours
  - VLM training: ~4-5 hours per epoch (batch_size=2)

## Tips & Best Practices

### For Classifier Training

1. **Start with pretrained weights**: `--pretrained` significantly improves convergence
2. **Monitor both tasks**: Check both diagnostic and subtype accuracy
3. **Use learning rate scheduling**: The script includes ReduceLROnPlateau
4. **Adjust batch size**: Reduce if OOM, increase if GPU underutilized

### For VLM Training

1. **Use 4-bit quantization**: Reduces VRAM usage significantly
2. **Enable contrastive learning**: Improves heatmap utilization
3. **Start with small LoRA rank**: r=16 is a good balance
4. **Save data preparation**: Use `--skip_data_preparation` for reruns
5. **Monitor GPU memory**: Reduce batch size if OOM

## Troubleshooting

### Out of Memory (OOM)

**Classifier**:
```bash
# Reduce batch size
python train_multiclass.py --batch_size 4

# Reduce frame count
python train_multiclass.py --num_frames 16
```

**VLM**:
```bash
# Reduce batch size
python train_llm.py --vlm_batch_size 1

# Use gradient accumulation (modify script)
# Or use smaller VLM model
python train_llm.py --vlm_model Qwen/Qwen2-VL-2B-Instruct
```

### Slow Training

```bash
# Increase num_workers for data loading
python train_multiclass.py --num_workers 8

# Use mixed precision (modify script to add autocast)
```

### Poor Performance

```bash
# Increase epochs
python train_multiclass.py --epochs 100

# Adjust learning rate
python train_multiclass.py --lr 5e-5

# Increase model capacity (reduce dropout)
python train_multiclass.py --dropout 0.2
```

## Evaluation

### Classifier Evaluation

```python
# Load best model and evaluate
from multiclass_model import create_multiclass_model
import torch

model = create_multiclass_model(num_diagnostic_classes=2, num_subtype_classes=4)
checkpoint = torch.load('checkpoints/multiclass/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Best epoch: {checkpoint['epoch']}")
print(f"Test accuracy: {checkpoint['test_acc']:.4f}")
print(f"Diagnostic accuracy: {checkpoint['test_diag_acc']:.4f}")
print(f"Subtype accuracy: {checkpoint['test_subtype_acc']:.4f}")
```

### VLM Evaluation

```python
# Use the inference script or vlm_pipeline.py
from vlm_pipeline import VLMDiagnosisPipeline

pipeline = VLMDiagnosisPipeline(
    classifier_checkpoint='checkpoints/multiclass/best_model_weights.pth'
)

pipeline.load_vlm('checkpoints/vlm/vlm_checkpoints/best_model')

# Run diagnosis on test video
result = pipeline.diagnose_video('path/to/test_video.mp4')
```

## Summary

1. **Train classifier first**: Get good diagnostic + subtype predictions
2. **Use best weights for VLM**: `best_model_weights.pth`
3. **Monitor both stages**: Check accuracy and loss curves
4. **Save checkpoints**: Both scripts save best models automatically
5. **Adjust hyperparameters**: Based on your hardware and performance

Happy training! 🚀
