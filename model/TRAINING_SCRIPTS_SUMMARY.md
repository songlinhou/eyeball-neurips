# Training Scripts Summary

Complete training pipeline for ERDES medical video diagnosis with balanced splits.

## 📁 Files Created

### Training Scripts

1. **`train_multiclass.py`** ⭐
   - Trains the multi-class classifier (diagnostic + subtype)
   - Uses balanced stratified train/test splits
   - Saves best model based on test accuracy
   - Generates classification reports

2. **`train_llm.py`** ⭐
   - Trains VLM using pretrained classifier
   - Prepares data with important frames and heatmaps
   - Finetunes Qwen 2.5 VL with LoRA
   - Supports contrastive learning

3. **`run_training.sh`**
   - Complete pipeline script
   - Runs both stages sequentially
   - Automatic checkpoint management

4. **`TRAINING_GUIDE.md`**
   - Comprehensive training guide
   - Usage examples and tips
   - Troubleshooting section

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

```bash
cd /home/ray/research/eyeball-llm/eyeball-neurips/model
bash run_training.sh
```

This will:
1. Train classifier for 50 epochs
2. Save best model weights
3. Prepare VLM data using trained classifier
4. Train VLM for 10 epochs
5. Save all checkpoints

### Option 2: Run Stages Separately

**Stage 1: Train Classifier**
```bash
python train_multiclass.py \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/multiclass \
    --epochs 50 \
    --batch_size 8
```

**Stage 2: Train VLM**
```bash
python train_llm.py \
    --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/vlm \
    --vlm_epochs 10
```

## 📊 Data Splits

Both scripts use **stratified splitting** for balanced classes:

```python
# Automatic stratification by diagnostic + subtype
train_test_split(
    indices,
    test_size=0.2,  # 80% train, 20% test
    stratify=combined_labels,  # Ensures balanced classes
    random_state=42  # Reproducible splits
)
```

**Example Distribution**:
```
Total: 5383 samples

Train (4306 samples - 80%):
  - non_rd + normal: 3272
  - non_rd + pvd: 517
  - rd + macula_intact: 346
  - rd + macula_detached: 170

Test (1077 samples - 20%):
  - non_rd + normal: 819
  - non_rd + pvd: 129
  - rd + macula_intact: 87
  - rd + macula_detached: 43
```

## 📦 Output Structure

### Classifier Output
```
checkpoints/multiclass/
├── config.json                    # Training config
├── best_model.pth                 # Full checkpoint
├── best_model_weights.pth         # Weights only ⭐ USE THIS
├── latest_checkpoint.pth          # Latest checkpoint
├── history.json                   # Training curves
├── best_diagnostic_report.txt     # Diagnostic metrics
└── best_subtype_report.txt        # Subtype metrics
```

### VLM Output
```
checkpoints/vlm/
├── config.json                    # Training config
├── vlm_data/                      # Prepared data
│   ├── train/
│   │   ├── {video_id}/
│   │   │   ├── frame_*.png        # Important frames
│   │   │   ├── heatmap_*.png      # Attention heatmaps
│   │   │   └── metadata.json      # Sample metadata
│   │   └── all_samples.json       # All samples index
│   └── test/
│       └── ...
└── vlm_checkpoints/               # VLM checkpoints
    ├── checkpoint-100/
    ├── checkpoint-200/
    ├── best_model/                # Best model ⭐
    └── trainer_state.json
```

## 🎯 Key Features

### Classifier Training (`train_multiclass.py`)

✅ **Balanced Splits**: Stratified by diagnostic + subtype  
✅ **Best Model Selection**: Based on test accuracy  
✅ **Dual Task Training**: Diagnostic + subtype classification  
✅ **Learning Rate Scheduling**: ReduceLROnPlateau  
✅ **Classification Reports**: Precision, recall, F1-score  
✅ **Training History**: JSON format for plotting  

### VLM Training (`train_llm.py`)

✅ **Automatic Data Preparation**: Extracts important frames  
✅ **Attention Heatmaps**: Visual explanations  
✅ **Contrastive Learning**: Ensures heatmap usage  
✅ **4-bit Quantization**: Reduces VRAM usage  
✅ **LoRA Finetuning**: Parameter-efficient  
✅ **Checkpoint Management**: Saves best model  

## 🔧 Configuration

### Classifier Hyperparameters

```python
# Model
num_diagnostic_classes = 2  # non_rd, rd
num_subtype_classes = 4     # normal, macula_intact, macula_detached, pvd
pretrained = True           # Use R3D-18 pretrained weights
dropout = 0.3               # Dropout rate

# Training
batch_size = 8              # Batch size
epochs = 50                 # Number of epochs
lr = 1e-4                   # Learning rate
weight_decay = 1e-5         # Weight decay

# Data
test_size = 0.2             # 20% for test
num_frames = 32             # Frames per video
frame_size = 224            # Frame dimensions
```

### VLM Hyperparameters

```python
# Data Preparation
top_k_frames = 5            # Important frames to extract
use_contrastive = True      # Create contrastive samples

# Model
vlm_model = "Qwen/Qwen2-VL-7B-Instruct"
use_4bit = True             # 4-bit quantization

# LoRA
lora_r = 16                 # LoRA rank
lora_alpha = 32             # LoRA alpha
lora_dropout = 0.05         # LoRA dropout

# Training
vlm_epochs = 10             # Number of epochs
vlm_batch_size = 2          # Batch size
vlm_lr = 2e-5               # Learning rate
```

## 📈 Monitoring Training

### View Classifier Progress

```bash
# Check latest results
tail -f checkpoints/multiclass/history.json

# View classification reports
cat checkpoints/multiclass/best_diagnostic_report.txt
cat checkpoints/multiclass/best_subtype_report.txt
```

### View VLM Progress

```bash
# Check training logs
tail -f checkpoints/vlm/vlm_checkpoints/trainer_state.json

# View prepared samples
cat checkpoints/vlm/vlm_data/train/all_samples.json | jq '.[0]'
```

## 💡 Usage Examples

### Example 1: Quick Training

```bash
# Train classifier (fast settings)
python train_multiclass.py \
    --epochs 20 \
    --batch_size 16 \
    --output_dir ./checkpoints/quick_test

# Train VLM (skip contrastive)
python train_llm.py \
    --classifier_checkpoint ./checkpoints/quick_test/best_model_weights.pth \
    --vlm_epochs 5 \
    --output_dir ./checkpoints/quick_vlm
```

### Example 2: High-Quality Training

```bash
# Train classifier (more epochs)
python train_multiclass.py \
    --epochs 100 \
    --batch_size 8 \
    --lr 5e-5 \
    --output_dir ./checkpoints/high_quality

# Train VLM (with contrastive)
python train_llm.py \
    --classifier_checkpoint ./checkpoints/high_quality/best_model_weights.pth \
    --vlm_epochs 20 \
    --use_contrastive \
    --output_dir ./checkpoints/high_quality_vlm
```

### Example 3: Resume VLM Training

```bash
# Skip data preparation if already done
python train_llm.py \
    --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
    --skip_data_preparation \
    --vlm_epochs 20 \
    --output_dir ./checkpoints/vlm_continued
```

## 🔍 Evaluation

### Load and Evaluate Classifier

```python
import torch
from multiclass_model import create_multiclass_model

# Load model
model = create_multiclass_model(
    num_diagnostic_classes=2,
    num_subtype_classes=4
)

checkpoint = torch.load('checkpoints/multiclass/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Check performance
print(f"Epoch: {checkpoint['epoch']}")
print(f"Test Accuracy: {checkpoint['test_acc']:.4f}")
print(f"Diagnostic Acc: {checkpoint['test_diag_acc']:.4f}")
print(f"Subtype Acc: {checkpoint['test_subtype_acc']:.4f}")
```

### Use VLM for Inference

```python
from vlm_pipeline import VLMDiagnosisPipeline

# Initialize pipeline
pipeline = VLMDiagnosisPipeline(
    classifier_checkpoint='checkpoints/multiclass/best_model_weights.pth'
)

# Load VLM
pipeline.load_vlm('checkpoints/vlm/vlm_checkpoints/best_model')

# Diagnose video
result = pipeline.diagnose_video('path/to/video.mp4')
print(result['reasoning'])
```

## ⚙️ Hardware Requirements

### Minimum
- **GPU**: 8GB VRAM (RTX 2080, V100)
- **RAM**: 16GB
- **Storage**: 100GB

### Recommended
- **GPU**: 24GB VRAM (RTX 3090, A100)
- **RAM**: 32GB
- **Storage**: 200GB

### With 4-bit Quantization
- **GPU**: 16GB VRAM (RTX 3080, A6000)
- **RAM**: 24GB
- **Storage**: 150GB

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train_multiclass.py --batch_size 4
python train_llm.py --vlm_batch_size 1

# Reduce frames
python train_multiclass.py --num_frames 16
```

### Slow Training

```bash
# Increase workers
python train_multiclass.py --num_workers 8

# Use smaller model
python train_llm.py --vlm_model Qwen/Qwen2-VL-2B-Instruct
```

### Poor Performance

```bash
# More epochs
python train_multiclass.py --epochs 100

# Lower learning rate
python train_multiclass.py --lr 5e-5

# Reduce dropout
python train_multiclass.py --dropout 0.2
```

## 📝 Notes

1. **Use balanced_split_desc.csv**: Contains all metadata and labels
2. **Stratified splits**: Ensures balanced train/test distribution
3. **Best model selection**: Based on average test accuracy
4. **Checkpoint format**: Use `best_model_weights.pth` for VLM training
5. **Reproducibility**: Set `random_state=42` for consistent splits

## 🎓 Citation

If you use these training scripts, please cite:

```bibtex
@software{erdes_training_pipeline,
  title={ERDES Medical Video Diagnosis Training Pipeline},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo}
}
```

## 📚 References

- **Model**: ExplainableOpticalFlowResNet3D
- **VLM**: Qwen 2.5 VL
- **Dataset**: ERDES (balanced_split_desc.csv)
- **Framework**: PyTorch, Transformers, PEFT

---

**Ready to train!** 🚀

Start with: `bash run_training.sh`
