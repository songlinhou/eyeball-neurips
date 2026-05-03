# Quick Reference - exp13

## 🚀 Quick Start

```bash
# Train
python exp13_train.py

# Evaluate
python exp13_evaluate.py
```

## 📊 Key Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | ExplainableResNet3D | R3D-18 + attention modules |
| Dropout | 0.3 | Lower for better learning |
| Learning Rate | 2e-4 | Higher for faster convergence |
| Epochs | 10 | 3 phase1 + 7 phase2 |
| Batch Size | 16 | Adjust for GPU memory |
| Frames | 32 | Temporal resolution |
| Image Size | 224×224 | Spatial resolution |
| Loss | Focal (γ=2.0) | Handles class imbalance |
| Augmentation | Mixup + TTA | Improves generalization |

## 🏗️ Model Architecture

```
Input: (B, 3, 32, 224, 224)
  ↓
R3D-18 Backbone (Kinetics-400 pretrained)
  ↓
Frame Importance (temporal attention)
  ↓
Spatial Explainability (spatial attention)
  ↓
Global Pooling → Classifier
  ↓
Output: (B, 2) [Intact, Detached]
```

## 🎯 Training Strategy

**Phase 1** (3 epochs):
- Freeze backbone
- LR: 2e-3
- No mixup

**Phase 2** (7 epochs):
- Unfreeze all
- LR: 2e-4
- Mixup enabled
- Early stopping

## 📁 Files

| File | Purpose | Lines |
|------|---------|-------|
| `exp13_config.py` | Configuration | 90 |
| `exp13_model.py` | Model definition | 200 |
| `exp13_train.py` | Training pipeline | 550 |
| `exp13_evaluate.py` | Evaluation | 350 |
| `README.md` | Documentation | 300 |

## 🔧 Common Modifications

### Change Learning Rate
```python
# In exp13_config.py
LEARNING_RATE = 1e-4  # Lower
# or
LEARNING_RATE = 5e-4  # Higher
```

### Adjust Batch Size
```python
# In exp13_config.py
BATCH_SIZE = 8   # Smaller GPU
BATCH_SIZE = 32  # Larger GPU
```

### More Epochs
```python
# In exp13_config.py
NUM_EPOCHS = 20
```

### Disable Mixup
```python
# In exp13_config.py
USE_MIXUP = False
```

## 📈 Expected Results

| Metric | Range |
|--------|-------|
| Validation Acc | 85-92% |
| Test Acc | 83-90% |
| Test F1 | 0.82-0.90 |
| Test AUC | 0.90-0.97 |

## 🎨 Outputs

### Training
- `models/exp13_*_best.pth` - Best model
- `logs/exp13_*.log` - Training log
- `plots/exp13_*_history.png` - Training curves
- `checkpoints/exp13_*.pth` - Checkpoints

### Evaluation
- `evaluation/*_roc_curve.png` - ROC curve
- `evaluation/*_pr_curve.png` - PR curve
- `evaluation/*_confusion_matrix_detailed.png` - CM
- `evaluation/*_frame_importance.png` - Attention
- `evaluation/*_classification_report.txt` - Report

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of Memory | Reduce `BATCH_SIZE` or `NUM_FRAMES` |
| Slow Training | Increase `NUM_WORKERS` |
| Poor Performance | Increase `NUM_EPOCHS`, adjust `LEARNING_RATE` |
| Import Error | Check `sys.path` includes `video_classification/` |

## 💡 Usage Examples

### Get Attention Maps
```python
from exp13_model import ExplainableResNet3D
import torch

model = ExplainableResNet3D(num_classes=2, pretrained=True, dropout=0.3)
model.eval()

# Forward with attention
outputs, attention = model(videos, return_attention=True)

# Access attention
frame_importance = attention['frame_importance']  # (B, T)
spatial_attention = attention['spatial_attention']  # (B, 1, T, H, W)
```

### Load Trained Model
```python
import torch
from exp13_model import ExplainableResNet3D

model = ExplainableResNet3D(num_classes=2, pretrained=False, dropout=0.3)
model.load_state_dict(torch.load('exp13_results/models/exp13_best.pth'))
model.eval()
```

### Custom Evaluation
```python
from exp13_evaluate import evaluate_model
from exp13_config import *
import torch

device = torch.device('cuda')
model = load_model('path/to/checkpoint.pth', device)
results = evaluate_model(model, test_loader, device)

print(f"Accuracy: {100 * np.mean(results['predictions'] == results['labels']):.2f}%")
```

## 📚 Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `get_config()` | exp13_config.py | Get config dict |
| `ExplainableResNet3D()` | exp13_model.py | Create model |
| `train_experiment()` | exp13_train.py | Run training |
| `evaluate_experiment()` | exp13_evaluate.py | Run evaluation |

## 🔗 Dependencies

**Required**:
- `torch`, `torchvision`
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `tqdm`

**Internal**:
- `../video_classification/improved_dataset.py`

## 📝 Notes

- GPU recommended (8GB+ VRAM)
- Dataset: ERDES macula_detached_vs_intact
- Pretrained on Kinetics-400
- Two-phase training strategy
- Explainable via attention maps
