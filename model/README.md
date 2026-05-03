# Improved Ocular Ultrasound Video Classifier

This folder contains all the improved code and documentation for enhancing your ResNet3D-based classifier for macula intact vs. detached classification.

## 📁 Folder Structure

```
improved_classifier/
├── README.md                      # This file - Quick start guide
├── IMPROVEMENT_GUIDE.md           # Comprehensive improvement explanations
├── README_IMPROVEMENTS.md         # Detailed usage instructions
├── architecture_summary.txt       # Visual architecture overview
│
├── improved_model.py              # Enhanced model architectures
├── improved_dataset.py            # Advanced data loading & augmentation
├── improved_training.py           # Complete training pipeline
├── compare_models.py              # Model comparison utilities
└── analysis_utils.py              # Visualization & analysis tools
```

## 🚀 Quick Start

### 1. Navigate to this folder

```bash
cd /content/improved_classifier
```

### 2. Train the best model

```bash
python improved_training.py
```

This will:
- Load data from `../erdes` (parent directory)
- Train ImprovedResNet3D with all improvements
- Use gradual unfreezing, focal loss, mixup, and TTA
- Save the best model as `best_improved_resnet3d.pth`

### 3. Compare different architectures

```bash
# Quick comparison (3 models, ~2-3 hours)
python compare_models.py --mode quick

# Full comparison (6 configurations, ~6-8 hours)
python compare_models.py --mode full

# Ablation study (understand component contributions)
python compare_models.py --mode ablation
```

## 📊 Dataset Path Configuration

The code expects your dataset to be located at:
```
/content/erdes/
├── splits/
│   └── macula_detached_vs_intact/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── [video files as specified in CSV paths]
```

If your dataset is in a different location, update the path in `improved_training.py`:

```python
# Line 21 in improved_training.py
DATA_DIR = "../erdes"  # Change this to your dataset path
```

## 💡 Key Features

### Enhanced Architectures
- **ImprovedResNet3D**: Temporal + Spatial Attention
- **MultiScaleResNet3D**: Multi-scale feature extraction
- **ResNet3DWithAuxiliary**: Auxiliary task learning

### Advanced Training
- **Gradual Unfreezing**: 2-phase training strategy
- **Focal Loss**: Handles class imbalance
- **Mixup Augmentation**: Synthetic training examples
- **Test-Time Augmentation**: Improved inference

### Data Augmentation
- Temporal jittering
- Spatial transforms (flip, rotation, brightness, contrast)
- Gaussian noise
- Mixup

## 📈 Expected Results

| Approach | Expected Accuracy |
|----------|------------------|
| Baseline (simple fine-tuning) | ~50-60% |
| **With all improvements** | **70-85%** |
| **Expected gain** | **+10-30%** |

## 🔧 Customization

### Train with custom settings

```python
from improved_training import train_with_gradual_unfreezing

model, val_acc, test_acc = train_with_gradual_unfreezing(
    model_class='improved',      # 'improved', 'multiscale', or 'auxiliary'
    num_epochs=30,
    use_focal_loss=True,
    use_mixup=True,
    use_tta=True,
    save_path='my_model.pth'
)
```

### Adjust hyperparameters

Edit `improved_training.py` lines 23-27:

```python
NUM_FRAMES = 32        # Try: 16, 32, 64
IMG_SIZE = 224         # Standard for pretrained models
BATCH_SIZE = 8         # Increase if GPU allows: 16, 32
NUM_EPOCHS = 30        # With early stopping
LEARNING_RATE = 1e-4   # Try: 5e-5, 1e-4, 5e-4
```

## 📚 Documentation

- **`IMPROVEMENT_GUIDE.md`**: Detailed explanations of all improvements
- **`README_IMPROVEMENTS.md`**: Comprehensive usage guide
- **`architecture_summary.txt`**: Visual architecture diagrams

## 🛠️ Analysis Tools

```python
from analysis_utils import *

# Visualize attention maps
visualize_attention_maps(model, video, device, save_path='attention.png')

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, save_path='confusion.png')

# Analyze misclassifications
analyze_misclassifications(model, test_loader, device)

# Plot ROC curve
plot_roc_curve(y_true, y_probs, save_path='roc.png')
```

## ⚙️ Requirements

All dependencies should already be installed in your environment:
- PyTorch >= 1.9
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size in improved_training.py
BATCH_SIZE = 4  # or 2
```

### Dataset not found
```python
# Update path in improved_training.py
DATA_DIR = "/path/to/your/erdes"
```

### Slow training
```python
# Reduce workers in improved_training.py line 196
num_workers=2  # instead of 4
```

## 📞 Next Steps

1. **Start simple**: Run `python improved_training.py`
2. **Monitor training**: Check validation metrics
3. **Analyze results**: Use analysis_utils.py
4. **Iterate**: Adjust hyperparameters based on results
5. **Compare**: Try different architectures

## ✅ Checklist

Before training:
- [ ] Dataset is at `/content/erdes/` or path is updated
- [ ] CSV files exist and have correct format
- [ ] GPU is available (`torch.cuda.is_available()`)
- [ ] Sufficient disk space for checkpoints

Good luck with your improved classifier! 🚀
