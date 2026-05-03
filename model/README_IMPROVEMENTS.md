# Ocular Ultrasound Video Classification - Improvement Package

This package provides comprehensive improvements for your ResNet3D-based classifier for macula intact vs. detached classification.

## 📁 Files Overview

### Core Implementation Files

1. **`improved_model.py`** - Enhanced model architectures
   - `ImprovedResNet3D`: ResNet3D + Temporal + Spatial Attention
   - `MultiScaleResNet3D`: Multi-scale feature extraction
   - `ResNet3DWithAuxiliary`: Auxiliary task learning
   - `FocalLoss`: For handling class imbalance
   - `LabelSmoothingCrossEntropy`: Regularization

2. **`improved_dataset.py`** - Advanced data loading and augmentation
   - `ImprovedVideoDataset`: Dataset with augmentation
   - `VideoAugmentation`: Temporal and spatial augmentation
   - `MultiViewVideoDataset`: Multi-view learning
   - `mixup_data()`: Mixup augmentation
   - `create_improved_dataloaders()`: Easy dataloader creation

3. **`improved_training.py`** - Advanced training pipeline
   - Gradual unfreezing strategy
   - Multiple loss functions
   - Learning rate scheduling
   - Early stopping
   - Test-time augmentation
   - Comprehensive evaluation

4. **`compare_models.py`** - Model comparison utilities
   - `quick_comparison()`: Compare 3 main architectures
   - `compare_all_models()`: Comprehensive comparison
   - `ablation_study()`: Understand component contributions

5. **`analysis_utils.py`** - Visualization and analysis tools
   - Attention map visualization
   - Confusion matrix plotting
   - ROC curve analysis
   - Misclassification analysis
   - Training history plots

### Documentation

6. **`IMPROVEMENT_GUIDE.md`** - Comprehensive improvement guide
   - Problem analysis
   - Detailed explanations of each improvement
   - Hyperparameter recommendations
   - Troubleshooting tips

7. **`README_IMPROVEMENTS.md`** - This file

## 🚀 Quick Start

### Option 1: Train Single Best Model (Recommended)

```python
python improved_training.py
```

This will train the `ImprovedResNet3D` with all improvements:
- ✅ Temporal and spatial attention
- ✅ Focal loss for class imbalance
- ✅ Mixup augmentation
- ✅ Gradual unfreezing
- ✅ Test-time augmentation
- ✅ Early stopping

**Expected time**: 2-4 hours on GPU
**Expected improvement**: 10-30% over baseline

### Option 2: Compare Different Architectures

```bash
# Quick comparison (3 models, 20 epochs each)
python compare_models.py --mode quick

# Full comparison (6 configurations, 30 epochs each)
python compare_models.py --mode full

# Ablation study (understand each component)
python compare_models.py --mode ablation
```

### Option 3: Custom Training

```python
from improved_training import train_with_gradual_unfreezing

# Train with custom settings
model, val_acc, test_acc = train_with_gradual_unfreezing(
    model_class='improved',      # 'improved', 'multiscale', or 'auxiliary'
    num_epochs=30,
    use_focal_loss=True,         # Handle class imbalance
    use_mixup=True,              # Data augmentation
    use_tta=True,                # Test-time augmentation
    save_path='my_best_model.pth'
)

print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
```

## 📊 Analysis and Visualization

### Analyze Training Results

```python
from analysis_utils import *
import torch

# Load your trained model
model = ImprovedResNet3D(num_classes=2, pretrained=True)
model.load_state_dict(torch.load('best_improved_resnet3d.pth'))
model.to(device)

# Visualize attention maps
video = test_dataset[0][0]  # Get a test video
visualize_attention_maps(model, video, device, save_path='attention.png')

# Analyze misclassifications
misclassified = analyze_misclassifications(model, test_loader, device, num_examples=5)

# Plot confusion matrix
y_true, y_pred = [], []
# ... collect predictions ...
plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')

# Plot ROC curve
y_probs = []  # Probability scores
# ... collect probabilities ...
plot_roc_curve(y_true, y_probs, save_path='roc_curve.png')
```

### Compare Multiple Models

```python
from analysis_utils import compare_model_predictions

models = {
    'Improved': model1,
    'MultiScale': model2,
    'Auxiliary': model3
}

video = test_dataset[0][0]
compare_model_predictions(models, video, device)
```

## 🎯 Key Improvements Explained

### 1. Architecture Enhancements

**Temporal Attention**
- Learns which frames are most important
- Focuses on critical moments in the video
- Improves temporal reasoning

**Spatial Attention (CBAM)**
- Highlights important regions (macula area)
- Reduces focus on irrelevant background
- Improves spatial localization

**Multi-scale Features**
- Combines low-level details and high-level semantics
- Better feature representation
- Improves classification accuracy

### 2. Training Strategy

**Gradual Unfreezing**
- Phase 1: Train classifier only (5 epochs)
- Phase 2: Fine-tune entire model (25 epochs)
- Prevents destroying pretrained features
- Better convergence

**Focal Loss**
- Focuses on hard examples
- Handles class imbalance
- Improves minority class performance

**Mixup Augmentation**
- Creates synthetic training examples
- Improves generalization
- Reduces overfitting

### 3. Data Augmentation

**Temporal**
- Random frame sampling
- Temporal jittering
- Increases temporal diversity

**Spatial**
- Horizontal flip
- Small rotations (±10°)
- Brightness/contrast adjustment
- Gaussian noise

### 4. Regularization

- Dropout (0.5 in classifier)
- Weight decay (1e-4)
- Gradient clipping (max_norm=1.0)
- Early stopping (patience=7)
- Label smoothing (optional)

## 📈 Expected Results

### Baseline (Your Current Model)
```
- Simple ResNet3D fine-tuning
- No augmentation
- Basic training
- Accuracy: ~50-60% (example)
```

### With Improvements
```
- ImprovedResNet3D + All techniques
- Full augmentation pipeline
- Advanced training strategy
- Expected Accuracy: 70-85%
- Expected Improvement: +10-30%
```

### Component Contributions (Approximate)
- Better architecture: +5-15%
- Data augmentation: +3-8%
- Training strategy: +2-5%
- Test-time augmentation: +1-3%

## 🔧 Hyperparameter Tuning

### Priority Order

1. **Learning Rate** (Most Important)
   ```python
   # Try: 5e-5, 1e-4, 5e-4
   LEARNING_RATE = 1e-4
   ```

2. **Dropout**
   ```python
   # Try: 0.3, 0.5, 0.7
   model = ImprovedResNet3D(dropout=0.5)
   ```

3. **Batch Size**
   ```python
   # Larger is often better (if GPU allows)
   # Try: 8, 16, 32
   BATCH_SIZE = 8
   ```

4. **Number of Frames**
   ```python
   # More frames = more context
   # Try: 16, 32, 64
   NUM_FRAMES = 32
   ```

5. **Focal Loss Gamma**
   ```python
   # Higher = more focus on hard examples
   # Try: 1.0, 2.0, 3.0
   criterion = FocalLoss(gamma=2.0)
   ```

## 🐛 Troubleshooting

### Low Accuracy After Training

1. **Check data quality**
   - Verify labels are correct
   - Check for corrupted videos
   - Ensure balanced train/val/test splits

2. **Reduce overfitting**
   - Increase dropout
   - Add more augmentation
   - Reduce model complexity

3. **Improve training**
   - Lower learning rate
   - Train longer
   - Use different optimizer (SGD vs Adam)

### Out of Memory Errors

1. **Reduce batch size**
   ```python
   BATCH_SIZE = 4  # or 2
   ```

2. **Reduce number of frames**
   ```python
   NUM_FRAMES = 16  # instead of 32
   ```

3. **Use gradient accumulation**
   ```python
   # Accumulate gradients over 4 steps
   # Effective batch size = 4 * actual batch size
   ```

### Training Too Slow

1. **Reduce num_workers**
   ```python
   num_workers = 2  # instead of 4
   ```

2. **Use mixed precision training**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```

3. **Reduce number of epochs**
   ```python
   NUM_EPOCHS = 20  # instead of 30
   ```

## 📚 Next Steps

### After Initial Training

1. **Analyze results**
   - Check confusion matrix
   - Identify error patterns
   - Visualize attention maps

2. **Iterate**
   - Adjust hyperparameters based on results
   - Try different architectures
   - Experiment with ensemble methods

3. **Advanced techniques** (if needed)
   - Semi-supervised learning
   - Self-supervised pretraining
   - Knowledge distillation
   - Neural architecture search

### For Production Deployment

1. **Model optimization**
   - Quantization
   - Pruning
   - ONNX export

2. **Inference optimization**
   - TensorRT
   - Batch inference
   - Model caching

3. **Monitoring**
   - Track prediction confidence
   - Monitor for distribution shift
   - Regular retraining

## 💡 Tips for Best Results

1. **Start simple**: Begin with `ImprovedResNet3D` and default settings
2. **Monitor training**: Use TensorBoard or similar for visualization
3. **Validate carefully**: Ensure validation set is representative
4. **Be patient**: Good models take time to train
5. **Iterate**: Don't expect perfect results on first try
6. **Consult experts**: Medical imaging benefits from domain knowledge

## 📞 Support

If you encounter issues:
1. Check the `IMPROVEMENT_GUIDE.md` for detailed explanations
2. Review error messages carefully
3. Verify your data pipeline is working correctly
4. Start with smaller experiments to debug

## 🎓 Learning Resources

- **Attention Mechanisms**: "Attention Is All You Need" paper
- **Focal Loss**: "Focal Loss for Dense Object Detection" paper
- **Video Classification**: "Quo Vadis, Action Recognition?" paper
- **Medical Imaging**: "Deep Learning in Medical Imaging" reviews

## ✅ Checklist

Before training:
- [ ] Data is properly organized
- [ ] CSV files have correct paths and labels
- [ ] GPU is available and working
- [ ] Dependencies are installed
- [ ] Sufficient disk space for checkpoints

During training:
- [ ] Monitor training/validation metrics
- [ ] Check for overfitting (train >> val)
- [ ] Watch for NaN losses or gradients
- [ ] Save best model checkpoints

After training:
- [ ] Evaluate on test set
- [ ] Analyze confusion matrix
- [ ] Visualize predictions
- [ ] Compare with baseline

Good luck with your improved classifier! 🚀
