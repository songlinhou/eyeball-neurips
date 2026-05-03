# ResNet3D Classifier Improvement Guide for Ocular Ultrasound Videos

## Problem Analysis

Your current implementation has several limitations:
1. **Simple architecture**: Only replacing the final FC layer of pretrained ResNet3D
2. **No data augmentation**: Medical videos benefit greatly from augmentation
3. **Aggressive LR decay**: StepLR with gamma=0.1 may cause premature convergence
4. **No class balancing**: Imbalanced datasets hurt minority class performance
5. **Limited temporal modeling**: Basic ResNet3D may miss subtle temporal patterns

## Recommended Solutions

### 1. Enhanced Architecture Options

#### Option A: ImprovedResNet3D (Recommended for Start)
- **Temporal Attention**: Focuses on critical frames in the video sequence
- **Spatial Attention (CBAM)**: Highlights important regions (macula area)
- **Enhanced Classifier Head**: Multi-layer with dropout and batch normalization
- **Benefits**: Better feature learning, interpretable attention maps

```python
from improved_model import ImprovedResNet3D
model = ImprovedResNet3D(num_classes=2, pretrained=True, dropout=0.5, use_attention=True)
```

#### Option B: MultiScaleResNet3D
- **Multi-scale features**: Extracts features from layers 2, 3, and 4
- **Feature fusion**: Combines low-level and high-level features
- **Benefits**: Captures both fine details and semantic information

```python
from improved_model import MultiScaleResNet3D
model = MultiScaleResNet3D(num_classes=2, pretrained=True, dropout=0.5)
```

#### Option C: ResNet3DWithAuxiliary
- **Auxiliary classifier**: Additional supervision from intermediate layers
- **Multi-task learning**: Helps prevent overfitting
- **Benefits**: Better gradient flow, improved generalization

```python
from improved_model import ResNet3DWithAuxiliary
model = ResNet3DWithAuxiliary(num_classes=2, pretrained=True, dropout=0.5)
```

### 2. Advanced Loss Functions

#### Focal Loss (Recommended for Imbalanced Data)
- Focuses on hard-to-classify examples
- Reduces weight of easy examples
- Handles class imbalance effectively

```python
from improved_model import FocalLoss
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

#### Label Smoothing
- Prevents overconfident predictions
- Improves generalization
- Reduces overfitting

```python
from improved_model import LabelSmoothingCrossEntropy
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

### 3. Data Augmentation Strategy

#### Temporal Augmentation
- **Random temporal sampling**: Different frame selections per epoch
- **Temporal jittering**: Adds variation in frame timing
- **Benefits**: Increases effective dataset size, improves temporal invariance

#### Spatial Augmentation
- **Horizontal flip**: Safe for medical imaging
- **Small rotations** (±10°): Accounts for probe orientation
- **Brightness/Contrast**: Handles different ultrasound settings
- **Gaussian noise**: Improves robustness

#### Mixup (Advanced)
- Mixes two videos and their labels
- Creates synthetic training examples
- Improves generalization significantly

```python
from improved_dataset import create_improved_dataloaders
train_loader, val_loader, test_loader, class_weights = create_improved_dataloaders(
    DATA_DIR, SPLITS_DIR, NUM_FRAMES, IMG_SIZE, BATCH_SIZE, 
    num_workers=4, use_augmentation=True
)
```

### 4. Training Strategy: Gradual Unfreezing

#### Phase 1: Train Classifier Head Only (5 epochs)
- Freeze backbone weights
- Train only the new classifier layers
- Higher learning rate (10x base LR)
- **Why**: Prevents destroying pretrained features with random classifier weights

#### Phase 2: Fine-tune Entire Model (20-25 epochs)
- Unfreeze all layers
- Lower learning rate (base LR)
- Use learning rate scheduling
- **Why**: Adapts pretrained features to your specific task

```python
# Phase 1
model.freeze_backbone()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=LEARNING_RATE * 10)

# Phase 2
model.unfreeze_backbone()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```

### 5. Learning Rate Scheduling

#### CosineAnnealingWarmRestarts (Phase 1)
- Periodic learning rate restarts
- Helps escape local minima
- Good for initial training

```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
```

#### ReduceLROnPlateau (Phase 2)
- Reduces LR when validation metric plateaus
- Adaptive to training dynamics
- Prevents premature convergence

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                 factor=0.5, patience=3)
```

### 6. Regularization Techniques

#### Dropout
- Applied in classifier head
- Prevents overfitting
- Recommended: 0.5 for first layer, 0.25 for subsequent layers

#### Weight Decay
- L2 regularization on weights
- Recommended: 1e-4

#### Gradient Clipping
- Prevents exploding gradients
- Recommended: max_norm=1.0

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Early Stopping
- Stops training when validation performance plateaus
- Prevents overfitting
- Recommended patience: 7 epochs

### 7. Test-Time Augmentation (TTA)

- Apply augmentations during inference
- Average predictions from multiple augmented versions
- Typically improves accuracy by 1-3%

```python
# Horizontal flip TTA
outputs1 = model(videos)
videos_flip = torch.flip(videos, dims=[4])
outputs2 = model(videos_flip)
final_output = (outputs1 + outputs2) / 2
```

### 8. Evaluation Metrics

Beyond accuracy, track:
- **Precision**: Important if false positives are costly
- **Recall**: Important if false negatives are costly (medical diagnosis)
- **F1-Score**: Balanced metric for imbalanced data
- **AUC-ROC**: Measures classifier's ability to distinguish classes
- **Confusion Matrix**: Shows specific error patterns

### 9. Hyperparameter Recommendations

```python
# Model
NUM_FRAMES = 32  # Try 16, 32, 64
IMG_SIZE = 224   # Standard for pretrained models
DROPOUT = 0.5    # Try 0.3-0.7

# Training
BATCH_SIZE = 8   # Increase if GPU memory allows (16, 32)
LEARNING_RATE = 1e-4  # Try 5e-5, 1e-4, 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30  # With early stopping

# Loss
FOCAL_GAMMA = 2.0  # Try 1.0-3.0
MIXUP_ALPHA = 0.2  # Try 0.1-0.4
```

### 10. Ensemble Methods (Advanced)

If single model performance is still insufficient:

#### Temporal Ensemble
- Train models with different frame counts (16, 32, 64)
- Average predictions at inference

#### Architecture Ensemble
- Train different architectures (ResNet3D, X3D, SlowFast)
- Combine predictions via voting or averaging

#### Multi-crop Ensemble
- Different spatial crops of the same video
- Average predictions

## Quick Start Guide

### Step 1: Train with Improved Architecture
```bash
python improved_training.py
```

This will:
1. Load data with augmentation
2. Train ImprovedResNet3D with attention
3. Use gradual unfreezing strategy
4. Apply focal loss for class imbalance
5. Use mixup augmentation
6. Apply test-time augmentation
7. Save best model based on validation accuracy

### Step 2: Compare Different Architectures

```python
# Try different models
models_to_try = ['improved', 'multiscale', 'auxiliary']

for model_type in models_to_try:
    model, val_acc, test_acc = train_with_gradual_unfreezing(
        model_class=model_type,
        num_epochs=30,
        use_focal_loss=True,
        use_mixup=True,
        use_tta=True,
        save_path=f'best_{model_type}_model.pth'
    )
    print(f"{model_type}: Val={val_acc:.2f}%, Test={test_acc:.2f}%")
```

### Step 3: Hyperparameter Tuning

Focus on these in order:
1. **Learning rate**: Most important (try 5e-5, 1e-4, 5e-4)
2. **Dropout**: Prevents overfitting (try 0.3, 0.5, 0.7)
3. **Batch size**: Larger is often better if GPU allows
4. **Number of frames**: More frames = more temporal context (try 16, 32, 64)
5. **Focal loss gamma**: Higher = more focus on hard examples (try 1.0, 2.0, 3.0)

### Step 4: Analyze Results

```python
# After training, analyze confusion matrix
# Identify which class has lower performance
# Adjust class weights or collect more data for that class

# Check attention maps (if using ImprovedResNet3D)
# Visualize which frames and regions the model focuses on
# Verify it's looking at relevant anatomical structures
```

## Expected Improvements

Based on these techniques, you should expect:
- **5-15% accuracy improvement** from better architecture
- **3-8% improvement** from data augmentation
- **2-5% improvement** from better training strategy
- **1-3% improvement** from test-time augmentation
- **Overall: 10-30% potential improvement** depending on your baseline

## Troubleshooting

### If accuracy is still low:

1. **Check data quality**
   - Verify labels are correct
   - Check for corrupted videos
   - Ensure train/val/test splits are representative

2. **Analyze errors**
   - Look at confusion matrix
   - Visualize misclassified examples
   - Check if errors are systematic

3. **Consider domain-specific features**
   - Macula detection might need region-specific processing
   - Consider adding segmentation as auxiliary task
   - Consult with medical experts on relevant features

4. **Try different backbones**
   - X3D (more efficient)
   - SlowFast (dual-pathway for temporal modeling)
   - I3D (inflated 2D convolutions)

5. **Collect more data**
   - If possible, increase dataset size
   - Consider data from multiple sources
   - Use semi-supervised learning if unlabeled data available

## Next Steps

1. **Start with ImprovedResNet3D**: Best balance of performance and complexity
2. **Monitor training carefully**: Use TensorBoard or similar for visualization
3. **Iterate based on results**: Adjust hyperparameters based on validation performance
4. **Consider ensemble**: If single model plateaus, combine multiple models
5. **Consult domain experts**: Ensure model is learning clinically relevant features

Good luck with your classifier improvement!
