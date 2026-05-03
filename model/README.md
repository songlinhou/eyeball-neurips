# Experiment 10: Explainable Flow Lower Dropout

This directory contains the extracted code for the **exp10_explainable_flow_lower_dropout** experiment from the video classification project.

## Overview

This experiment uses an **ExplainableOpticalFlowResNet3D** model with:
- **RGB stream with explainability**: Frame importance + spatial attention modules
- **Optical flow stream**: Captures motion information for better temporal understanding
- **Feature fusion**: Combines RGB and flow features for robust classification
- **Lower dropout** (0.3) for better feature learning
- **Standard learning rate** (1e-4) for stable convergence

## Files

### Configuration
- **`config.py`**: All experiment configurations and hyperparameters
  - Model settings (dropout=0.3, pretrained=True)
  - Data settings (32 frames, 224x224 images, batch_size=16)
  - Training settings (10 epochs, lr=1e-4, focal loss)
  - Augmentation settings (mixup, TTA enabled)

### Model
- **`model.py`**: ExplainableOpticalFlowResNet3D model implementation
  - `OpticalFlowExtractor`: Lightweight flow estimation network
  - `FrameImportanceModule`: Temporal attention for frame selection
  - `SpatialExplainabilityModule`: Spatial attention maps
  - `ExplainableResNet3D`: Main model combining RGB + flow + explainability

### Training
- **`exp13_train.py`**: Complete training pipeline
  - Two-phase training (classifier head → full fine-tuning)
  - Focal loss for class imbalance
  - Mixup data augmentation
  - Early stopping and checkpointing
  - Comprehensive logging and visualization

### Evaluation
- **`evaluate_classifier.py`**: Detailed evaluation and analysis
  - ROC and Precision-Recall curves
  - Confusion matrix with percentages
  - Frame importance visualization
  - Classification report generation

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 2. Train the Model

```bash
cd /home/ray/research/eyeball-llm/eyeball-neurips/model
python exp13_train.py
```

This will:
- Load the ERDES dataset (macula detached vs intact)
- Train the ExplainableOpticalFlowResNet3D model for 10 epochs
- Save checkpoints, logs, and plots to `./exp10_results/`
- Evaluate on test set with TTA (test-time augmentation)

### 3. Evaluate the Model

```bash
python evaluate_classifier.py
```

Or evaluate a specific checkpoint:

```bash
python evaluate_classifier.py --model_path ./exp10_results/checkpoints/exp10_phase2_best_epoch8.pth
```

## Configuration Details

### Model Architecture
```python
ExplainableOpticalFlowResNet3D(
    num_classes=2,
    pretrained=True,  # Kinetics-400 pretrained weights for RGB stream
    dropout=0.3       # Lower dropout for better learning
)

# Architecture:
# RGB Stream: R3D-18 → Frame Importance → Spatial Attention → Features (512)
# Flow Stream: OpticalFlowExtractor → Conv3D layers → Features (256)
# Fusion: Concat(RGB, Flow) → Linear(768→512) → Classifier(512→256→2)
```

### Training Strategy

**Phase 1** (3 epochs): Classifier head training
- Freeze RGB backbone (flow extractor remains trainable)
- Learning rate: 1e-3 (10× base LR)
- Scheduler: CosineAnnealingWarmRestarts
- No mixup

**Phase 2** (7 epochs): Full fine-tuning
- Unfreeze all layers
- Learning rate: 1e-4
- Scheduler: ReduceLROnPlateau
- Mixup enabled (α=0.2, p=0.5)
- Early stopping (patience=7)

### Loss Function
```python
FocalLoss(
    alpha=class_weights,  # Handle class imbalance
    gamma=2.0             # Focus on hard examples
)
```

### Data Augmentation
- Temporal augmentation (frame sampling)
- Spatial augmentation (random crops, flips)
- Mixup (α=0.2, probability=0.5)
- Test-time augmentation (horizontal flip averaging)

## Output Structure

```
exp10_results/
├── models/
│   └── exp10_explainable_flow_lower_dropout_best.pth
├── checkpoints/
│   ├── exp10_phase1_best_epoch3.pth
│   ├── exp10_phase2_best_epoch8.pth
│   └── exp10_epoch5.pth
├── logs/
│   ├── exp10_explainable_flow_lower_dropout.log
│   └── exp10_explainable_flow_lower_dropout_metrics.json
├── plots/
│   ├── exp10_explainable_flow_lower_dropout_history.png
│   └── exp10_explainable_flow_lower_dropout_confusion_matrix.png
├── results/
│   └── exp10_explainable_flow_lower_dropout_results.json
└── evaluation/
    ├── exp10_explainable_flow_lower_dropout_roc_curve.png
    ├── exp10_explainable_flow_lower_dropout_pr_curve.png
    ├── exp10_explainable_flow_lower_dropout_confusion_matrix_detailed.png
    ├── exp10_explainable_flow_lower_dropout_frame_importance.png
    ├── exp10_explainable_flow_lower_dropout_classification_report.txt
    └── exp10_explainable_flow_lower_dropout_evaluation_results.json
```

## Key Features

### 1. Explainability
The model provides interpretable outputs:
- **Frame importance scores**: Which frames contribute most to the decision
- **Spatial attention maps**: Which regions in each frame are important

Access attention maps:
```python
from model import ExplainableResNet3D

model = ExplainableResNet3D(num_classes=2, pretrained=True, dropout=0.3)
model.eval()

# Forward with attention
outputs, attention = model(videos, return_attention=True)

frame_importance = attention['frame_importance']  # (B, T)
spatial_attention = attention['spatial_attention']  # (B, 1, T, H, W)
```

### 2. Optical Flow
The model captures motion information:
- **Lightweight flow extractor**: Learns temporal differences between frames
- **Flow stream**: Processes motion features independently
- **Feature fusion**: Combines appearance (RGB) and motion (flow) cues

### 3. Two-Phase Training
- **Phase 1**: Quickly adapt pretrained features to medical domain
- **Phase 2**: Fine-tune entire network for optimal performance

### 4. Robust Training
- Focal loss handles class imbalance
- Mixup improves generalization
- Gradient clipping prevents instability
- Early stopping prevents overfitting

### 5. Comprehensive Logging
- Real-time training progress
- Metrics history (loss, accuracy, F1, AUC)
- Automatic plotting and visualization
- Checkpoint saving at key points

## Expected Results

Based on the configuration:
- **Validation Accuracy**: 85-92%
- **Test Accuracy**: 83-90%
- **Test F1 Score**: 0.82-0.90
- **Test AUC**: 0.90-0.97

The explainability features allow you to:
- Identify which frames are most diagnostic
- Visualize important anatomical regions
- Validate model decisions against clinical knowledge

## Customization

### Modify Hyperparameters

Edit `exp13_config.py`:
```python
# Increase training epochs
NUM_EPOCHS = 20

# Adjust learning rate
LEARNING_RATE = 1e-4

# Change dropout
DROPOUT = 0.4

# Disable mixup
USE_MIXUP = False
```

### Use Different Data Split

Edit `exp13_config.py`:
```python
SPLITS_DIR = os.path.join(DATA_DIR, "splits", "non_rd_vs_rd")
```

### Change Batch Size

Edit `exp13_config.py`:
```python
BATCH_SIZE = 8  # For smaller GPU
# or
BATCH_SIZE = 32  # For larger GPU
```

## Troubleshooting

### Out of Memory
Reduce batch size or number of frames:
```python
BATCH_SIZE = 8
NUM_FRAMES = 16
```

### Slow Training
Increase number of workers:
```python
NUM_WORKERS = 4
```

### Poor Performance
- Increase number of epochs
- Adjust learning rate
- Check data quality and class balance
- Try different augmentation strategies

## Dependencies

The code requires access to:
- `../video_classification/improved_dataset.py`: Dataset utilities
- `../erdes/`: ERDES dataset directory

Make sure these are available in the correct locations.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ozkuterdes,
  title={ERDES: A Benchmark Video Dataset for Retinal Detachment and Macular Status Classification in Ocular Ultrasound},
  author={Ozkut, Yasemin and Navard, Pouyan and Adhikari, Srikar and Situ-LaCasse, Elaine and Acu{\~n}a, Josie and Yarnish, Adrienne A and Yilmaz, Alper},
  journal={arXiv preprint arXiv:2508.04735},
  year={2025}
}
```

## License

This code is part of the ERDES project research.
