# Experiment Documentation

## Overview

This document describes the comprehensive experiments designed to find the best model for ocular ultrasound video classification (macula intact vs. detached).

## Experiment Setup

### Save Location
All results are permanently saved to:
```
/content/drive/MyDrive/EyeballProject/classifier_experiment/
```

### Directory Structure
```
classifier_experiment/
├── models/              # Saved model checkpoints (.pth files)
├── logs/                # Training logs and metrics (.log, .json)
├── plots/               # Visualization plots (.png)
├── results/             # Summary results (.csv, .json)
└── EXPERIMENT_REPORT.md # Comprehensive report
```

## Experiments

### Experiment 1: Improved ResNet3D (Baseline)
- **Name:** `exp01_improved_baseline`
- **Model:** ImprovedResNet3D with attention
- **Frames:** 32
- **Batch Size:** 8
- **Learning Rate:** 1e-4
- **Dropout:** 0.5
- **Features:** Focal loss, Mixup, TTA
- **Goal:** Establish baseline with all improvements

### Experiment 2: MultiScale ResNet3D
- **Name:** `exp02_multiscale`
- **Model:** MultiScaleResNet3D
- **Frames:** 32
- **Batch Size:** 8
- **Learning Rate:** 1e-4
- **Dropout:** 0.5
- **Features:** Multi-scale feature extraction
- **Goal:** Test multi-scale architecture

### Experiment 3: Auxiliary Task Learning
- **Name:** `exp03_auxiliary`
- **Model:** ResNet3DWithAuxiliary
- **Frames:** 32
- **Batch Size:** 8
- **Learning Rate:** 1e-4
- **Dropout:** 0.5
- **Features:** Auxiliary classifier
- **Goal:** Test auxiliary task learning

### Experiment 4: Fewer Frames (16)
- **Name:** `exp04_improved_16frames`
- **Model:** ImprovedResNet3D
- **Frames:** 16 (vs 32)
- **Batch Size:** 16 (doubled)
- **Learning Rate:** 1e-4
- **Dropout:** 0.5
- **Goal:** Test if fewer frames work well (faster training)

### Experiment 5: More Frames (64)
- **Name:** `exp05_improved_64frames`
- **Model:** ImprovedResNet3D
- **Frames:** 64 (vs 32)
- **Batch Size:** 4 (halved due to memory)
- **Learning Rate:** 1e-4
- **Dropout:** 0.5
- **Goal:** Test if more frames improve accuracy

### Experiment 6: Higher Learning Rate
- **Name:** `exp06_improved_higher_lr`
- **Model:** ImprovedResNet3D
- **Frames:** 32
- **Batch Size:** 8
- **Learning Rate:** 5e-4 (5x higher)
- **Dropout:** 0.5
- **Goal:** Test faster convergence

### Experiment 7: Lower Dropout
- **Name:** `exp07_improved_lower_dropout`
- **Model:** ImprovedResNet3D
- **Frames:** 32
- **Batch Size:** 8
- **Learning Rate:** 1e-4
- **Dropout:** 0.3 (vs 0.5)
- **Goal:** Test if less regularization helps

## Training Strategy

All experiments use:

### Phase 1: Classifier Head Training (5 epochs)
- Freeze backbone
- Train attention + classifier only
- Learning rate: 10x base LR
- Scheduler: CosineAnnealingWarmRestarts

### Phase 2: Full Fine-tuning (20 epochs)
- Unfreeze entire model
- Learning rate: base LR
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=7

### Data Augmentation
- Temporal jittering
- Horizontal flip
- Small rotation (±10°)
- Brightness/contrast adjustment
- Gaussian noise
- Mixup (50% probability)

### Loss Function
- Focal Loss with class weights
- Gamma: 2.0
- Handles class imbalance

### Test-Time Augmentation
- Original + Horizontal flip
- Average predictions

## Metrics Tracked

For each experiment, we track:

### Training Metrics (per epoch)
- Loss
- Accuracy
- F1 Score
- AUC-ROC

### Validation Metrics (per epoch)
- Loss
- Accuracy
- F1 Score
- AUC-ROC

### Test Metrics (final)
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
- Confusion Matrix

## Output Files

### Per Experiment
- `{experiment_name}.log` - Training log
- `{experiment_name}_metrics.json` - Metrics history
- `{experiment_name}_history.png` - Training curves
- `{experiment_name}_confusion_matrix.png` - Confusion matrix
- `{experiment_name}_best.pth` - Best model checkpoint

### Summary Files
- `experiment_summary.csv` - All results in table format
- `experiment_results_detailed.json` - Detailed results
- `EXPERIMENT_REPORT.md` - Comprehensive report
- `accuracy_comparison.png` - Accuracy comparison plot
- `metrics_comparison.png` - Multi-metric comparison

## How to Run

### Run All Experiments
```bash
cd /content/improved_classifier
python run_experiments.py
```

This will:
1. Run all 7 experiments sequentially
2. Save all results to Google Drive
3. Generate comprehensive report
4. Create comparison plots

**Estimated time:** 8-12 hours (depends on GPU)

### Monitor Progress
Check logs in real-time:
```bash
tail -f /content/drive/MyDrive/EyeballProject/classifier_experiment/logs/exp01_improved_baseline.log
```

### View Results
After completion, check:
```bash
cat /content/drive/MyDrive/EyeballProject/classifier_experiment/EXPERIMENT_REPORT.md
```

## Expected Outcomes

### Best Model Selection
The best model will be selected based on:
1. **Primary:** Test Accuracy
2. **Secondary:** Test F1 Score
3. **Tertiary:** Test AUC

### Insights to Gain
- Which architecture works best?
- Optimal number of frames?
- Best learning rate?
- Effect of dropout?
- Training time vs accuracy tradeoff

## Troubleshooting

### Out of Memory
If GPU runs out of memory:
1. Reduce batch size in experiment config
2. Reduce number of frames
3. Skip 64-frame experiment

### Slow Training
If training is too slow:
1. Reduce num_workers to 2
2. Reduce number of epochs to 15
3. Skip some experiments

### Dataset Not Found
Update DATA_DIR in run_experiments.py:
```python
DATA_DIR = "/path/to/your/erdes"
```

## Post-Experiment Analysis

After experiments complete:

1. **Review Report**
   - Check `EXPERIMENT_REPORT.md`
   - Identify best model

2. **Analyze Plots**
   - Training curves for overfitting
   - Confusion matrices for error patterns
   - Comparison plots for trends

3. **Select Best Model**
   - Based on test accuracy and F1
   - Consider training time
   - Check generalization (train vs val gap)

4. **Further Experiments** (if needed)
   - Fine-tune hyperparameters of best model
   - Try ensemble of top 3 models
   - Collect more data if accuracy insufficient

## Notes

- All files in `/content/` are temporary
- Only files in `/content/drive/MyDrive/` persist
- Experiments run sequentially (not parallel)
- Each experiment is independent
- Failed experiments are logged but don't stop others

Good luck with your experiments! 🚀
