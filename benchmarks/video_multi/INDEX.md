# Multi-Class Video Classification Benchmark - Index

## Overview

This benchmark evaluates video classification models on **hierarchical multi-class diagnosis** tasks using retinal ultrasound videos.

## Key Differences from Binary Benchmark

| Aspect | Binary Benchmark | Multi-Class Benchmark |
|--------|------------------|----------------------|
| **Location** | `benchmarks/video_binary/` | `benchmarks/video_multi/` |
| **Task** | Binary classification (intact vs detached) | Hierarchical multi-class |
| **Output** | Single classifier | Two classifiers (diagnostic + subtype) |
| **Classes** | 2 classes | 2 diagnostic + 4 subtype classes |
| **Dataset** | Pre-defined splits | Random stratified split (reproducible) |
| **Samples** | Variable by split | 300 train / 100 test |
| **Model Architecture** | Single-head models | Multi-head models |

## Classification Tasks

### Task 1: Diagnostic Classification (2 classes)
- **non_rd** (0): Non-retinal detachment
- **rd** (1): Retinal detachment

### Task 2: Subtype Classification (4 classes)
- **macula_detached** (0): Macula detached
- **macula_intact** (1): Macula intact  
- **normal** (2): Normal
- **pvd** (3): Posterior vitreous detachment

## Files and Scripts

### Core Scripts

1. **`prepare_splits.py`** - Create reproducible train/test splits
   - Stratified sampling by subtype
   - 300 training / 100 test samples
   - Fixed random seed (42)

2. **`multiclass_dataset.py`** - Dataset and dataloader implementation
   - Loads videos with hierarchical labels
   - Data augmentation for training
   - Automatic class weight computation

3. **`multiclass_models.py`** - Multi-class model definitions
   - Adapted from binary models
   - Dual classification heads
   - Shared feature extraction

4. **`train_benchmark.py`** - Main training and evaluation script
   - Two-phase training strategy
   - Multi-task loss
   - Comprehensive metrics

5. **`compare_results.py`** - Results analysis and visualization
   - Comparison tables
   - Performance plots
   - Markdown reports

### Shell Scripts

- **`run_quick_test.sh`** - Quick test (1 model, 5 epochs)
- **`run_full_benchmark.sh`** - Full benchmark (all models, 20 epochs)

### Documentation

- **`README.md`** - Detailed usage guide
- **`INDEX.md`** - This file
- **`requirements.txt`** - Python dependencies

## Supported Models (9 Total)

### Baseline Models (8)
1. **ResNet3D** - Baseline 3D ResNet (CVPR 2018)
2. **I3D** - Inflated 3D ConvNet (CVPR 2017)
3. **SlowFast** - Dual-pathway architecture (ICCV 2019)
4. **X3D** - Efficient video network (CVPR 2020)
5. **MViT** - Multiscale Vision Transformer (ICCV 2021)
6. **VideoMAE** - Masked autoencoder (NeurIPS 2022)
7. **TimeSformer** - Space-time attention (ICML 2021)
8. **C3D** - Classic 3D CNN (ICCV 2015)

### Proposed Method (1)
9. **Explainable** - Multi-class ExplainableResNet3D with temporal/spatial attention, optical flow, and frame importance modules

All models are adapted with dual classification heads for hierarchical diagnosis.

## Workflow

### Step 1: Prepare Data Splits

```bash
python prepare_splits.py \
    --metadata ../../erdes/metadata.csv \
    --output_dir ./splits \
    --train_size 300 \
    --test_size 100 \
    --seed 42
```

**Output:**
- `splits/train.txt` - Training samples with labels
- `splits/test.txt` - Test samples with labels
- `splits/label_mappings.json` - Class mappings
- `splits/split_statistics.json` - Distribution statistics

### Step 2: Train Models

**Quick test (fast):**
```bash
./run_quick_test.sh
```

**Full benchmark:**
```bash
./run_full_benchmark.sh
```

**Custom configuration:**
```bash
python train_benchmark.py \
    --models resnet3d i3d slowfast \
    --num_epochs 20 \
    --batch_size 8 \
    --num_frames 32
```

### Step 3: Analyze Results

```bash
python compare_results.py --results_dir ./results
```

**Output:**
- `comparison_table.csv` - Performance comparison
- `model_comparison.png` - Comparison plots
- `accuracy_comparison.png` - Accuracy comparison
- `BENCHMARK_REPORT.md` - Detailed report

## Directory Structure

```
video_multi/
├── README.md                      # Usage guide
├── INDEX.md                       # This file
├── requirements.txt               # Dependencies
├── __init__.py                    # Package init
│
├── prepare_splits.py              # Data preparation
├── multiclass_dataset.py          # Dataset implementation
├── multiclass_models.py           # Model definitions
├── train_benchmark.py             # Training script
├── compare_results.py             # Results analysis
│
├── run_quick_test.sh              # Quick test script
├── run_full_benchmark.sh          # Full benchmark script
│
├── splits/                        # Generated splits
│   ├── train.txt
│   ├── test.txt
│   ├── label_mappings.json
│   └── split_statistics.json
│
└── results/                       # Training results
    ├── models/                    # Model weights
    │   ├── resnet3d_best.pth
    │   ├── i3d_best.pth
    │   ├── slowfast_best.pth
    │   ├── x3d_best.pth
    │   ├── mvit_best.pth
    │   ├── videomae_best.pth
    │   ├── timesformer_best.pth
    │   ├── c3d_best.pth
    │   └── explainable_best.pth    # PROPOSED METHOD
    │
    ├── logs/                      # Training logs
    │   ├── resnet3d_training.log
    │   ├── resnet3d_history.json
    │   └── ...
    │
    ├── plots/                     # Visualizations
    │   ├── resnet3d_history.png
    │   ├── resnet3d_confusion_matrices.png
    │   ├── model_comparison.png
    │   └── accuracy_comparison.png
    │
    ├── benchmark_results.json     # Results summary
    ├── comparison_table.csv       # Performance table
    └── BENCHMARK_REPORT.md        # Analysis report
```

## Training Strategy

### Phase 1: Classifier Head Training (5 epochs)
- Freeze backbone weights
- Train only classification heads
- Higher learning rate (10x base LR)
- Fast convergence on task-specific layers

### Phase 2: Full Model Fine-tuning (15 epochs)
- Unfreeze all weights
- End-to-end training
- Normal learning rate
- Early stopping (patience=7)

### Multi-Task Loss
```
Total Loss = Loss_diagnostic + Loss_subtype
```

Both losses use weighted CrossEntropy based on class distribution.

## Evaluation Metrics

For each task (diagnostic and subtype):

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall  
- **F1 Score**: Weighted average F1
- **Confusion Matrix**: Per-class performance breakdown

## Example Results Format

```json
{
  "model_name": "resnet3d",
  "status": "completed",
  "num_params": 33169922,
  "diagnostic_acc": 85.00,
  "diagnostic_precision": 0.850,
  "diagnostic_recall": 0.850,
  "diagnostic_f1": 0.850,
  "subtype_acc": 72.00,
  "subtype_precision": 0.720,
  "subtype_recall": 0.720,
  "subtype_f1": 0.720,
  "training_time_minutes": 45.2
}
```

## Reproducibility

All experiments are **fully reproducible**:

- Fixed random seed: 42
- Deterministic data splits
- Consistent preprocessing
- Same hyperparameters across runs

## Integration with Main Model

The benchmark uses the multi-class model from:
```
model/multiclass_model.py
```

Specifically, the `MultiClassExplainableResNet3D` class with attention mechanisms added from `exp07_improved_lower_dropout`.

## Performance Expectations

Based on similar benchmarks:

| Model | Diagnostic Acc | Subtype Acc | Training Time | Type |
|-------|---------------|-------------|---------------|------|
| ResNet3D | 80-85% | 65-75% | ~30-40 min | Baseline |
| I3D | 82-87% | 68-78% | ~35-45 min | Baseline |
| SlowFast | 83-88% | 70-78% | ~50-60 min | Baseline |
| X3D | 81-86% | 67-76% | ~30-40 min | Baseline |
| MViT | 85-90% | 72-80% | ~50-60 min | Baseline |
| VideoMAE | 84-89% | 71-79% | ~45-55 min | Baseline |
| TimeSformer | 86-91% | 73-81% | ~60-75 min | Baseline |
| C3D | 78-83% | 63-73% | ~40-50 min | Baseline |
| **Explainable** | **87-93%** | **75-85%** | ~60-70 min | **PROPOSED** |

*Note: Actual performance depends on hardware and exact configuration.*
*The proposed method is expected to outperform baselines due to attention mechanisms and optical flow.*

## Common Issues

### 1. Out of Memory
- Reduce `--batch_size` (try 4 or 2)
- Reduce `--num_frames` (try 16)
- Use smaller models first

### 2. Slow Training
- Increase `--num_workers` (try 8)
- Use smaller `--img_size` (try 112)
- Enable mixed precision training

### 3. Poor Performance
- Increase `--num_epochs` (try 30)
- Adjust `--learning_rate` (try 5e-5 or 2e-4)
- Check class balance in splits

## Citation

If you use this benchmark, please cite:

```bibtex
@article{eyeball2026,
  title={Multi-Class Video Classification for Retinal Diagnosis},
  author={...},
  journal={...},
  year={2026}
}
```

## Contact

For questions or issues, please open an issue in the repository.
