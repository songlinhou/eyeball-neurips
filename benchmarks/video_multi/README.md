# Multi-Class Video Classification Benchmark

This benchmark trains and evaluates video classification models on hierarchical diagnosis classification tasks.

## Overview

Unlike the binary classification benchmark (`video_binary`), this benchmark handles **multi-class hierarchical classification**:

- **Diagnostic Class**: Primary diagnosis (2 classes)
  - `non_rd` (0): Non-retinal detachment
  - `rd` (1): Retinal detachment

- **Subtype**: Detailed subtype classification (4 classes)
  - `macula_detached` (0): Macula detached
  - `macula_intact` (1): Macula intact
  - `normal` (2): Normal
  - `pvd` (3): Posterior vitreous detachment

## Dataset

The benchmark uses a stratified random split from `metadata.csv`:
- **Training**: 300 samples
- **Testing**: 100 samples
- **Random seed**: 42 (for reproducibility)

The split is stratified by subtype to ensure balanced representation across all classes.

## Models

The benchmark supports **9 video classification models** adapted for multi-class outputs:

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
9. **Explainable** - Multi-class ExplainableResNet3D with:
   - Temporal & spatial attention (from exp07_improved_lower_dropout)
   - Optical flow extraction
   - Frame importance module
   - Dual classification heads

Each model outputs two classification heads:
- Diagnostic classifier (2 classes)
- Subtype classifier (4 classes)

## Quick Start

### 1. Prepare Splits

First, create the train/test splits:

```bash
cd /home/ray/research/eyeball-llm/eyeball-neurips/benchmarks/video_multi
python prepare_splits.py --metadata ../../erdes/metadata.csv --output_dir ./splits
```

This will create:
- `splits/train.txt`: Training split (300 samples)
- `splits/test.txt`: Test split (100 samples)
- `splits/label_mappings.json`: Label mappings
- `splits/split_statistics.json`: Split statistics

### 2. Run Benchmark

Train and evaluate all models:

```bash
python train_benchmark.py \
    --data_dir ../../erdes \
    --splits_dir ./splits \
    --save_dir ./results \
    --num_epochs 20 \
    --batch_size 8
```

Train specific models:

```bash
python train_benchmark.py \
    --models resnet3d i3d slowfast \
    --num_epochs 20 \
    --batch_size 8
```

### 3. View Results

Results are saved to `./results/`:
- `models/`: Trained model weights
- `logs/`: Training logs and metrics
- `plots/`: Training curves and confusion matrices
- `benchmark_results.json`: Summary of all results

## Configuration

Key parameters:

- `--num_frames`: Number of frames per video (default: 32)
- `--img_size`: Image size (default: 224)
- `--batch_size`: Batch size (default: 8)
- `--num_epochs`: Training epochs (default: 20)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--dropout`: Dropout rate (default: 0.5)

## Training Strategy

The benchmark uses a two-phase training strategy:

**Phase 1**: Train classifier heads only (5 epochs)
- Freeze backbone
- Higher learning rate (10x)
- Fast convergence on classification heads

**Phase 2**: Fine-tune entire model
- Unfreeze backbone
- Normal learning rate
- Early stopping with patience=7

## Evaluation Metrics

For each classification task (diagnostic and subtype):

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision across classes
- **Recall**: Weighted recall across classes
- **F1 Score**: Weighted F1 score
- **Confusion Matrix**: Per-class performance

## File Structure

```
video_multi/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── __init__.py                  # Package initialization
├── prepare_splits.py            # Create train/test splits
├── multiclass_dataset.py        # Dataset and dataloader
├── multiclass_models.py         # Multi-class model definitions
├── train_benchmark.py           # Main training script
├── splits/                      # Generated splits
│   ├── train.txt
│   ├── test.txt
│   ├── label_mappings.json
│   └── split_statistics.json
└── results/                     # Training results
    ├── models/
    ├── logs/
    ├── plots/
    └── benchmark_results.json
```

## Comparison with Binary Benchmark

| Feature | Binary (`video_binary`) | Multi-Class (`video_multi`) |
|---------|------------------------|----------------------------|
| Task | Binary classification | Hierarchical multi-class |
| Classes | 2 (intact/detached) | 2 diagnostic + 4 subtype |
| Output | Single classifier | Two classifiers |
| Loss | Single CrossEntropy | Multi-task loss |
| Dataset | Pre-split | Random stratified split |
| Samples | Variable | 300 train / 100 test |

## Example Output

```
MULTI-CLASS VIDEO CLASSIFICATION BENCHMARK
================================================================================
Data directory: ../../erdes
Splits directory: ./splits
Save directory: ./results
Device: cuda
================================================================================

Label mappings loaded:
  Diagnostic classes: ['non_rd', 'rd']
  Subtypes: ['macula_detached', 'macula_intact', 'normal', 'pvd']

Loading dataset...
Dataset loaded: Train=300, Test=100
Diagnostic class weights: [0.4 0.6]
Subtype class weights: [0.2 0.15 0.55 0.1]

================================================================================
Training RESNET3D
================================================================================

--- PHASE 1: Training classifier heads (5 epochs) ---
Epoch 1: Train Loss=1.2345, Diag Acc=75.00%, Subtype Acc=60.00% | ...
...

--- FINAL EVALUATION ON TEST SET ---

Diagnostic Classification:
  Accuracy: 85.00%
  Precision: 0.850
  Recall: 0.850
  F1 Score: 0.850

Subtype Classification:
  Accuracy: 72.00%
  Precision: 0.720
  Recall: 0.720
  F1 Score: 0.720
```

## Notes

- The benchmark uses **stratified sampling** to ensure balanced class representation
- **Class weights** are automatically computed based on training set distribution
- **Early stopping** prevents overfitting
- All results are **reproducible** with fixed random seed (42)
- Models are saved with best validation performance

## Citation

If you use this benchmark, please cite the corresponding paper and reference the multi-class model architecture from `model/multiclass_model.py`.
