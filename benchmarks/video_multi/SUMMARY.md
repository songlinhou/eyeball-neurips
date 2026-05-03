# Multi-Class Video Classification Benchmark - Summary

## What Was Created

A complete benchmark system for **hierarchical multi-class video classification** on retinal ultrasound videos, similar to the binary classification benchmark but adapted for multi-class tasks.

## Key Features

✅ **Hierarchical Multi-Class Classification**
- Diagnostic class: 2 classes (non_rd, rd)
- Subtype: 4 classes (macula_detached, macula_intact, normal, pvd)

✅ **Reproducible Data Splits**
- 300 training samples
- 100 test samples
- Stratified by subtype
- Fixed random seed (42)

✅ **Multiple Model Architectures**
- ResNet3D (baseline)
- I3D (inflated 3D)
- MViT (vision transformer)
- Explainable (with attention from exp07)

✅ **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1
- Confusion matrices for both tasks
- Training curves and visualizations
- Automated comparison and reporting

## Files Created

### Core Implementation (5 files)

1. **`prepare_splits.py`** (155 lines)
   - Creates reproducible train/test splits from metadata.csv
   - Stratified sampling by subtype
   - Saves label mappings and statistics

2. **`multiclass_dataset.py`** (190 lines)
   - Multi-class video dataset with hierarchical labels
   - Data augmentation pipeline
   - Automatic class weight computation
   - Dataloader factory function

3. **`multiclass_models.py`** (185 lines)
   - Multi-class versions of video models
   - Dual classification heads (diagnostic + subtype)
   - Integration with main multiclass_model.py
   - Model factory function

4. **`train_benchmark.py`** (465 lines)
   - Complete training pipeline
   - Two-phase training strategy
   - Multi-task loss computation
   - Comprehensive logging and evaluation

5. **`compare_results.py`** (270 lines)
   - Results analysis and visualization
   - Comparison tables and plots
   - Automated report generation

### Documentation (4 files)

6. **`README.md`** - Comprehensive usage guide
7. **`INDEX.md`** - Complete index and reference
8. **`SUMMARY.md`** - This file
9. **`requirements.txt`** - Python dependencies

### Automation Scripts (2 files)

10. **`run_quick_test.sh`** - Quick test (1 model, 5 epochs)
11. **`run_full_benchmark.sh`** - Full benchmark (all models, 20 epochs)

### Package File (1 file)

12. **`__init__.py`** - Package initialization

## Total: 12 Files Created

## Quick Start

### 1. Prepare Splits (30 seconds)
```bash
cd /home/ray/research/eyeball-llm/eyeball-neurips/benchmarks/video_multi
python prepare_splits.py
```

### 2. Run Quick Test (10-15 minutes)
```bash
./run_quick_test.sh
```

### 3. Run Full Benchmark (2-4 hours)
```bash
./run_full_benchmark.sh
```

### 4. Analyze Results (1 minute)
```bash
python compare_results.py
```

## Architecture Overview

```
Input Video (B, C, T, H, W)
         ↓
   Backbone Network
    (ResNet3D/I3D/MViT)
         ↓
   Shared Features
         ↓
    ┌────┴────┐
    ↓         ↓
Diagnostic  Subtype
Classifier  Classifier
    ↓         ↓
  (2 cls)   (4 cls)
```

## Training Pipeline

```
Phase 1 (5 epochs):
  - Freeze backbone
  - Train classifiers only
  - LR = 10 × base_lr
  
Phase 2 (15 epochs):
  - Unfreeze backbone
  - End-to-end training
  - LR = base_lr
  - Early stopping
```

## Comparison with Binary Benchmark

### Similarities
- Same training strategy (2-phase)
- Same model backbones
- Similar data augmentation
- Comparable evaluation metrics

### Differences
- **Multi-class outputs** vs single binary output
- **Multi-task loss** vs single loss
- **Random stratified split** vs pre-defined splits
- **Fixed dataset size** (300/100) vs variable
- **Hierarchical evaluation** vs binary metrics

## Integration with Existing Code

The benchmark integrates seamlessly with:

1. **`model/multiclass_model.py`**
   - Uses `MultiClassExplainableResNet3D`
   - Includes attention from exp07_improved_lower_dropout

2. **`erdes/metadata.csv`**
   - Source of all video metadata
   - Contains diagnostic_class and subtype labels

3. **Binary benchmark patterns**
   - Similar file structure
   - Consistent naming conventions
   - Compatible evaluation framework

## Expected Outputs

After running the full benchmark, you'll have:

### Results Directory
```
results/
├── models/
│   ├── resnet3d_best.pth
│   ├── i3d_best.pth
│   ├── mvit_best.pth
│   └── explainable_best.pth
│
├── logs/
│   ├── {model}_training.log
│   └── {model}_history.json
│
├── plots/
│   ├── {model}_history.png
│   ├── {model}_confusion_matrices.png
│   ├── model_comparison.png
│   └── accuracy_comparison.png
│
├── benchmark_results.json
├── comparison_table.csv
└── BENCHMARK_REPORT.md
```

### Splits Directory
```
splits/
├── train.txt                  # 300 samples
├── test.txt                   # 100 samples
├── label_mappings.json        # Class mappings
└── split_statistics.json      # Distribution stats
```

## Performance Metrics

For each model, you get:

### Diagnostic Classification
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Confusion Matrix (2×2)

### Subtype Classification
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Confusion Matrix (4×4)

### Training Info
- Number of parameters
- Training time (minutes)
- Training curves (loss, accuracy)

## Example Output

```
================================================================================
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

--- PHASE 2: Fine-tuning entire model ---
Epoch 6: Train Loss=0.8234, Diag Acc=82.00%, Diag F1=0.820, ...

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

Model saved to: ./results/models/resnet3d_best.pth
```

## Next Steps

1. **Run the benchmark**
   ```bash
   ./run_full_benchmark.sh
   ```

2. **Analyze results**
   ```bash
   python compare_results.py
   ```

3. **Compare with binary benchmark**
   - Check `../video_binary/` for comparison
   - Evaluate multi-class vs binary performance

4. **Experiment with configurations**
   - Try different hyperparameters
   - Test additional models
   - Adjust dataset size

5. **Integrate best model**
   - Use best performing model for downstream tasks
   - Export for deployment
   - Create inference pipeline

## Technical Details

### Data Format
- Videos: MP4 format
- Labels: CSV with diagnostic_class and subtype
- Splits: Text files with paths and integer labels

### Label Encoding
- Diagnostic: non_rd=0, rd=1
- Subtype: macula_detached=0, macula_intact=1, normal=2, pvd=3

### Class Weights
Automatically computed as:
```python
weight[i] = 1.0 / count[i]
normalized_weight = weight / sum(weight)
```

### Multi-Task Loss
```python
loss = loss_diagnostic + loss_subtype
```
Both use weighted CrossEntropyLoss.

## Reproducibility Checklist

✅ Fixed random seed (42)
✅ Deterministic data splits
✅ Consistent preprocessing
✅ Same hyperparameters
✅ Version-controlled code
✅ Documented dependencies

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch_size or num_frames
```bash
python train_benchmark.py --batch_size 4 --num_frames 16
```

### Issue: Slow data loading
**Solution**: Increase num_workers
```bash
python train_benchmark.py --num_workers 8
```

### Issue: Poor performance
**Solution**: Train longer or adjust learning rate
```bash
python train_benchmark.py --num_epochs 30 --learning_rate 5e-5
```

## Validation

The benchmark has been designed to:
- ✅ Match binary benchmark structure
- ✅ Use same training strategies
- ✅ Integrate with existing models
- ✅ Provide reproducible results
- ✅ Generate comprehensive reports

## Credits

Based on:
- Binary benchmark: `benchmarks/video_binary/`
- Multi-class model: `model/multiclass_model.py`
- Attention mechanism: exp07_improved_lower_dropout

## License

Same as parent project.

---

**Ready to use!** Run `./run_quick_test.sh` to get started.
